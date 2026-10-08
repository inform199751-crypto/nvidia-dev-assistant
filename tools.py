"""Agent 工具。原則:小而精準的輸出 + 錯誤回給模型自己修正 + 不直接改檔。"""
from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from langchain_core.tools import tool

WORKSPACE = Path(os.getenv("AGENT_WORKSPACE", ".")).resolve()
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "dev-assistant-sandbox")  # sandbox/Dockerfile 建的(含 pytest)
MAX_READ_LINES = 400
PATCHES: list[tuple[Path, bool]] = []  # 提出過的 patch(檔案, 沙箱測試是否通過);互動模式拿來問要不要套用


def configure(workspace: str | Path | None = None, sandbox_image: str | None = None) -> None:
    """讀完 .env 之後要再呼叫一次 —— 模組載入時 .env 還沒讀,上面兩個值會是預設值。"""
    global WORKSPACE, SANDBOX_IMAGE
    if workspace is not None:
        WORKSPACE = Path(workspace).resolve()
    if sandbox_image:
        SANDBOX_IMAGE = sandbox_image


_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"}
_SECRET_PREFIXES = ("id_rsa", "id_ed25519", "id_ecdsa")


def _is_secret(p: Path) -> bool:
    """金鑰、憑證、.env 一律不給模型看 —— 讀到就等於把金鑰送出公司。"""
    name = p.name.lower()
    if name == ".env" or (name.startswith(".env.") and not name.endswith((".example", ".sample", ".template"))):
        return True
    return p.suffix.lower() in _SECRET_SUFFIXES or name.startswith(_SECRET_PREFIXES)


def _safe_path(rel: str) -> Path:
    p = (WORKSPACE / rel).resolve()
    if not p.is_relative_to(WORKSPACE):
        raise ValueError(f"路徑 {rel} 超出工作區,拒絕存取")
    if _is_secret(p) or ".git" in p.relative_to(WORKSPACE).parts:
        raise PermissionError(f"{rel} 是機密檔(金鑰 / 憑證 / .env / .git),agent 不能讀")
    return p


@tool
def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """讀取工作區檔案的指定行號範圍,每行前面附行號。一次最多 400 行;大檔請先看大綱或 search_code 再分段讀。"""
    lines = _safe_path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(start_line, 1)
    end = min(end_line or len(lines), len(lines), start + MAX_READ_LINES - 1)
    body = "\n".join(f"{n:>5}| {lines[n - 1]}" for n in range(start, end + 1))
    more = f"\n[檔案共 {len(lines)} 行,已顯示 {start}-{end}]" if end < len(lines) or start > 1 else ""
    return body + more


_SKIP_DIRS = {".agent_patches", ".git", ".venv", "__pycache__", "node_modules"}


def _visible(f: Path) -> bool:
    """列檔 / 搜尋看得到的:不是隱藏路徑、不在大型目錄裡、不是機密檔、實際位置在工作區內。"""
    rel = f.relative_to(WORKSPACE)
    if any(part.startswith(".") or part in _SKIP_DIRS for part in rel.parts) or _is_secret(f):
        return False
    real = f.resolve()  # symlink / Windows junction 可能指到工作區外或機密檔,看實際位置
    return real.is_relative_to(WORKSPACE) and not _is_secret(real)


@tool
def list_files(glob: str = "*", max_entries: int = 200) -> str:
    """列出工作區的檔案與資料夾(資料夾結尾加 /,檔案附大小)。整個專案用 "**/*",某資料夾用 "src/*"。"""
    rows: list[str] = []
    for f in sorted(WORKSPACE.glob(glob)):
        if not _visible(f):
            continue
        rel = f.relative_to(WORKSPACE).as_posix()
        rows.append(f"{rel}/" if f.is_dir() else f"{rel}  ({f.stat().st_size:,} bytes)")
        if len(rows) >= max_entries:
            return "\n".join(rows) + f"\n[已達 {max_entries} 筆上限,請把 glob 縮小]"
    return "\n".join(rows) or "沒有符合的檔案"


@tool
def search_code(pattern: str, glob: str = "**/*.py", max_hits: int = 50) -> str:
    """在工作區用正規表示式搜尋,回傳「檔案:行號: 內容」。先搜再讀,不要整檔讀。"""
    rx = re.compile(pattern)
    hits: list[str] = []
    for f in WORKSPACE.glob(glob):
        if not f.is_file() or not _visible(f):
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{f.relative_to(WORKSPACE).as_posix()}:{n}: {line.strip()[:200]}")
                if len(hits) >= max_hits:
                    return "\n".join(hits) + f"\n[已達 {max_hits} 筆上限,請把 pattern 寫得更精確]"
    return "\n".join(hits) or "沒有符合的結果"


def _copy_ignore(directory: str, names: list[str]) -> list[str]:
    """暫存複本不帶機密檔(跟 _safe_path 同一套判斷)、大型目錄、指到工作區外的連結:
    沙箱裡的輸出會回到模型手上。"""
    skipped = []
    for n in names:
        p = Path(directory) / n
        if n in _SKIP_DIRS or _is_secret(p) or not p.resolve().is_relative_to(WORKSPACE):
            skipped.append(n)
    return skipped


@contextmanager
def _sandbox_copy():
    """工作區的暫存複本,沙箱一律掛它而不是工作區本身;正式環境大 repo 改用 git worktree,不要整包複製。"""
    with tempfile.TemporaryDirectory(prefix="agent_sandbox_") as tmp:
        copy = Path(tmp) / "ws"
        shutil.copytree(WORKSPACE, copy, ignore=_copy_ignore)
        yield copy


def _docker(root: Path, argv: list[str], timeout: int = 180) -> str:
    """在隔離的 Docker 容器裡執行:無網路、唯讀掛載、限 CPU / 記憶體 / 行程數 / 時間。"""
    if not shutil.which("docker"):
        return "錯誤:這台機器沒有 Docker,沙箱無法啟動。為了安全,不會改在主機上直接執行程式碼。"
    name = f"agent-sandbox-{uuid.uuid4().hex[:12]}"
    cmd = [
        "docker", "run", "--rm", "--name", name, "--network", "none", "--read-only", "--tmpfs", "/tmp",
        "--memory", "1g", "--cpus", "1", "--pids-limit", "256",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{root}:/ws:ro", "-w", "/ws",
        SANDBOX_IMAGE, *argv,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)  # 只停掉 docker CLI 的話,容器會在背景繼續跑
        return f"錯誤:超過 {timeout} 秒被中止(可能有無窮迴圈或死結)"
    return f"exit code {r.returncode}\n{r.stdout}\n{r.stderr}".strip()


def _run_sandbox(root: Path, target: str) -> str:
    return _docker(root, ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", target])


def _check_target(target: str) -> str | None:
    if target.startswith("-"):
        return "錯誤:target 必須是路徑,不能是參數"
    _safe_path(target)
    return None


@tool
def run_tests(target: str = "tests") -> str:
    """在 Docker 沙箱跑 pytest,測的是「目前的原始碼」(提出的 patch 不會套用)。改 bug 前先用它重現問題。"""
    if err := _check_target(target):
        return err
    with _sandbox_copy() as copy:
        return _run_sandbox(copy, target)


@tool
def run_command(command: str) -> str:
    """在 Docker 沙箱用 sh 執行任意指令(例如 python script.py、ls -R、wc -l),看到的是「目前的原始碼」。
    沙箱沒有網路、沒有 git、檔案唯讀(只有 /tmp 可寫)、看不到機密檔、180 秒上限;不能用來修改工作區。"""
    with _sandbox_copy() as copy:
        return _docker(copy, ["sh", "-c", command])


def _git(*args: str) -> str:
    if not shutil.which("git"):
        return "錯誤:這台機器沒有 git"
    r = subprocess.run(["git", "--no-pager", *args], cwd=WORKSPACE, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=30)
    if r.returncode != 0:
        return f"錯誤:{r.stderr.strip() or 'git 執行失敗'}"
    return r.stdout.strip() or "(沒有輸出)"


@tool
def git_log(path: str = "", max_count: int = 10) -> str:
    """看 git 提交紀錄:誰、哪天、為什麼改,附改了哪些檔案。給 path(檔案或資料夾)就只看跟它有關的提交。"""
    args = ["log", f"--max-count={min(max(max_count, 1), 50)}", "--stat", "--date=short",
            "--format=%h %ad %an%n    %s"]
    if path:
        _safe_path(path)
        args += ["--", path]
    return _git(*args)


@tool
def git_blame(path: str, start_line: int, end_line: int) -> str:
    """看某段程式每一行最後是哪次提交、誰改的(追 bug 是什麼時候引入的)。一次最多 100 行。"""
    _safe_path(path)
    return _git("blame", "--date=short", "-L", f"{start_line},{min(end_line, start_line + 99)}", "--", path)


class Edit(TypedDict):
    path: str
    old_string: str
    new_string: str


def _apply_edit(text: str, old: str, new: str) -> str:
    """把唯一出現的 old 換成 new;找不到或不唯一就丟 ValueError(訊息會回給模型)。"""
    if not old:
        raise ValueError("old_string 不能是空的(檔案已存在;old_string 留空只用在新增檔案)")
    if old not in text and "\r\n" in text:  # read_file 給模型看的是 \n,CRLF 檔要換回來比對
        old, new = (s.replace("\r\n", "\n").replace("\n", "\r\n") for s in (old, new))
    n = text.count(old)
    if n == 0:
        raise ValueError("找不到 old_string。請用 read_file 重讀,逐字複製(不要帶「    12| 」行號前綴,縮排要一致)")
    if n > 1:
        raise ValueError(f"old_string 出現 {n} 次,請多帶幾行上下文讓它唯一")
    return text.replace(old, new, 1)


def _unified_diff(rel: str, before: str | None, after: str) -> str:
    """git apply 吃得下的 unified diff;before=None 是新檔(--- /dev/null)。
    最後一行沒換行要補標記(difflib 不會加)。"""
    out = []
    for line in difflib.unified_diff((before or "").splitlines(keepends=True), after.splitlines(keepends=True),
                                     "/dev/null" if before is None else f"a/{rel}", f"b/{rel}"):
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out)


@tool
def propose_patch(edits: list[Edit], reason: str, test_target: str = "tests") -> str:
    """提出修改,存成 .patch 等人工審核,原始檔案不會被改動。
    edits:每一筆把 path 檔案裡的 old_string 換成 new_string。old_string 要從 read_file 的輸出逐字複製
    (不要帶「    12| 」行號前綴)、在檔案裡只出現一次,不夠唯一就多帶幾行上下文。可以一次改多處、多個檔案;
    每次都是對「原始檔案」修改(之前提的 patch 不會套用)。
    新增檔案(例如補回歸測試):path 用不存在的路徑、old_string 留空、new_string 放整個檔案內容。
    diff 由工具產生,套到工作區的暫存複本上、在沙箱跑 test_target,回傳 diff 與套用後的真實測試結果。"""
    if err := _check_target(test_target):
        return err
    if not shutil.which("git"):
        return "錯誤:這台機器沒有 git,無法驗證 patch"
    if not edits:
        return "錯誤:edits 是空的"
    files: dict[str, tuple[str | None, str]] = {}  # 相對路徑 → (原文,新檔為 None;改後)
    for i, e in enumerate(edits, 1):
        p = _safe_path(e["path"])
        rel = p.relative_to(WORKSPACE).as_posix()
        if rel in files:
            before, after = files[rel]
        elif p.is_file():
            before = after = p.read_bytes().decode("utf-8")  # 不轉換行,CRLF 檔才不會整檔變動
        elif not p.exists() and not e["old_string"]:
            files[rel] = (None, e["new_string"])
            continue
        else:
            return f"錯誤:第 {i} 筆修改:{rel} 不是既有檔案;要新增檔案請用不存在的路徑並把 old_string 留空"
        try:
            files[rel] = (before, _apply_edit(after, e["old_string"], e["new_string"]))
        except ValueError as err:
            return f"錯誤:第 {i} 筆修改({rel}):{err}"
    diff = "".join(_unified_diff(rel, before, after) for rel, (before, after) in files.items())
    if not diff:
        return "錯誤:old_string 和 new_string 一樣,沒有任何變更"

    out_dir = WORKSPACE / ".agent_patches"
    out_dir.mkdir(exist_ok=True)
    name = Path(next(iter(files))).name + (f"+{len(files) - 1}" if len(files) > 1 else "")
    patch_file = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{name}.patch"
    patch_file.write_text(f"# {' '.join(reason.split())}\n{diff}", encoding="utf-8", newline="\n")
    r = subprocess.run(["git", "apply", "--check", str(patch_file)],
                       cwd=WORKSPACE, capture_output=True, text=True)
    if r.returncode != 0:  # diff 是從原檔算出來的,照理一定套得上;套不上多半是編碼或換行的怪狀況
        patch_file.unlink()
        return f"錯誤:產生的 diff 套不上去:\n{r.stderr.strip()}"

    with _sandbox_copy() as copy:
        # 關掉換行轉換:Git for Windows 預設 autocrlf=true,會把複本裡的 LF 檔改成 CRLF,測到的就不是原本的內容
        subprocess.run(["git", "-c", "core.autocrlf=false", "apply", str(patch_file)],
                       cwd=copy, check=True, capture_output=True)
        result = _run_sandbox(copy, test_target)
    PATCHES.append((patch_file, result.startswith("exit code 0")))
    saved = patch_file.relative_to(WORKSPACE).as_posix()
    # 10/9 實測:模型先單獨修 bug、再單獨送新測試檔,以為會疊加,結果測試失敗多繞好幾步
    return (f"已存成 {saved},等工程師審核後套用。\n```diff\n{diff}```\n套用此 patch 後的測試結果:\n{result}\n"
            "<note>給你的內部提醒,不要轉述給使用者:每個 patch 都是單獨套在原始檔案上驗證,"
            "之前的 patch 不會一起套用;要調整就重送包含全部修改的 edits。</note>")


def apply_patch(patch_file: Path) -> str:
    """把 patch 真的套用到工作區。只給人在互動模式確認後呼叫,不是給模型的工具。"""
    # 關掉換行轉換:patch 是照檔案實際的位元組產生的,autocrlf 會把整個檔案的換行都改掉
    r = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", str(patch_file)], cwd=WORKSPACE,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return f"套用失敗(工作區可能已經改過):{r.stderr.strip()}"
    return f"已套用 {patch_file.name} 到工作區"


ALL_TOOLS = [list_files, read_file, search_code, git_log, git_blame, run_tests, run_command, propose_patch]
