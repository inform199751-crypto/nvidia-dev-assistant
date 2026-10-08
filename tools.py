"""Agent 工具。原則:小而精準的輸出 + 錯誤回給模型自己修正 + 不直接改檔。"""
from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from langchain_core.tools import tool

WORKSPACE = Path(os.getenv("AGENT_WORKSPACE", ".")).resolve()
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "dev-assistant-sandbox")  # sandbox/Dockerfile 建的(含 pytest)
MAX_READ_LINES = 400


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


@tool
def search_code(pattern: str, glob: str = "**/*.py", max_hits: int = 50) -> str:
    """在工作區用正規表示式搜尋,回傳「檔案:行號: 內容」。先搜再讀,不要整檔讀。"""
    rx = re.compile(pattern)
    hits: list[str] = []
    for f in WORKSPACE.glob(glob):
        if (not f.is_file() or _is_secret(f)
                or any(part.startswith(".") for part in f.relative_to(WORKSPACE).parts)):
            continue
        real = f.resolve()  # symlink / Windows junction 可能指到工作區外或機密檔,看實際位置
        if not real.is_relative_to(WORKSPACE) or _is_secret(real):
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{f.relative_to(WORKSPACE).as_posix()}:{n}: {line.strip()[:200]}")
                if len(hits) >= max_hits:
                    return "\n".join(hits) + f"\n[已達 {max_hits} 筆上限,請把 pattern 寫得更精確]"
    return "\n".join(hits) or "沒有符合的結果"


# 暫存複本不帶 .env 和金鑰檔:沙箱裡的測試輸出會回到模型手上
_COPY_IGNORE = shutil.ignore_patterns(".agent_patches", ".git", ".venv", "__pycache__", "node_modules",
                                      ".env", ".env.local", ".env.prod*", "*.pem", "*.key", "id_rsa*")


def _run_sandbox(root: Path, target: str) -> str:
    """在隔離的 Docker 容器裡跑 pytest:無網路、唯讀掛載、限 CPU / 記憶體 / 行程數 / 時間。"""
    if not shutil.which("docker"):
        return "錯誤:這台機器沒有 Docker,沙箱無法啟動。為了安全,不會改在主機上直接執行程式碼。"
    cmd = [
        "docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp",
        "--memory", "1g", "--cpus", "1", "--pids-limit", "256",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{root}:/ws:ro", "-w", "/ws",
        SANDBOX_IMAGE, "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", target,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return "錯誤:測試超過 180 秒被中止(可能有無窮迴圈或死結)"
    return f"exit code {r.returncode}\n{r.stdout}\n{r.stderr}".strip()


def _check_target(target: str) -> str | None:
    if target.startswith("-"):
        return "錯誤:target 必須是路徑,不能是參數"
    _safe_path(target)
    return None


@tool
def run_tests(target: str = "tests") -> str:
    """在 Docker 沙箱跑 pytest,測的是「目前的原始碼」(提出的 patch 不會套用)。改 bug 前先用它重現問題。"""
    return _check_target(target) or _run_sandbox(WORKSPACE, target)


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

    # 套到暫存複本上驗證;正式環境大 repo 改用 git worktree,不要整包複製
    with tempfile.TemporaryDirectory(prefix="agent_verify_") as tmp:
        copy = Path(tmp) / "ws"
        shutil.copytree(WORKSPACE, copy, ignore=_COPY_IGNORE)
        # 關掉換行轉換:Git for Windows 預設 autocrlf=true,會把複本裡的 LF 檔改成 CRLF,測到的就不是原本的內容
        subprocess.run(["git", "-c", "core.autocrlf=false", "apply", str(patch_file)],
                       cwd=copy, check=True, capture_output=True)
        result = _run_sandbox(copy, test_target)
    saved = patch_file.relative_to(WORKSPACE).as_posix()
    # 10/9 實測:模型先單獨修 bug、再單獨送新測試檔,以為會疊加,結果測試失敗多繞好幾步
    return (f"已存成 {saved},等工程師審核後套用。\n```diff\n{diff}```\n套用此 patch 後的測試結果:\n{result}\n"
            "(提醒:每個 patch 都是單獨套在原始檔案上驗證,之前的 patch 不會一起套用;"
            "要調整就重送包含全部修改的 edits)")


ALL_TOOLS = [read_file, search_code, run_tests, propose_patch]
