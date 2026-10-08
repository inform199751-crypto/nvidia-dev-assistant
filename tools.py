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
from pathlib import Path, PurePosixPath
from typing import TypedDict

from langchain_core.tools import tool

import documents

WORKSPACE = Path(os.getenv("AGENT_WORKSPACE", ".")).resolve()
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "dev-assistant-sandbox")  # sandbox/Dockerfile 建的(含 pytest)
OUTPUT_DIR = Path(os.getenv("AGENT_OUTPUT", Path(__file__).with_name("output"))).resolve()  # save_document 存這裡
MAX_READ_LINES = 400
PATCHES: list[tuple[Path, bool]] = []  # 提出過的 patch(檔案, 沙箱測試是否通過);互動模式拿來問要不要套用
# 小助理的草稿:工作區相對路徑 → 改過的完整內容。propose_patch 的修改疊在這裡,讀檔 / 搜尋 / 沙箱都看草稿版本;
# 使用者套用(apply_patch)或捨棄(discard_draft)之前,真正的檔案都不會動
DRAFT: dict[str, str] = {}


def configure(workspace: str | Path | None = None, sandbox_image: str | None = None,
              output_dir: str | Path | None = None) -> None:
    """讀完 .env 之後要再呼叫一次 —— 模組載入時 .env 還沒讀,上面幾個值會是預設值。"""
    global WORKSPACE, SANDBOX_IMAGE, OUTPUT_DIR
    if workspace is not None:
        WORKSPACE = Path(workspace).resolve()
    if sandbox_image:
        SANDBOX_IMAGE = sandbox_image
    if output_dir:
        OUTPUT_DIR = Path(output_dir).resolve()


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


def _rel(p: Path) -> str:
    return p.relative_to(WORKSPACE).as_posix()


def current_text(p: Path) -> str | None:
    """檔案「現在」的內容:草稿裡有就用草稿(小助理改過、還沒套用的版本),沒有就讀工作區;都沒有回傳 None。"""
    if (rel := _rel(p)) in DRAFT:
        return DRAFT[rel]
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None


@tool
def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """讀取工作區檔案的指定行號範圍,每行前面附行號。一次最多 400 行;大檔請先看大綱或 search_code 再分段讀。
    你改過、還沒套用的檔案,讀到的是草稿版本。"""
    p = _safe_path(path)
    if (text := current_text(p)) is None:
        raise FileNotFoundError(f"{path} 不存在")
    lines = text.splitlines()
    start = max(start_line, 1)
    end = min(end_line or len(lines), len(lines), start + MAX_READ_LINES - 1)
    body = "\n".join(f"{n:>5}| {lines[n - 1]}" for n in range(start, end + 1))
    more = f"\n[檔案共 {len(lines)} 行,已顯示 {start}-{end}]" if end < len(lines) or start > 1 else ""
    return body + more


DOC_PART_TOKENS = 3000  # read_document 每段的大小,要比 limits.tool_output_max_tokens 小才不會被截斷


@tool
def read_document(path: str, part: int = 1) -> str:
    """讀 PDF、Word(.docx)、Excel(.xlsx)的文字內容(也能讀一般文字檔)。很長的檔案會分段,用 part=2、3… 讀後面。"""
    parts = documents.split_parts(documents.extract_text(_safe_path(path)), DOC_PART_TOKENS)
    if not 1 <= part <= len(parts):
        return f"錯誤:{path} 只有 {len(parts)} 段"
    more = f"\n[還有 {len(parts) - part} 段,用 part={part + 1} 繼續讀]" if part < len(parts) else ""
    return f"[{path} 第 {part}/{len(parts)} 段]\n{parts[part - 1]}{more}"


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
        rel = _rel(f)
        mark = "  (草稿已修改)" if rel in DRAFT else ""
        rows.append(f"{rel}/" if f.is_dir() else f"{rel}  ({f.stat().st_size:,} bytes){mark}")
        if len(rows) >= max_entries:
            return "\n".join(rows) + f"\n[已達 {max_entries} 筆上限,請把 glob 縮小]"
    rows += [f"{rel}  (草稿新增,還沒套用)" for rel in sorted(_draft_only(glob))]
    return "\n".join(rows) or "沒有符合的檔案"


def _draft_only(glob: str) -> list[str]:
    """草稿裡新增、工作區還沒有的檔案中,符合 glob 的。"""
    return [rel for rel in DRAFT if not (WORKSPACE / rel).exists() and PurePosixPath(rel).full_match(glob)]


@tool
def search_code(pattern: str, glob: str = "**/*.py", max_hits: int = 50) -> str:
    """在工作區用正規表示式搜尋,回傳「檔案:行號: 內容」。先搜再讀,不要整檔讀。"""
    rx = re.compile(pattern)
    hits: list[str] = []
    files = [f for f in WORKSPACE.glob(glob) if f.is_file() and _visible(f)]
    for f in files + [WORKSPACE / rel for rel in _draft_only(glob)]:  # 草稿版本優先,草稿新增的檔案也搜
        for n, line in enumerate((current_text(f) or "").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{_rel(f)}:{n}: {line.strip()[:200]}")
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
        for rel, text in DRAFT.items():  # 草稿的修改也放進去:沙箱測的是小助理改過的版本
            target = copy / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
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
    """在 Docker 沙箱跑 pytest,測的是目前的版本(原始碼加上你草稿裡還沒套用的修改)。改 bug 前先用它重現問題。"""
    if err := _check_target(target):
        return err
    with _sandbox_copy() as copy:
        return _run_sandbox(copy, target)


@tool
def run_command(command: str) -> str:
    """在 Docker 沙箱用 sh 執行任意指令(例如 python script.py、ls -R、wc -l),看到的是目前的版本(含草稿修改)。
    沙箱沒有網路、沒有 git、檔案唯讀(只有 /tmp 可寫)、看不到機密檔、180 秒上限;不能用來修改檔案。"""
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


def _original(rel: str) -> str | None:
    """工作區裡真正的內容(不轉換行,CRLF 檔才不會整檔變動);不存在回傳 None。"""
    p = WORKSPACE / rel
    return p.read_bytes().decode("utf-8") if p.is_file() else None


@tool
def propose_patch(edits: list[Edit], reason: str, test_target: str = "tests") -> str:
    """修改程式:改在你的草稿上(使用者真正的檔案不會動),在沙箱跑 test_target,存成 .patch 等使用者確認後套用。
    edits:每一筆把 path 檔案裡的 old_string 換成 new_string。old_string 要從 read_file 的輸出逐字複製
    (不要帶「    12| 」行號前綴)、在檔案裡只出現一次,不夠唯一就多帶幾行上下文。可以一次改多處、多個檔案。
    修改會累積:之後 read_file、search_code、run_tests 看到的都是改過的版本,下一次修改也疊在上面。
    新增檔案(例如補回歸測試):path 用不存在的路徑、old_string 留空、new_string 放整個檔案內容。"""
    if err := _check_target(test_target):
        return err
    if not shutil.which("git"):
        return "錯誤:這台機器沒有 git,無法驗證 patch"
    if not edits:
        return "錯誤:edits 是空的"
    before: dict[str, str | None] = {}  # 這次改到的檔案,改之前(草稿或原始)的內容
    changed: dict[str, str] = {}        # 這次改到的檔案 → 改後
    for i, e in enumerate(edits, 1):
        p = _safe_path(e["path"])
        rel = _rel(p)
        if rel not in changed:
            before[rel] = DRAFT[rel] if rel in DRAFT else _original(rel)
        current = changed.get(rel, before[rel])
        if current is None:
            if e["old_string"] or p.exists():
                return f"錯誤:第 {i} 筆修改:{rel} 不是既有檔案;要新增檔案請用不存在的路徑並把 old_string 留空"
            changed[rel] = e["new_string"]
            continue
        try:
            changed[rel] = _apply_edit(current, e["old_string"], e["new_string"])
        except ValueError as err:
            return f"錯誤:第 {i} 筆修改({rel}):{err}"
    step_diff = "".join(_unified_diff(rel, before[rel], text) for rel, text in changed.items())
    if not step_diff:
        return "錯誤:old_string 和 new_string 一樣,沒有任何變更"

    # 新的草稿:改回跟原始一模一樣的檔案就拿掉。存檔的是「原始 → 草稿」的完整修改,使用者套用這一份就拿到全部
    draft = {rel: text for rel, text in {**DRAFT, **changed}.items() if text != _original(rel)}
    diff = "".join(_unified_diff(rel, _original(rel), draft[rel]) for rel in sorted(draft))
    if not diff:
        DRAFT.clear()
        return "這次的修改讓檔案回到原本的樣子,草稿已經清空,沒有要套用的修改。"
    out_dir = WORKSPACE / ".agent_patches"
    out_dir.mkdir(exist_ok=True)
    first = sorted(draft)[0]
    stem = f"{datetime.now():%Y%m%d_%H%M%S}_{Path(first).name}" + (f"+{len(draft) - 1}" if len(draft) > 1 else "")
    patch_file, n = out_dir / f"{stem}.patch", 2
    while patch_file.exists():  # 同一秒內改兩次不要互相蓋掉
        patch_file, n = out_dir / f"{stem}_{n}.patch", n + 1
    patch_file.write_text(f"# {' '.join(reason.split())}\n{diff}", encoding="utf-8", newline="\n")
    r = subprocess.run(["git", "apply", "--check", str(patch_file)],
                       cwd=WORKSPACE, capture_output=True, text=True)
    if r.returncode != 0:  # diff 是從原檔算出來的,照理一定套得上;套不上多半是編碼或換行的怪狀況
        patch_file.unlink()
        return f"錯誤:產生的 diff 套不上去:\n{r.stderr.strip()}"
    DRAFT.clear()
    DRAFT.update(draft)

    with _sandbox_copy() as copy:  # 複本 = 工作區 + 草稿,就是套用這份 patch 之後的樣子
        result = _run_sandbox(copy, test_target)
    PATCHES.append((patch_file, result.startswith("exit code 0")))
    saved = patch_file.relative_to(WORKSPACE).as_posix()
    # 10/9 實測:模型以為 patch 會疊加,一步一步送;以前每個 patch 各自對原始檔案驗證,它一直看不到自己的修改
    return (f"已改在草稿上,存成 {saved}(包含草稿裡全部 {len(draft)} 個檔案的修改,等使用者確認後套用)。\n"
            f"這次的修改:\n```diff\n{step_diff}```\n改完後的測試結果:\n{result}\n"
            "<note>給你的內部提醒,不要轉述給使用者:修改會累積在草稿上,之後讀檔、搜尋、跑測試看到的都是草稿版本;"
            "要調整就直接在目前的內容上再改。使用者確認後才會套用到真正的檔案。</note>")


_DOC_SUFFIXES = {".md", ".txt", ".csv", ".json", ".html", ".yaml", ".yml"}


@tool
def save_document(filename: str, content: str) -> str:
    """把寫好的文件、翻譯、摘要、信件草稿、表格存成檔案給使用者(存在輸出資料夾,不會動到工作區)。
    filename 只給檔名,例如 "技術文件.md"、"摘要.txt"、"清單.csv";同名檔案不會被覆蓋,會自動加編號。"""
    name = Path(filename.replace("\\", "/")).name  # 只取檔名:不能藉 ../ 寫到別的地方
    suffix = Path(name).suffix.lower()
    if not name or name.startswith(".") or suffix not in _DOC_SUFFIXES:
        return f"錯誤:檔名要像「報告.md」,副檔名限 {' '.join(sorted(_DOC_SUFFIXES))}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target, n = OUTPUT_DIR / name, 2
    while target.exists():
        target, n = OUTPUT_DIR / f"{Path(name).stem}_{n}{suffix}", n + 1
    # CSV 加 BOM:Excel 開沒有 BOM 的 UTF-8 CSV,中文會變亂碼
    target.write_text(content, encoding="utf-8-sig" if suffix == ".csv" else "utf-8", newline="\n")
    return f"已存成 {target}"


class PlanStep(TypedDict):
    step: str
    status: str  # pending / in_progress / done


_PLAN_MARKS = {"done": "☑", "in_progress": "◐"}


@tool
def update_plan(steps: list[PlanStep]) -> str:
    """照計畫執行時用來更新進度清單。每次都送完整清單;status 是 pending / in_progress / done。
    跟下一個實際動作放在同一次回應裡一起呼叫,不要單獨呼叫。step 用使用者的語言(繁體中文)寫,簡短一句。"""
    return "已更新進度:\n" + "\n".join(f"{_PLAN_MARKS.get(s.get('status'), '☐')} {s.get('step')}" for s in steps)


WRITE_TOOLS = {"propose_patch", "save_document"}  # 規劃模式不給:只能調查,不能留下任何變更


def apply_patch(patch_file: Path) -> str:
    """把 patch 真的套用到工作區,成功就清空草稿。只給人在互動模式確認後呼叫,不是給模型的工具。"""
    if PATCHES and patch_file != PATCHES[-1][0]:  # 每個 patch 都是累積的,套舊的會丟掉後面的修改
        return "這不是最新的修改;最新的那一份已經包含全部修改,請套用最新的"
    # 關掉換行轉換:patch 是照檔案實際的位元組產生的,autocrlf 會把整個檔案的換行都改掉
    r = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", str(patch_file)], cwd=WORKSPACE,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return f"套用失敗(工作區可能已經改過):{r.stderr.strip()}"
    DRAFT.clear()  # 草稿的內容已經是工作區的內容了
    return f"已套用 {patch_file.name} 到工作區"


def discard_draft() -> str:
    """捨棄草稿:小助理改過、還沒套用的內容全部丟掉(.patch 檔留著)。只給人呼叫。"""
    n = len(DRAFT)
    DRAFT.clear()
    return f"已捨棄草稿裡 {n} 個檔案的修改(.patch 檔還留著)" if n else "草稿是空的,沒有要捨棄的修改"


ALL_TOOLS = [list_files, read_file, search_code, git_log, git_blame, run_tests, run_command, propose_patch,
             read_document, save_document]
