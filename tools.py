"""Agent 工具。原則:小而精準的輸出 + 錯誤回給模型自己修正 + 不直接改檔。"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

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


@tool
def propose_patch(path: str, unified_diff: str, reason: str, test_target: str = "tests") -> str:
    """提出修改:輸入 unified diff,存成 .patch 等人工審核,原始檔案不會被改動。
    會先檢查 diff 能不能套用,再把 patch 套到工作區的暫存複本上、在沙箱跑 test_target,回傳套用後的真實測試結果。"""
    _safe_path(path)
    if err := _check_target(test_target):
        return err
    if not shutil.which("git"):
        return "錯誤:這台機器沒有 git,無法驗證 patch"
    out_dir = WORKSPACE / ".agent_patches"
    out_dir.mkdir(exist_ok=True)
    patch_file = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{Path(path).name}.patch"
    patch_file.write_text(f"# {reason}\n{unified_diff.rstrip()}\n", encoding="utf-8", newline="\n")
    r = subprocess.run(["git", "apply", "--check", str(patch_file)],
                       cwd=WORKSPACE, capture_output=True, text=True)
    if r.returncode != 0:
        patch_file.unlink()
        return f"diff 套不上去,請依錯誤修正後重送:\n{r.stderr.strip()}"

    # 套到暫存複本上驗證;正式環境大 repo 改用 git worktree,不要整包複製
    with tempfile.TemporaryDirectory(prefix="agent_verify_") as tmp:
        copy = Path(tmp) / "ws"
        shutil.copytree(WORKSPACE, copy, ignore=_COPY_IGNORE)
        subprocess.run(["git", "apply", str(patch_file)], cwd=copy, check=True, capture_output=True)
        result = _run_sandbox(copy, test_target)
    saved = patch_file.relative_to(WORKSPACE).as_posix()
    return f"已存成 {saved},等工程師審核後套用。\n套用此 patch 後的測試結果:\n{result}"


ALL_TOOLS = [read_file, search_code, run_tests, propose_patch]
