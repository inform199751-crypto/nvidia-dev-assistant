"""網頁可選的資料夾:設定檔清單 + 上傳的暫存複本。

上傳一律先檢查再落地:路徑穿越、符號連結、超量整包拒收;機密檔與 .git 丟掉並告知。
"""
from __future__ import annotations

import io
import json
import lzma
import ntpath
import shutil
import subprocess
import time
import uuid
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from tools import _is_secret

MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_FILES = 2000
MAX_FILE_BYTES = 2 * 1024 * 1024
TTL_SECONDS = 24 * 3600
TEXT_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".vue", ".java", ".kt", ".go", ".rs", ".c", ".h",
    ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".sql", ".sh", ".ps1", ".md", ".rst",
    ".txt", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json", ".html", ".css", ".scss", ".j2",
    ".jinja", ".xml", ".gradle",
}
_SKIP_DIRS = {".git", "__macosx"}  # 比對時一律轉小寫:Windows 上 .GIT 就是 .git
_BAD_CHARS = set(':<>"|?*\0')


class UploadRejected(ValueError):
    """整包拒收,訊息會直接顯示給使用者。"""


@dataclass(frozen=True)
class Workspace:
    id: str
    name: str
    path: Path
    kind: str  # "config" | "upload"
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "path": str(self.path)}


def _clean_rel(name: str) -> PurePosixPath | None:
    """回傳安全的相對路徑;機密檔、.git、macOS 雜檔回 None(丟掉);危險路徑整包拒收。
    Windows 裝置名稱(aux.c、con.h、nul、lpt1.py …,不管副檔名)寫入時會卡住或打到真的裝置,
    但真實專案裡有這種檔(例如 Linux 核心的 aux.c),所以和 .env 一樣丟掉並列在 skipped,不整包拒收。

    Windows 上 `C:..` 這種片段會被 pathlib 當成磁碟機路徑而跳出資料夾,所以帶冒號、保留字元、
    結尾是點或空白的片段一律拒收;寫入前還有一道 resolve 後的範圍檢查。
    """
    name = name.replace("\\", "/")
    p = PurePosixPath(name)
    if (name.startswith("/") or ".." in p.parts
            or any(set(part) & _BAD_CHARS or part.endswith((".", " ")) for part in p.parts)):
        raise UploadRejected(f"路徑可疑,整包拒收:{name}")
    if not p.parts or any(part.lower() in _SKIP_DIRS or ntpath.isreserved(part) for part in p.parts) or _is_secret(Path(p.name)):
        return None
    return p


class WorkspaceRegistry:
    def __init__(self, config_path: Path, base_dir: Path, uploads_dir: Path, ttl_seconds: int = TTL_SECONDS):
        self.base_dir = Path(base_dir).resolve()
        self.uploads_dir = Path(uploads_dir)
        self.ttl = ttl_seconds
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) if Path(config_path).exists() else {}
        self._config: list[Workspace] = []
        for item in (cfg or {}).get("workspaces", []):
            path = Path(item["path"])
            path = (path if path.is_absolute() else self.base_dir / path).resolve()
            self._config.append(Workspace(id=item["name"], name=item["name"], path=path, kind="config"))

    # ---------- 查詢 ----------
    def list(self) -> list[Workspace]:
        out = [w for w in self._config if w.path.is_dir()]
        if self.uploads_dir.exists():
            for d in sorted(self.uploads_dir.iterdir()):
                w = self._load_upload(d)
                if w and time.time() - w.created_at < self.ttl:
                    out.append(w)
        return out

    def get(self, workspace_id: str) -> Workspace:
        for w in self.list():
            if w.id == workspace_id:
                return w
        raise KeyError(workspace_id)

    def _load_upload(self, d: Path) -> Workspace | None:
        try:
            m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            return Workspace(id=d.name, name=f"上傳:{m['name']}", path=d / "files", kind="upload",
                             created_at=float(m["created_at"]))
        except (OSError, ValueError, KeyError, TypeError):  # 沒有、沒寫完或損壞的 meta.json:當作不存在,cleanup 會清掉
            return None

    # ---------- 上傳 ----------
    def add_upload(self, files: list[tuple[str, bytes]]) -> tuple[Workspace, list[str]]:
        """files:[(檔名, 內容)]。只有一個 .zip 就解壓,否則當一般檔案。回傳 (資料夾, 被丟掉的檔名)。"""
        if not files:
            raise UploadRejected("沒有收到檔案")
        if sum(len(data) for _, data in files) > MAX_TOTAL_BYTES:
            raise UploadRejected("超過 20 MB 上限")
        if len(files) == 1 and files[0][0].lower().endswith(".zip"):
            keep, skipped = self._zip_entries(files[0][1])
            label = files[0][0]
        else:
            keep, skipped = self._plain_entries(files)
            label = files[0][0] if len(files) == 1 else f"{len(files)} 個檔案"
        if not keep:
            raise UploadRejected("過濾掉機密檔與系統檔之後,沒有可用的檔案")
        root = self.uploads_dir / f"up-{uuid.uuid4().hex[:8]}"
        files_dir = (root / "files").resolve()
        try:
            for rel, data in keep:
                target = files_dir.joinpath(*rel.parts).resolve()
                if not target.is_relative_to(files_dir):  # 最後一道防線:不管檔名怎麼寫,都只能落在這次上傳的資料夾裡
                    raise UploadRejected(f"路徑可疑,整包拒收:{rel}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            (root / "meta.json").write_text(json.dumps({"name": label, "created_at": time.time()}, ensure_ascii=False),
                                            encoding="utf-8")
        except UploadRejected:
            shutil.rmtree(root, ignore_errors=True)
            raise
        except (OSError, ValueError) as e:
            shutil.rmtree(root, ignore_errors=True)
            raise UploadRejected(f"檔名或內容無法寫入:{e}") from None
        return self._load_upload(root), skipped

    @staticmethod
    def _plain_entries(files: list[tuple[str, bytes]]) -> tuple[list[tuple[PurePosixPath, bytes]], list[str]]:
        keep, skipped = [], []
        for name, data in files:
            rel = _clean_rel(name)
            if rel is None:
                skipped.append(name)
                continue
            if len(data) > MAX_FILE_BYTES:
                raise UploadRejected(f"單檔超過 2 MB:{name}")
            keep.append((rel, data))
        if len(keep) > MAX_FILES:
            raise UploadRejected(f"檔案數超過 {MAX_FILES} 個")
        return keep, skipped

    @staticmethod
    def _zip_entries(blob: bytes) -> tuple[list[tuple[PurePosixPath, bytes]], list[str]]:
        """先過濾再算上限(整個 repo 打包時,.git 不該把上傳擋掉);大小用實際解壓出來的位元組算,不信 zip 自報的大小。"""
        try:
            zf = zipfile.ZipFile(io.BytesIO(blob))
            keep, skipped, total = [], [], 0
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise UploadRejected(f"zip 裡有捷徑檔(符號連結),整包拒收:{info.filename}")
                rel = _clean_rel(info.filename)
                if rel is None:
                    skipped.append(info.filename)
                    continue
                if len(keep) >= MAX_FILES:
                    raise UploadRejected(f"檔案數超過 {MAX_FILES} 個")
                with zf.open(info) as f:
                    data = f.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    raise UploadRejected(f"單檔超過 2 MB:{info.filename}")
                total += len(data)
                if total > MAX_TOTAL_BYTES:
                    raise UploadRejected("解壓後超過 20 MB 上限")
                keep.append((rel, data))
            return keep, skipped
        except UploadRejected:
            raise
        except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, NotImplementedError, EOFError,
                zlib.error, lzma.LZMAError, ValueError, OSError) as e:
            raise UploadRejected(f"zip 檔損壞或格式不支援(例如加密):{e}") from None

    def cleanup(self, now: float | None = None) -> int:
        """刪掉過期或沒寫完的上傳;回傳刪了幾個。"""
        now = now if now is not None else time.time()
        if not self.uploads_dir.exists():
            return 0
        removed = 0
        for d in list(self.uploads_dir.iterdir()):
            w = self._load_upload(d)
            if w is None or now - w.created_at >= self.ttl:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        return removed

    # ---------- 整份文件模式 ----------
    def whole_files(self, ws: Workspace) -> list[str]:
        """整份文件要附上的檔案:設定清單裡的 git 專案用 git ls-files,其餘掃描;只留文字檔,排除機密檔與隱藏資料夾。

        上傳的資料夾一律不跑 git:就算裡面有 .git,也是使用者給的,其中的設定(例如 fsmonitor)能執行指令。
        """
        root = ws.path
        names = None
        if ws.kind == "config" and (root / ".git").exists():
            try:
                r = subprocess.run(["git", "-c", "core.quotepath=off", "-c", "core.fsmonitor=false", "ls-files"],
                                   cwd=root, capture_output=True, text=True, encoding="utf-8")
                if r.returncode == 0:
                    names = [n for n in r.stdout.splitlines() if (root / n).is_file()]  # 刪掉但還沒 git rm 的檔案不列
            except FileNotFoundError:  # 沒裝 git 就改用掃描
                names = None
        if names is None:
            names = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
        out = []
        for name in sorted(names):
            p = PurePosixPath(name)
            if (p.suffix.lower() not in TEXT_SUFFIXES or _is_secret(Path(p.name))
                    or any(part.startswith(".") or part in {"node_modules", "__pycache__"} for part in p.parts[:-1])):
                continue
            out.append(name)
        return out
