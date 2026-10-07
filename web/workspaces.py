"""網頁可選的資料夾:設定檔清單 + 上傳的暫存複本。

上傳一律先檢查再落地:路徑穿越、符號連結、超量整包拒收;機密檔與 .git 丟掉並告知。
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import time
import uuid
import zipfile
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
_SKIP_DIRS = {".git", "__MACOSX"}


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
    """回傳安全的相對路徑;機密檔、.git、macOS 雜檔回 None(丟掉);危險路徑整包拒收。"""
    name = name.replace("\\", "/")
    p = PurePosixPath(name)
    if name.startswith("/") or (len(name) > 1 and name[1] == ":") or ".." in p.parts:
        raise UploadRejected(f"路徑可疑,整包拒收:{name}")
    if not p.parts or any(part in _SKIP_DIRS for part in p.parts) or _is_secret(Path(p.name)):
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
        meta = d / "meta.json"
        if not meta.exists():
            return None
        m = json.loads(meta.read_text(encoding="utf-8"))
        return Workspace(id=d.name, name=f"上傳:{m['name']}", path=d / "files", kind="upload",
                         created_at=m["created_at"])

    # ---------- 上傳 ----------
    def add_upload(self, files: list[tuple[str, bytes]]) -> tuple[Workspace, list[str]]:
        """files:[(檔名, 內容)]。只有一個 .zip 就解壓,否則當一般檔案。回傳 (資料夾, 被丟掉的檔名)。"""
        if not files:
            raise UploadRejected("沒有收到檔案")
        if sum(len(data) for _, data in files) > MAX_TOTAL_BYTES:
            raise UploadRejected("超過 20 MB 上限")
        if len(files) == 1 and files[0][0].lower().endswith(".zip"):
            entries, label = self._zip_entries(files[0][1]), files[0][0]
        else:
            entries = files
            label = files[0][0] if len(files) == 1 else f"{len(files)} 個檔案"
        if len(entries) > MAX_FILES:
            raise UploadRejected(f"檔案數超過 {MAX_FILES} 個")
        keep, skipped = [], []
        for name, data in entries:
            rel = _clean_rel(name)
            if rel is None:
                skipped.append(name)
                continue
            if len(data) > MAX_FILE_BYTES:
                raise UploadRejected(f"單檔超過 2 MB:{name}")
            keep.append((rel, data))
        if not keep:
            raise UploadRejected("過濾掉機密檔與系統檔之後,沒有可用的檔案")
        root = self.uploads_dir / f"up-{uuid.uuid4().hex[:8]}"
        for rel, data in keep:
            target = (root / "files").joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (root / "meta.json").write_text(json.dumps({"name": label, "created_at": time.time()}, ensure_ascii=False),
                                        encoding="utf-8")
        return self._load_upload(root), skipped

    @staticmethod
    def _zip_entries(blob: bytes) -> list[tuple[str, bytes]]:
        try:
            zf = zipfile.ZipFile(io.BytesIO(blob))
        except zipfile.BadZipFile:
            raise UploadRejected("不是有效的 zip 檔") from None
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > MAX_FILES:
            raise UploadRejected(f"檔案數超過 {MAX_FILES} 個")
        total, out = 0, []
        for info in infos:
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise UploadRejected(f"zip 裡有捷徑檔(符號連結),整包拒收:{info.filename}")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise UploadRejected("解壓後超過 20 MB 上限")
            if info.file_size > MAX_FILE_BYTES:
                raise UploadRejected(f"單檔超過 2 MB:{info.filename}")
            out.append((info.filename, zf.read(info)))
        return out

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
        """整份文件要附上的檔案:git 專案用 git ls-files,否則掃描;只留文字檔,排除機密檔與隱藏資料夾。"""
        root = ws.path
        if (root / ".git").exists():
            r = subprocess.run(["git", "-c", "core.quotepath=off", "ls-files"], cwd=root,
                               capture_output=True, text=True, encoding="utf-8")
            names = r.stdout.splitlines()
        else:
            names = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
        out = []
        for name in sorted(names):
            p = PurePosixPath(name)
            if (p.suffix.lower() not in TEXT_SUFFIXES or _is_secret(Path(p.name))
                    or any(part.startswith(".") or part in {"node_modules", "__pycache__"} for part in p.parts[:-1])):
                continue
            out.append(name)
        return out
