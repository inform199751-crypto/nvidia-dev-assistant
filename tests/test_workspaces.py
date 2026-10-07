"""資料夾清單、上傳安全檢查、過期清理、整份文件的檔案清單。"""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from web.workspaces import MAX_FILE_BYTES, UploadRejected, WorkspaceRegistry


def make_zip(entries: dict[str, bytes], symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "/etc/passwd")
    return buf.getvalue()


@pytest.fixture
def reg(tmp_path):
    (tmp_path / "proj").mkdir()
    (tmp_path / "workspaces.yaml").write_text(
        "workspaces:\n  - name: demo\n    path: proj\n  - name: 不存在\n    path: nope\n", encoding="utf-8")
    return WorkspaceRegistry(tmp_path / "workspaces.yaml", base_dir=tmp_path, uploads_dir=tmp_path / "uploads")


def test_config_list_skips_missing_folders(reg, tmp_path):
    assert [(w.id, w.kind) for w in reg.list()] == [("demo", "config")]
    assert reg.get("demo").path == (tmp_path / "proj").resolve()
    with pytest.raises(KeyError):
        reg.get("不存在")


def test_zip_upload_filters_secrets_and_git(reg):
    blob = make_zip({"src/app.py": b"print(1)\n", ".env": b"K=1", "keys/server.key": b"x",
                     ".git/config": b"x", "__MACOSX/._app.py": b"x"})
    ws, skipped = reg.add_upload([("proj.zip", blob)])
    assert ws.kind == "upload" and ws.name == "上傳:proj.zip"
    assert (ws.path / "src" / "app.py").read_bytes() == b"print(1)\n"
    assert sorted(skipped) == sorted([".env", "keys/server.key", ".git/config", "__MACOSX/._app.py"])
    assert reg.get(ws.id) == ws


def test_zip_path_traversal_rejected(reg):
    with pytest.raises(UploadRejected, match="路徑可疑"):
        reg.add_upload([("evil.zip", make_zip({"../../evil.py": b"x"}))])


def test_zip_absolute_path_rejected(reg):
    with pytest.raises(UploadRejected, match="路徑可疑"):
        reg.add_upload([("evil.zip", make_zip({"/etc/cron.d/x": b"x"}))])


def test_zip_symlink_rejected(reg):
    with pytest.raises(UploadRejected, match="符號連結"):
        reg.add_upload([("evil.zip", make_zip({"a.py": b"x"}, symlink="link"))])


def test_oversize_file_rejected(reg):
    with pytest.raises(UploadRejected, match="2 MB"):
        reg.add_upload([("big.py", b"x" * (MAX_FILE_BYTES + 1))])


def test_bad_zip_rejected(reg):
    with pytest.raises(UploadRejected, match="zip"):
        reg.add_upload([("x.zip", b"not a zip")])


def test_plain_files_upload(reg):
    ws, skipped = reg.add_upload([("a.py", b"A\n"), ("b.py", b"B\n"), (".env", b"K=1")])
    assert ws.name == "上傳:3 個檔案" and skipped == [".env"]
    assert (ws.path / "a.py").exists() and not (ws.path / ".env").exists()


def test_expired_uploads_hidden_and_cleaned(reg):
    ws, _ = reg.add_upload([("a.py", b"A\n")])
    meta = ws.path.parent / "meta.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["created_at"] -= 25 * 3600
    meta.write_text(json.dumps(data), encoding="utf-8")
    assert ws.id not in [w.id for w in reg.list()]
    assert reg.cleanup() == 1 and not ws.path.parent.exists()


def test_whole_files_picks_text_and_skips_secrets(reg, tmp_path):
    proj = tmp_path / "proj"
    files = {"app.py": "x", "docs/a.md": "y", "logo.png": "z", ".env": "K",
             "node_modules/m.js": "n", ".agent_patches/p.patch": "p"}
    for rel, data in files.items():
        (proj / rel).parent.mkdir(parents=True, exist_ok=True)
        (proj / rel).write_text(data, encoding="utf-8")
    assert reg.whole_files(reg.get("demo")) == ["app.py", "docs/a.md"]
