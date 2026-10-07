"""資料夾清單、上傳安全檢查、過期清理、整份文件的檔案清單。"""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from web.workspaces import MAX_FILE_BYTES, MAX_FILES, UploadRejected, WorkspaceRegistry


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


def no_upload_left(tmp_path):
    uploads = tmp_path / "uploads"
    return not uploads.exists() or not any(uploads.iterdir())


def test_drive_qualified_parts_cannot_escape(reg, tmp_path):
    for name in ["x/C:../C:../C:../C:../pwned.py", "D:x/evil.py", "a/b:c.py"]:
        with pytest.raises(UploadRejected, match="路徑可疑"):
            reg.add_upload([(name, b"x")])
        with pytest.raises(UploadRejected, match="路徑可疑"):
            reg.add_upload([("evil.zip", make_zip({name: b"x"}))])
    assert not list(tmp_path.rglob("pwned.py")) and not list(tmp_path.rglob("evil.py"))
    assert no_upload_left(tmp_path)


def test_case_variant_git_and_secrets_dropped(reg):
    blob = make_zip({"app.py": b"x", ".GIT/config": b"[core]\n\tfsmonitor = touch pwned\n",
                     ".Git/HEAD": b"ref", "conf/.ENV": b"K=1"})
    ws, skipped = reg.add_upload([("repo.zip", blob)])
    assert sorted(skipped) == sorted([".GIT/config", ".Git/HEAD", "conf/.ENV"])
    assert not (ws.path / ".GIT").exists() and (ws.path / "app.py").exists()


def test_upload_whole_files_never_runs_git(reg, monkeypatch):
    ws, _ = reg.add_upload([("a.py", b"x")])
    (ws.path / ".git").mkdir()  # 模擬:就算上傳資料夾裡有 .git,也不能執行 git

    def boom(*args, **kwargs):
        raise AssertionError("上傳的資料夾不該執行 git")

    monkeypatch.setattr("web.workspaces.subprocess.run", boom)
    assert reg.whole_files(ws) == ["a.py"]


def test_git_internals_in_zip_do_not_count_toward_limits(reg):
    entries = {"a.py": b"x", ".git/objects/pack/p.pack": b"\0" * (MAX_FILE_BYTES + 1)}
    entries.update({f".git/objects/{i:04d}": b"o" for i in range(MAX_FILES + 1)})
    ws, skipped = reg.add_upload([("repo.zip", make_zip(entries))])
    assert (ws.path / "a.py").exists() and len(skipped) == MAX_FILES + 2


def test_oversize_entry_in_compressed_zip_rejected(reg):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.py", b"0" * (MAX_FILE_BYTES + 1))
    with pytest.raises(UploadRejected, match="2 MB"):
        reg.add_upload([("big.zip", buf.getvalue())])


def test_corrupt_zip_entry_rejected_without_leftovers(reg, tmp_path):
    blob = make_zip({"a.py": b"hello world"}).replace(b"hello world", b"jello world", 1)
    with pytest.raises(UploadRejected, match="zip"):
        reg.add_upload([("bad.zip", blob)])
    assert no_upload_left(tmp_path)


def test_invalid_windows_filename_rejected_without_leftovers(reg, tmp_path):
    with pytest.raises(UploadRejected):
        reg.add_upload([("a?b.py", b"x")])
    assert no_upload_left(tmp_path)
