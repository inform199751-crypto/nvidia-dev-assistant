"""工具綁定資料夾:網頁同時跑多個任務、各自在不同資料夾時不能互相覆蓋。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import tools


def by_name(toolset):
    return {t.name: t for t in toolset}


def test_each_toolset_reads_its_own_folder(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "x.py").write_text("print('A')\n", encoding="utf-8")
    (b / "x.py").write_text("print('B')\n", encoding="utf-8")
    ta, tb = by_name(tools.make_tools(a)), by_name(tools.make_tools(b))
    assert "print('A')" in ta["read_file"].invoke({"path": "x.py"})
    assert "print('B')" in tb["read_file"].invoke({"path": "x.py"})
    assert tb["search_code"].invoke({"pattern": "print"}).startswith("x.py:1: print('B')")


def test_toolsets_stay_isolated_across_threads(tmp_path):
    roots = []
    for i in range(4):
        d = tmp_path / f"w{i}"
        d.mkdir()
        (d / "id.txt").write_text(f"ws{i}\n", encoding="utf-8")
        roots.append(d)
    sets = [by_name(tools.make_tools(d)) for d in roots]
    order = [0, 1, 2, 3] * 5
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda i: sets[i]["read_file"].invoke({"path": "id.txt"}), order))
    assert all(f"ws{i}" in out for i, out in zip(order, outs))


def test_fixed_toolset_ignores_global_workspace(tmp_path, monkeypatch):
    fixed, other = tmp_path / "fixed", tmp_path / "other"
    fixed.mkdir()
    other.mkdir()
    (fixed / "f.py").write_text("FIXED\n", encoding="utf-8")
    t = by_name(tools.make_tools(fixed))
    monkeypatch.setattr(tools, "WORKSPACE", other)
    assert "FIXED" in t["read_file"].invoke({"path": "f.py"})


def test_fixed_toolset_blocks_secrets_and_escape(tmp_path):
    (tmp_path / ".env").write_text("K=1\n", encoding="utf-8")
    t = by_name(tools.make_tools(tmp_path))
    with pytest.raises(PermissionError):
        t["read_file"].invoke({"path": ".env"})
    with pytest.raises(ValueError):
        t["read_file"].invoke({"path": "../outside.py"})


def test_safe_path_resolves_under_root(tmp_path):
    (tmp_path / "pkg").mkdir()
    assert tools.safe_path(tmp_path, "pkg/a.py") == (tmp_path / "pkg" / "a.py").resolve()
