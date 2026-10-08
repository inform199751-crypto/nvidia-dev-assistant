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


CALC = "def add(a, b):\n    return a - b\n\n\ndef average(values):\n    return add(sum(values), 0) / len(values)\n"


def test_propose_patch_keeps_trailing_blank_context_line(tmp_path, monkeypatch):
    """模型的 diff 最後一行常是只有一個空白的空行 context;以前 rstrip() 把它刪掉,git 判定 corrupt patch(10/8 實測)。"""
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8", newline="\n")
    monkeypatch.setattr(tools, "_run_sandbox", lambda root, target, **kw: "exit code 0\n2 passed")
    t = by_name(tools.make_tools(tmp_path))
    diff = "--- a/calc.py\n+++ b/calc.py\n@@ -1,4 +1,4 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n \n "
    out = t["propose_patch"].invoke({"path": "calc.py", "unified_diff": diff, "reason": "修加法", "test_target": "."})
    assert out.startswith("已存成"), out


def test_propose_patch_tolerates_miscounted_hunk_header(tmp_path, monkeypatch):
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8", newline="\n")
    monkeypatch.setattr(tools, "_run_sandbox", lambda root, target, **kw: "exit code 0\n2 passed")
    t = by_name(tools.make_tools(tmp_path))
    diff = "--- a/calc.py\n+++ b/calc.py\n@@ -1,9 +1,9 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n \n"  # 標頭寫 9 行,實際 4 行
    out = t["propose_patch"].invoke({"path": "calc.py", "unified_diff": diff, "reason": "修加法", "test_target": "."})
    assert out.startswith("已存成"), out
    assert (tmp_path / "calc.py").read_text(encoding="utf-8") == CALC  # 原檔沒被改


def test_normalize_diff_recounts_and_strips_git_headers():
    raw = ("diff --git a/calc.py b/calc.py\nindex 1234567..89abcde 100644\n--- a/calc.py\n+++ b/calc.py\n"
           "@@ -1,9 +1,9 @@ def add\n def add(a, b):\n-    return a - b\n+    return a + b\n\n")
    assert tools.normalize_diff(raw) == ("--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@ def add\n"
                                         " def add(a, b):\n-    return a - b\n+    return a + b\n")
