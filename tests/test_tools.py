"""工具綁定資料夾:網頁同時跑多個任務、各自在不同資料夾時不能互相覆蓋。"""
from __future__ import annotations

import subprocess
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


def saved_patch(root, out):
    """propose_patch 的回傳第一行是「已存成 <路徑>,等工程師審核後套用。」"""
    return root / out.split()[1].split(",", 1)[0]


# calc.py 的修正 hunk:要帶後置 context(兩行空白行)。git 規定沒有後置 context 的 hunk 必須對到檔尾,calc.py 後面還有內容,會被拒收
CALC_HUNK = "--- a/calc.py\n+++ b/calc.py\n@@ -1,4 +1,4 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n \n \n"


def test_propose_patch_two_file_legacy_diff(tmp_path, monkeypatch):
    """兩個檔的傳統格式 diff:以前加了 --recount 會把第二個檔的 ---/+++ 算進第一個 hunk,整個被拒(最終審查重現)。"""
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8", newline="\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                                                     encoding="utf-8", newline="\n")
    monkeypatch.setattr(tools, "_run_sandbox", lambda root, target, **kw: "exit code 0\n3 passed")
    t = by_name(tools.make_tools(tmp_path))
    diff = (CALC_HUNK +
            "--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n@@ -4,1 +4,4 @@\n def test_add():\n     assert add(2, 3) == 5\n"
            "+\n+\n+def test_add_negative():\n+    assert add(-1, 1) == 0\n")
    out = t["propose_patch"].invoke({"path": "calc.py", "unified_diff": diff, "reason": "修加法", "test_target": "."})
    assert out.startswith("已存成"), out
    saved = saved_patch(tmp_path, out)
    check = subprocess.run(["git", "apply", "--check", str(saved)], cwd=tmp_path, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr  # 工程師不加任何旗標直接套用也要成功


def test_propose_patch_in_folder_nested_in_parent_repo(tmp_path, monkeypatch):
    """工作區巢狀在別的 git repo 裡(demo_workspace 就是):diff --git 格式不能被靜靜略過(最終審查重現)。"""
    outer = tmp_path / "outer"
    ws = outer / "ws"
    ws.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=outer, check=True)
    (ws / "calc.py").write_text(CALC, encoding="utf-8", newline="\n")
    monkeypatch.setattr(tools, "_run_sandbox", lambda root, target, **kw: "exit code 0\n2 passed")
    t = by_name(tools.make_tools(ws))
    good = "diff --git a/calc.py b/calc.py\nindex 1111111..2222222 100644\n" + CALC_HUNK
    bad = good.replace("-    return a - b", "-    return a * b")
    assert t["propose_patch"].invoke({"path": "calc.py", "unified_diff": bad, "reason": "x", "test_target": "."}).startswith("diff ")
    out = t["propose_patch"].invoke({"path": "calc.py", "unified_diff": good, "reason": "修加法", "test_target": "."})
    assert out.startswith("已存成"), out
    saved = saved_patch(ws, out)
    subprocess.run(["git", "-C", str(ws), "apply", str(saved)], check=True, capture_output=True)  # 頁面複製的指令
    assert (ws / "calc.py").read_text(encoding="utf-8").startswith("def add(a, b):\n    return a + b\n")


def test_reason_cannot_smuggle_extra_hunks(tmp_path, monkeypatch):
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8", newline="\n")
    (tmp_path / "other.py").write_text("X = 1\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(tools, "_run_sandbox", lambda root, target, **kw: "exit code 0\n2 passed")
    t = by_name(tools.make_tools(tmp_path))
    reason = "修加法\n--- a/other.py\n+++ b/other.py\n@@ -1 +1 @@\n-X = 1\n+X = 2\n"
    out = t["propose_patch"].invoke({"path": "calc.py", "unified_diff": CALC_HUNK, "reason": reason, "test_target": "."})
    saved = saved_patch(tmp_path, out)
    numstat = subprocess.run(["git", "apply", "--numstat", str(saved)], cwd=tmp_path, capture_output=True, text=True).stdout
    assert "other.py" not in numstat and "calc.py" in numstat


def test_patch_files_unique_within_same_second(tmp_path, monkeypatch):
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8", newline="\n")
    monkeypatch.setattr(tools, "_run_sandbox", lambda root, target, **kw: "exit code 0\n2 passed")
    t = by_name(tools.make_tools(tmp_path))
    for _ in range(2):
        t["propose_patch"].invoke({"path": "calc.py", "unified_diff": CALC_HUNK, "reason": "x", "test_target": "."})
    assert len(list((tmp_path / ".agent_patches").glob("*.patch"))) == 2


def test_normalize_diff_drops_blank_separator_between_files():
    """模型在兩個檔的 diff 之間空一行;以前被當成 context,第一個 hunk 多一行空白就套不上(10/8 實測卡滿 15 步)。"""
    raw = ("--- a/calc.py\n+++ b/calc.py\n@@ -1,4 +1,4 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n \n \n\n"
           "--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n@@ -6,4 +6,6 @@\n \n \n def test_average():\n"
           "     assert average([2, 4, 6]) == 4\n+\n+def test_add_negative():\n")
    out = tools.normalize_diff(raw)
    assert "@@ -1,4 +1,4 @@" in out and "@@ -6,4 +6,6 @@" in out and "\n\n" not in out


def test_propose_patch_two_files_with_blank_separator(tmp_path, monkeypatch):
    """10/8 實測:Nemotron 給的兩檔 diff 中間空一行,內容正確卻被判套不上,卡滿 15 步、約 11 萬 tokens。"""
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8", newline="\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(
        "from calc import add, average\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_average():\n    assert average([2, 4, 6]) == 4\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(tools, "_run_sandbox", lambda root, target, **kw: "exit code 0\n3 passed")
    t = by_name(tools.make_tools(tmp_path))
    diff = ("--- a/calc.py\n+++ b/calc.py\n@@ -1,4 +1,4 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n \n \n\n"
            "--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n@@ -6,4 +6,9 @@\n \n \n def test_average():\n"
            "     assert average([2, 4, 6]) == 4\n+\n+def test_add_negative():\n+    assert add(-1, -1) == -2\n"
            "+    assert add(-5, 3) == -2\n")
    out = t["propose_patch"].invoke({"path": ".", "unified_diff": diff, "reason": "修加法並補測試", "test_target": "."})
    assert out.startswith("已存成"), out
    check = subprocess.run(["git", "apply", "--check", str(saved_patch(tmp_path, out))], cwd=tmp_path,
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
