"""run_agent 的事件、資料夾綁定、追問與停止。"""
from __future__ import annotations

import copy

from langchain_core.messages import HumanMessage

import agent
import tools
from agent import omit_attachments, run_agent
from helpers import CONFIG, ai, scripted_registry
from token_guard import TokenQuota


def quota():
    return TokenQuota(10**7, 2)


def test_events_in_order(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    reg, _ = scripted_registry([
        ai(tool_calls=[{"name": "read_file", "args": {"path": "calc.py"}, "id": "c1"}]),
        ai("找到了"),
    ])
    events = []
    r = run_agent("add 錯了", registry=reg, quota=quota(), workspace=tmp_path, on_event=events.append)
    assert [e["type"] for e in events] == ["start", "step", "tool_call", "tool_result", "step", "answer"]
    assert events[0]["role"] == "code" and events[0]["model"] == "deepseek-ai/deepseek-v4.1-flash"
    assert events[0]["mode"] == "fix" and events[0]["est_tokens"] > 0
    assert events[2]["args"] == {"path": "calc.py"}
    assert events[3]["ok"] is True and "return a - b" in events[3]["preview"]
    assert events[5]["text"] == "找到了" and r.messages[-1].content == "找到了"


def test_workspace_param_binds_attachments_and_tools(tmp_path, monkeypatch):
    other, ws = tmp_path / "other", tmp_path / "ws"
    other.mkdir()
    ws.mkdir()
    (ws / "a.py").write_text("A = 1\n", encoding="utf-8")
    monkeypatch.setattr(tools, "WORKSPACE", other)
    reg, seen = scripted_registry([
        ai(tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "c1"}]),
        ai("ok"),
    ])
    run_agent("看", registry=reg, quota=quota(), workspace=ws, attachments=["a.py"])
    assert "A = 1" in seen[0][1][1].content       # 附檔從 ws 讀
    assert "A = 1" in seen[1][1][-1].content      # 工具也綁 ws


def test_tool_error_marks_result_not_ok(tmp_path):
    reg, _ = scripted_registry([
        ai(tool_calls=[{"name": "read_file", "args": {"path": "../x.py"}, "id": "c1"}]),
        ai("好"),
    ])
    events = []
    run_agent("讀", registry=reg, quota=quota(), workspace=tmp_path, on_event=events.append)
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["ok"] is False and "超出工作區" in result["preview"]


def test_patch_event_from_propose_patch(tmp_path, monkeypatch):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(tools, "_run_sandbox", lambda root, target, **kw: "exit code 0\n2 passed")
    diff = ("--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n"
            "-    return a - b\n+    return a + b\n")
    reg, _ = scripted_registry([
        ai(tool_calls=[{"name": "propose_patch", "id": "p1",
                        "args": {"path": "calc.py", "unified_diff": diff, "reason": "修", "test_target": "."}}]),
        ai("改好了"),
    ])
    events = []
    run_agent("修", registry=reg, quota=quota(), workspace=tmp_path, on_event=events.append)
    patch = next(e for e in events if e["type"] == "patch")
    assert patch["path"] == "calc.py" and patch["diff"] == diff and patch["passed"] is True
    assert patch["patch_file"].startswith(".agent_patches/") and patch["patch_file"].endswith(".patch")
    assert "2 passed" in patch["test_output"]


def test_should_stop_before_next_step(tmp_path):
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    reg, seen = scripted_registry([
        ai(tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "c1"}]),
        ai("不該跑到這裡"),
    ])
    checks = {"n": 0}

    def stop():
        checks["n"] += 1
        return checks["n"] > 1  # 第 1 步前不停,第 2 步前停

    r = run_agent("看", registry=reg, quota=quota(), workspace=tmp_path, should_stop=stop)
    assert r.stopped and r.steps == 1 and len(seen) == 1


def test_history_continues_without_reattaching_files(tmp_path):
    (tmp_path / "a.py").write_text("SECRET_SAUCE = 42\n", encoding="utf-8")
    reg, seen = scripted_registry([ai("第一輪"), ai("第二輪")])
    first = run_agent("看 a.py", registry=reg, quota=quota(), workspace=tmp_path, attachments=["a.py"])
    second = run_agent("那它是常數嗎", registry=reg, quota=quota(), workspace=tmp_path, history=first.messages)
    sent = seen[1][1]
    assert sent[0].type == "system" and sum(1 for m in sent if m.type == "system") == 1
    assert "SECRET_SAUCE" not in sent[1].content and '<file path="a.py" omitted="true"/>' in sent[1].content
    assert sent[2].content == "第一輪" and "那它是常數嗎" in sent[-1].content
    assert second.answer == "第二輪"
    assert "SECRET_SAUCE" in first.messages[1].content  # 原物件沒被改


def test_omit_attachments_after_whole_mode_restores_tool_hint():
    whole = HumanMessage('<file path="a.py" lines="1-1">\n    1| x\n</file>\n\n'
                         + agent.WHOLE_NOTE + "\n\n<task>\n寫文件\n</task>")
    out = omit_attachments([whole])[0].content
    assert '<file path="a.py" omitted="true"/>' in out
    assert "沒有工具可用" not in out and agent.OMITTED_NOTE in out


def test_model_switch_event(tmp_path):
    cfg = copy.deepcopy(CONFIG)
    cfg["roles"]["code"].update(context_window=6000, max_output=1000)
    cfg["limits"]["tool_output_max_tokens"] = 5000
    (tmp_path / "a.py").write_text("y = 2  # padding padding padding\n" * 1500, encoding="utf-8")
    reg, _ = scripted_registry([
        ai(tool_calls=[{"name": "read_file", "args": {"path": "a.py", "end_line": 100}, "id": "c1"}]),
        ai(tool_calls=[{"name": "read_file", "args": {"path": "a.py", "start_line": 101, "end_line": 400}, "id": "c2"}]),
        ai("讀完了"),
    ], cfg)
    events = []
    run_agent("讀 a.py", registry=reg, quota=quota(), workspace=tmp_path, on_event=events.append)
    switches = [e for e in events if e["type"] == "model_switch"]
    assert len(switches) == 1
    assert switches[0]["from_role"] == "code" and switches[0]["to_role"] == "long_context"
