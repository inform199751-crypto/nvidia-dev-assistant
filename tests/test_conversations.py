"""對話管理:狀態流轉、存檔與重載、追問、排隊、取消、錯誤。"""
from __future__ import annotations

import json

import pytest

from agent import AgentResult
from helpers import InlineExecutor, ManualExecutor, ai, scripted_registry
from token_guard import TokenQuota
from web.conversations import Busy, ConversationManager
from web.workspaces import WorkspaceRegistry


@pytest.fixture
def workspaces(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (proj / "notes.md").write_text("# 筆記\n", encoding="utf-8")
    (tmp_path / "workspaces.yaml").write_text("workspaces:\n  - name: demo\n    path: proj\n", encoding="utf-8")
    return WorkspaceRegistry(tmp_path / "workspaces.yaml", base_dir=tmp_path, uploads_dir=tmp_path / "uploads")


def make_manager(tmp_path, workspaces, script, executor=None, quota=None):
    reg, seen = scripted_registry(script)
    mgr = ConversationManager(tmp_path / "conversations", workspaces, reg, quota or TokenQuota(10**7, 2),
                              executor=executor or InlineExecutor())
    return mgr, seen


def types(turn):
    return [e["type"] for e in turn.events]


def test_turn_runs_and_is_saved(tmp_path, workspaces):
    mgr, _ = make_manager(tmp_path, workspaces, [
        ai(tool_calls=[{"name": "read_file", "args": {"path": "calc.py"}, "id": "c1"}]),
        ai("找到了"),
    ])
    cid, tid = mgr.start("demo", "add 錯了", "fix")
    turn = mgr.get(cid).turn(tid)
    assert turn.status == "done"
    assert types(turn) == ["start", "step", "tool_call", "tool_result", "step", "answer", "end"]
    assert [e["seq"] for e in turn.events] == list(range(7)) and all("t" in e for e in turn.events)
    saved = json.loads((tmp_path / "conversations" / f"{cid}.json").read_text(encoding="utf-8"))
    assert saved["turns"][0]["status"] == "done" and saved["messages"]
    again, _ = make_manager(tmp_path, workspaces, [])
    assert again.get(cid).turn(tid).events == turn.events
    assert again.list()[0]["title"] == "add 錯了"


def test_follow_up_carries_history(tmp_path, workspaces):
    mgr, seen = make_manager(tmp_path, workspaces, [ai("第一輪答案"), ai("第二輪答案")])
    cid, _ = mgr.start("demo", "看 calc.py", "fix", {"files": "calc.py"})
    tid2 = mgr.add_turn(cid, "那 average 呢", "fix")
    sent = seen[1][1]
    assert any(m.content == "第一輪答案" for m in sent)
    assert "return a - b" not in sent[1].content  # 第一輪的附檔在追問時省略
    assert mgr.get(cid).turn(tid2).status == "done"


def test_busy_while_turn_pending(tmp_path, workspaces):
    mgr, _ = make_manager(tmp_path, workspaces, [ai("x")], executor=ManualExecutor())
    cid, _ = mgr.start("demo", "q1", "fix")
    with pytest.raises(Busy):
        mgr.add_turn(cid, "q2", "fix")


def test_cancel_before_start_skips_model(tmp_path, workspaces):
    ex = ManualExecutor()
    mgr, seen = make_manager(tmp_path, workspaces, [ai("不該被呼叫")], executor=ex)
    cid, tid = mgr.start("demo", "q", "fix")
    mgr.cancel(cid, tid)
    ex.run_all()
    turn = mgr.get(cid).turn(tid)
    assert turn.status == "cancelled" and seen == [] and types(turn) == ["end"]


def test_unfinished_turn_marked_interrupted_on_reload(tmp_path, workspaces):
    mgr, _ = make_manager(tmp_path, workspaces, [], executor=ManualExecutor())
    cid, tid = mgr.start("demo", "q", "fix")
    reloaded, _ = make_manager(tmp_path, workspaces, [])
    turn = reloaded.get(cid).turn(tid)
    assert turn.status == "interrupted" and turn.events[-1]["type"] == "end"
    assert turn.events[-1]["status"] == "interrupted"


def test_quota_error_becomes_error_event(tmp_path, workspaces):
    mgr, _ = make_manager(tmp_path, workspaces, [ai("x")], quota=TokenQuota(10, 2))
    cid, tid = mgr.start("demo", "q", "fix")
    turn = mgr.get(cid).turn(tid)
    assert turn.status == "failed" and types(turn)[-2:] == ["error", "end"]
    err = next(e for e in turn.events if e["type"] == "error")
    assert err["kind"] == "quota" and "剩 10 tokens" in err["message"]  # 要告訴使用者還剩多少


def test_whole_mode_attaches_every_text_file(tmp_path, workspaces):
    mgr, seen = make_manager(tmp_path, workspaces, [ai("# 文件")])
    mgr.start("demo", "寫文件", "whole")
    prompt = seen[0][1][1].content
    assert '<file path="calc.py"' in prompt and '<file path="notes.md"' in prompt and "沒有工具可用" in prompt


def test_validation(tmp_path, workspaces):
    mgr, _ = make_manager(tmp_path, workspaces, [])
    with pytest.raises(ValueError):
        mgr.start("demo", "  ", "fix")
    with pytest.raises(ValueError):
        mgr.start("demo", "q", "chat")
    with pytest.raises(KeyError):
        mgr.start("nope", "q", "fix")


def test_expired_upload_follow_up_reports_workspace_error(tmp_path, workspaces):
    mgr, _ = make_manager(tmp_path, workspaces, [ai("第一輪")])
    ws, _ = workspaces.add_upload([("a.py", b"A = 1\n")])
    cid, _ = mgr.start(ws.id, "看 a.py", "fix")
    meta = ws.path.parent / "meta.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["created_at"] -= 25 * 3600
    meta.write_text(json.dumps(data), encoding="utf-8")
    tid = mgr.add_turn(cid, "再看一次", "fix")
    err = next(e for e in mgr.get(cid).turn(tid).events if e["type"] == "error")
    assert err["kind"] == "workspace"


def test_cancel_while_running_stops_after_current_step(tmp_path, workspaces):
    ex = ManualExecutor()
    holder = {}

    def fake_agent(question, *, on_event, should_stop, **kwargs):
        on_event({"type": "step", "step": 1, "role": "code", "model": "m"})
        holder["mgr"].cancel(holder["cid"], holder["tid"])  # 使用者在等模型回應時按了停止
        assert should_stop()
        return AgentResult("[已停止]", "code", 1, 50, [], stopped=True)

    reg, _ = scripted_registry([])
    mgr = ConversationManager(tmp_path / "conversations", workspaces, reg, TokenQuota(10**7, 2),
                              executor=ex, agent_fn=fake_agent)
    holder["mgr"] = mgr
    holder["cid"], holder["tid"] = mgr.start("demo", "q", "fix")
    ex.run_all()
    turn = mgr.get(holder["cid"]).turn(holder["tid"])
    assert turn.status == "cancelled" and types(turn) == ["step", "end"]
    assert turn.events[-1]["steps"] == 1 and turn.events[-1]["tokens"] == 50


def test_nvidia_overload_becomes_busy_error(tmp_path, workspaces):
    def overloaded(question, **kwargs):
        raise Exception("[503] {'message': 'Service temporarily overloaded'}")

    reg, _ = scripted_registry([])
    mgr = ConversationManager(tmp_path / "conversations", workspaces, reg, TokenQuota(10**7, 2),
                              executor=InlineExecutor(), agent_fn=overloaded)
    cid, tid = mgr.start("demo", "q", "fix")
    turn = mgr.get(cid).turn(tid)
    err = next(e for e in turn.events if e["type"] == "error")
    assert turn.status == "failed" and err["kind"] == "busy" and "服務忙線" in err["message"]


def test_unknown_role_is_not_reported_as_missing_folder(tmp_path, workspaces):
    mgr, _ = make_manager(tmp_path, workspaces, [ai("x")])
    cid, tid = mgr.start("demo", "q", "fix", {"role": "nope"})
    err = next(e for e in mgr.get(cid).turn(tid).events if e["type"] == "error")
    assert err["kind"] == "other" and "nope" in err["message"]


def test_save_failure_does_not_abort_the_turn(tmp_path, workspaces, monkeypatch):
    mgr, _ = make_manager(tmp_path, workspaces, [ai("好")])

    def locked(*args, **kwargs):
        raise PermissionError("檔案被防毒軟體鎖住")

    monkeypatch.setattr("web.conversations.os.replace", locked)
    cid, tid = mgr.start("demo", "q", "fix")
    turn = mgr.get(cid).turn(tid)
    assert turn.status == "done" and types(turn)[-1] == "end"
