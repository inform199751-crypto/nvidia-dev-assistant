"""API 與 SSE(FastAPI TestClient,假模型,不需金鑰)。"""
from __future__ import annotations

import io
import json
import threading
import zipfile

from fastapi.testclient import TestClient

from helpers import InlineExecutor, ManualExecutor, ai, scripted_registry
from token_guard import TokenQuota
from web.app import create_app
from web.conversations import ConversationManager
from web.workspaces import WorkspaceRegistry


def build(tmp_path, script, executor=None):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "workspaces.yaml").write_text("workspaces:\n  - name: demo\n    path: proj\n", encoding="utf-8")
    workspaces = WorkspaceRegistry(tmp_path / "workspaces.yaml", base_dir=tmp_path, uploads_dir=tmp_path / "uploads")
    reg, seen = scripted_registry(script)
    mgr = ConversationManager(tmp_path / "conversations", workspaces, reg, TokenQuota(10**7, 2),
                              executor=executor or InlineExecutor())
    return TestClient(create_app(mgr, workspaces, reg, poll_seconds=0.01, allowed_hosts=("testserver",))), seen


def sse(client, url, headers=None):
    text = client.get(url, headers=headers or {}).text
    events = []
    for block in text.strip().split("\n\n"):
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if data:
            events.append(json.loads(data[0]))
    return events


def test_workspaces_and_estimate(tmp_path):
    client, _ = build(tmp_path, [])
    assert [w["id"] for w in client.get("/api/workspaces").json()] == ["demo"]
    est = client.get("/api/estimate", params={"workspace": "demo"}).json()
    assert est["files"] == 1 and est["ok"] is True and est["role"] == "code" and est["tokens"] > 0
    assert client.get("/api/estimate", params={"workspace": "nope"}).status_code == 404


def test_conversation_flow_with_follow_up(tmp_path):
    client, _ = build(tmp_path, [
        ai(tool_calls=[{"name": "read_file", "args": {"path": "calc.py"}, "id": "c1"}]),
        ai("找到了"),
        ai("追問答案"),
    ])
    r = client.post("/api/conversations", json={"workspace": "demo", "question": "add 錯了", "mode": "fix"})
    assert r.status_code == 200
    cid, tid = r.json()["conversation_id"], r.json()["turn_id"]
    events = sse(client, f"/api/conversations/{cid}/turns/{tid}/events")
    assert [e["type"] for e in events] == ["start", "step", "tool_call", "tool_result", "step", "answer", "end"]
    assert events[-2]["text"] == "找到了"
    tid2 = client.post(f"/api/conversations/{cid}/turns", json={"question": "還有呢", "mode": "fix"}).json()["turn_id"]
    assert sse(client, f"/api/conversations/{cid}/turns/{tid2}/events")[-2]["text"] == "追問答案"
    detail = client.get(f"/api/conversations/{cid}").json()
    assert [t["status"] for t in detail["turns"]] == ["done", "done"] and "messages" not in detail
    assert client.get("/api/conversations").json()[0]["id"] == cid


def test_sse_resumes_after_last_event_id(tmp_path):
    client, _ = build(tmp_path, [ai("好")])
    ids = client.post("/api/conversations", json={"workspace": "demo", "question": "q", "mode": "fix"}).json()
    url = f"/api/conversations/{ids['conversation_id']}/turns/{ids['turn_id']}/events"
    full = sse(client, url)
    resumed = sse(client, url, headers={"Last-Event-ID": "1"})
    assert [e["seq"] for e in resumed] == [e["seq"] for e in full[2:]]
    assert sse(client, url, headers={"Last-Event-ID": str(full[-1]["seq"])}) == []  # 結束後重連不會卡住


def test_busy_and_cancel(tmp_path):
    client, seen = build(tmp_path, [ai("不該被呼叫")], executor=ManualExecutor())
    ids = client.post("/api/conversations", json={"workspace": "demo", "question": "q", "mode": "fix"}).json()
    cid, tid = ids["conversation_id"], ids["turn_id"]
    assert client.post(f"/api/conversations/{cid}/turns", json={"question": "q2"}).status_code == 409
    assert client.post(f"/api/conversations/{cid}/turns/{tid}/cancel").json() == {"ok": True}
    client.app.state.manager.executor.run_all()
    assert client.get(f"/api/conversations/{cid}").json()["turns"][0]["status"] == "cancelled"
    assert seen == []


def test_validation_and_404(tmp_path):
    client, _ = build(tmp_path, [])
    assert client.post("/api/conversations", json={"workspace": "nope", "question": "q", "mode": "fix"}).status_code == 400
    assert client.post("/api/conversations", json={"workspace": "demo", "question": " ", "mode": "fix"}).status_code == 400
    assert client.get("/api/conversations/nope").status_code == 404
    assert client.get("/api/conversations/nope/turns/x/events").status_code == 404
    assert client.post("/api/conversations/nope/turns/x/cancel").status_code == 404


def test_upload_endpoint(tmp_path):
    client, _ = build(tmp_path, [])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("src/a.py", "A = 1\n")
        zf.writestr(".env", "K=1")
    r = client.post("/api/uploads", files=[("files", ("proj.zip", buf.getvalue(), "application/zip"))])
    assert r.status_code == 200 and r.json()["skipped"] == [".env"]
    assert r.json()["workspace"]["name"] == "上傳:proj.zip"
    bad = client.post("/api/uploads", files=[("files", ("x.zip", b"not zip", "application/zip"))])
    assert bad.status_code == 400 and "zip" in bad.json()["detail"]


def test_estimate_reports_when_too_large(tmp_path, monkeypatch):
    client, _ = build(tmp_path, [])
    registry = client.app.state.manager.registry
    monkeypatch.setattr(registry, "budget", lambda role: 10)  # 讓任何內容都超過上限
    est = client.get("/api/estimate", params={"workspace": "demo"}).json()
    assert est["ok"] is False and est["role"] is None and est["limit"] == 10 and est["tokens"] > 10


def test_sse_pushes_events_that_arrive_after_connect(tmp_path):
    client, _ = build(tmp_path, [ai("晚到的答案")], executor=ManualExecutor())
    ids = client.post("/api/conversations", json={"workspace": "demo", "question": "q", "mode": "fix"}).json()
    url = f"/api/conversations/{ids['conversation_id']}/turns/{ids['turn_id']}/events"
    timer = threading.Timer(0.2, client.app.state.manager.executor.run_all)  # 連上之後才開始跑
    timer.start()
    try:
        events = sse(client, url)
    finally:
        timer.join()
    assert [e["type"] for e in events] == ["start", "step", "answer", "end"]
    assert events[2]["text"] == "晚到的答案"


def test_sse_after_param_and_204_when_finished(tmp_path):
    client, _ = build(tmp_path, [ai("好")])
    ids = client.post("/api/conversations", json={"workspace": "demo", "question": "q", "mode": "fix"}).json()
    url = f"/api/conversations/{ids['conversation_id']}/turns/{ids['turn_id']}/events"
    full = sse(client, url)
    assert [e["seq"] for e in sse(client, url + "?after=1")] == [e["seq"] for e in full[2:]]
    assert [e["seq"] for e in sse(client, url + "?after=-5")] == [e["seq"] for e in full]  # 負數不會從尾巴開始
    finished = client.get(url, headers={"Last-Event-ID": str(full[-1]["seq"])})
    assert finished.status_code == 204


def test_untrusted_host_rejected(tmp_path):
    client, _ = build(tmp_path, [])
    assert client.get("/api/workspaces", headers={"host": "evil.example"}).status_code == 400


def test_index_and_static_files_served(tmp_path):
    client, _ = build(tmp_path, [])
    page = client.get("/")
    assert page.status_code == 200 and "<title>研發助理</title>" in page.text
    for path in ["/static/app.js", "/static/style.css",
                 "/static/vendor/marked.min.js", "/static/vendor/purify.min.js"]:
        assert client.get(path).status_code == 200, path


def test_index_sends_content_security_policy(tmp_path):
    client, _ = build(tmp_path, [])
    csp = client.get("/").headers["content-security-policy"]
    assert "default-src 'self'" in csp and "media-src 'none'" in csp


def test_cross_site_writes_rejected(tmp_path):
    client, _ = build(tmp_path, [])
    body = {"workspace": "demo", "question": "q", "mode": "fix"}
    assert client.post("/api/conversations", json=body, headers={"origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/conversations", json=body, headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert client.get("/api/workspaces", headers={"origin": "https://evil.example"}).status_code == 200


def test_static_files_not_cached(tmp_path):
    client, _ = build(tmp_path, [])
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"
