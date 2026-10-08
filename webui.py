"""網頁介面:在瀏覽器裡跟小助理聊天(像 Claude 的網頁版)。

    python webui.py            → 開 http://127.0.0.1:8765 並自動打開瀏覽器

功能跟終端機的互動模式一樣:回答邊產生邊顯示、工具進度、上網前確認、套用 patch 前確認、新對話、記住規則。
單人使用、只聽本機。安全:API 要帶啟動時隨機產生的 token(其他網站讀不到頁面,拿不到 token),
並檢查 Host 標頭(擋 DNS rebinding);只用標準函式庫,不需要另外裝套件。
"""
from __future__ import annotations

import argparse
import json
import logging
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

import agent
import tools as toolbox
import web

PAGE = Path(__file__).with_name("webui.html")
CONFIRM_TIMEOUT = 300  # 秒;沒人回答就當作不允許
MAX_DIFF_CHARS = 20000
MAX_BODY = 1_000_000  # POST 內容上限(問題文字、規則)


class Hub:
    """一段對話的狀態 + 事件紀錄。agent 在背景執行緒跑,事件記下來給瀏覽器(重新整理時從頭重播這段對話)。"""

    def __init__(self, new_session: Callable[[Callable], agent.AgentSession], status: str = "", model: str = ""):
        self.new_session = new_session
        self.status = status
        self.model = model  # 頁面上方顯示的模型名稱
        self.cond = threading.Condition()
        self.events: list[dict] = []
        self.view_start = 0  # 最近一次「新對話」之後的第一個事件
        self.busy = False
        self.note = ""  # 下一個問題要先告訴模型的事(例如已經套用 patch)
        self.pending: dict[str, list] = {}  # 等使用者回答的確認:id → [threading.Event, 答案]
        self.always: set[str] = set()
        self.session = new_session(self.publish)

    # ---------- 事件 ----------

    def publish(self, kind: str, data: dict | None = None) -> None:
        with self.cond:
            self.events.append({"seq": len(self.events) + 1, "kind": kind, **(data or {})})
            self.cond.notify_all()

    def events_since(self, seq: int, timeout: float) -> list[dict]:
        with self.cond:
            start = max(seq, self.view_start)
            if len(self.events) <= start:
                self.cond.wait(timeout)
            return self.events[max(seq, self.view_start):]

    # ---------- 對話 ----------

    def ask(self, text: str) -> bool:
        with self.cond:
            if self.busy:
                return False
            self.busy = True
        self.publish("user", {"text": text})
        threading.Thread(target=self._run, args=(text,), daemon=True).start()
        return True

    def _run(self, text: str) -> None:
        before = len(toolbox.PATCHES)
        try:
            r = self.session.ask(self.note + text, agent.file_mentions(text))
            self.note = ""
            self.publish("done", {"role": r.role, "steps": r.steps, "tokens": r.tokens_used,
                                  "total": self.session.tokens_used, "answer": r.answer})
            if len(toolbox.PATCHES) > before:  # 這一輪提了 patch:問要不要套用(沒通過測試也給看,但會警告)
                index = len(toolbox.PATCHES) - 1
                patch, passed = toolbox.PATCHES[index]
                self.publish("offer_apply", {"patch": index, "passed": passed,
                                             "name": patch.relative_to(toolbox.WORKSPACE).as_posix(),
                                             "diff": patch.read_text(encoding="utf-8")[:MAX_DIFF_CHARS]})
        except Exception as e:  # 額度用完、模型全掛:這輪已經退掉,對話還能繼續
            self.publish("error", {"text": f"{type(e).__name__}: {e}"})
        finally:
            with self.cond:
                self.busy = False
                self.cond.notify_all()

    def clear(self) -> bool:
        with self.cond:
            if self.busy:
                return False
            self.session, self.note = self.new_session(self.publish), ""
        self.publish("clear")
        with self.cond:
            self.view_start = len(self.events)
        return True

    def remember(self, rule: str) -> None:
        path = agent.remember(rule)
        self.session.refresh_system_prompt()
        self.publish("notice", {"text": f"已記住,寫進 {path.name}:{rule}"})

    # ---------- 需要使用者確認的動作 ----------

    def confirm(self, action: str, detail: str) -> bool:
        """給 web.CONFIRM 用:在網頁上跳出確認卡片,等使用者按按鈕(agent 執行緒會在這裡等)。"""
        if action in self.always:
            self.publish("notice", {"text": f"已允許{action}:{detail}"})
            return True
        cid = secrets.token_hex(4)
        slot = [threading.Event(), "n"]
        with self.cond:
            self.pending[cid] = slot
        self.publish("confirm", {"id": cid, "action": action, "detail": detail})
        slot[0].wait(CONFIRM_TIMEOUT)
        with self.cond:
            self.pending.pop(cid, None)
        if slot[1] == "a":
            self.always.add(action)
        self.publish("confirm_done", {"id": cid, "answer": slot[1]})
        return slot[1] in ("y", "a")

    def answer(self, cid: str, answer: str) -> bool:
        with self.cond:
            slot = self.pending.get(cid)
        if not slot:
            return False
        slot[1] = answer if answer in ("y", "a", "n") else "n"
        slot[0].set()
        return True

    def apply(self, index: int, do_apply: bool) -> str:
        with self.cond:
            if self.busy:
                return "小助理還在回答,等這一輪結束再套用"
        if not 0 <= index < len(toolbox.PATCHES):
            return "找不到這個 patch"
        patch, _ = toolbox.PATCHES[index]
        name = patch.relative_to(toolbox.WORKSPACE).as_posix()
        if not do_apply:
            result = "沒有套用,patch 留著給你之後審核。"
        else:
            result = toolbox.apply_patch(patch)
            if result.startswith("已套用"):
                self.note += f"<note>使用者已把 {name} 套用到工作區,相關檔案內容已經改變,需要時請重讀。</note>\n"
        self.publish("applied", {"patch": index, "text": result})
        return result


def make_handler(hub: Hub, token: str, port: int) -> type[BaseHTTPRequestHandler]:
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # 安靜一點,不要每個請求都印
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            if self.headers.get("Host") in allowed_hosts:
                return True
            self._json(403, {"error": "只接受本機連線"})
            return False

        def _token_ok(self, given: str | None) -> bool:
            if given and secrets.compare_digest(given, token):
                return True
            self._json(403, {"error": "token 不對"})
            return False

        def do_GET(self) -> None:
            if not self._host_ok():
                return
            url = urlsplit(self.path)
            if url.path == "/":
                page = PAGE.read_text(encoding="utf-8").replace("__TOKEN__", token)
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                if self._token_ok(self.headers.get("X-Token")):
                    self._json(200, {"workspace": str(toolbox.WORKSPACE), "status": hub.status, "busy": hub.busy,
                                     "model": hub.model,
                                     "instructions": (toolbox.WORKSPACE / agent.INSTRUCTIONS_FILE).is_file()})
            elif url.path == "/api/events":
                qs = parse_qs(url.query)
                if self._token_ok((qs.get("token") or [None])[0]):  # EventSource 不能帶標頭,token 放網址
                    self._stream(int(self.headers.get("Last-Event-ID") or 0))
            else:
                self._json(404, {"error": "沒有這個頁面"})

        def _stream(self, since: int) -> None:
            """Server-Sent Events:有新事件就送,沒有就每 15 秒送一次 ping 保持連線。"""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    batch = hub.events_since(since, timeout=15)
                    if not batch:
                        self.wfile.write(b": ping\n\n")
                    for ev in batch:
                        self.wfile.write(f"id: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                        since = ev["seq"]
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return  # 瀏覽器關掉或重新整理

        def do_POST(self) -> None:
            # 先把內容讀完再回應:拒絕時沒讀完就關連線,Windows 會直接中斷(WinError 10053),對方收不到錯誤碼
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(min(length, MAX_BODY))
            if not self._host_ok() or not self._token_ok(self.headers.get("X-Token")):
                return
            if length > MAX_BODY:
                return self._json(413, {"error": "內容太大"})
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(415, {"error": "要用 JSON"})
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "JSON 格式錯誤"})
            path = urlsplit(self.path).path
            if path == "/api/ask":
                text = str(body.get("text", "")).strip()
                if not text:
                    return self._json(400, {"error": "問題是空的"})
                ok = hub.ask(text)
                self._json(202 if ok else 409, {} if ok else {"error": "上一個問題還在回答中"})
            elif path == "/api/confirm":
                ok = hub.answer(str(body.get("id")), str(body.get("answer")))
                self._json(200 if ok else 404, {} if ok else {"error": "這個確認已經結束了"})
            elif path == "/api/apply":
                self._json(200, {"text": hub.apply(int(body.get("patch", -1)), bool(body.get("apply")))})
            elif path == "/api/clear":
                ok = hub.clear()
                self._json(200 if ok else 409, {} if ok else {"error": "還在回答中,等結束再開新對話"})
            elif path == "/api/remember":
                rule = " ".join(str(body.get("rule", "")).split())
                if not rule:
                    return self._json(400, {"error": "規則是空的"})
                hub.remember(rule)
                self._json(200, {})
            else:
                self._json(404, {"error": "沒有這個 API"})

    return Handler


def serve(hub: Hub, port: int = 8765, token: str | None = None) -> ThreadingHTTPServer:
    """建立伺服器(還沒開始跑);port=0 會自動挑一個空的(測試用)。"""
    token = token or secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(hub, token, port))
    if port == 0:  # 實際的 port 要等綁定後才知道,Host 檢查要用它
        server.RequestHandlerClass = make_handler(hub, token, server.server_address[1])
    server.daemon_threads = True
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description="小助理的網頁介面(只開在本機)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="不要自動打開瀏覽器")
    ap.add_argument("--role", help="強制指定模型角色:code(日常)/ code_deep(難題)/ long_context")
    ap.add_argument("--think", choices=["on", "off"], help="思考模式;不給就用伺服器預設")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    registry, quota, tools, allow_web = agent.prepare()
    thinking = {"on": True, "off": False}.get(args.think)
    hub = Hub(lambda on_event: agent.AgentSession(registry, quota, role=args.role, thinking=thinking, tools=tools,
                                                  on_event=on_event),
              status=agent.web_status(allow_web),
              model=registry.spec(args.role or registry.routing["default_role"]).model.split("/")[-1])
    web.CONFIRM = hub.confirm  # 上網前在網頁上跳確認
    server = serve(hub, args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"小助理網頁介面:{url}  (工作區 {toolbox.WORKSPACE};按 Ctrl+C 結束)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已結束")


if __name__ == "__main__":
    main()
