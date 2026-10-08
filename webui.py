"""網頁介面:在瀏覽器裡跟小助理聊天(像 Claude 的網頁版)。

    python webui.py            → 開 http://127.0.0.1:8765 並自動打開瀏覽器(只有這台電腦連得到)
    python webui.py --lan      → 同一個 Wi-Fi 的手機也能開:終端機會印出網址和 6 位數 PIN

功能跟終端機的互動模式一樣:回答邊產生邊顯示、工具進度、上網前確認、套用 patch 前確認、新對話、記住規則。
單人使用。安全:API 要帶啟動時隨機產生的 token(其他網站讀不到頁面,拿不到 token),並檢查 Host 標頭
(擋 DNS rebinding);--lan 時,本機以外的裝置要先輸入 PIN 才拿得到頁面。只用標準函式庫。
"""
from __future__ import annotations

import argparse
import json
import logging
import secrets
import socket
import threading
import time
import webbrowser
from http.cookies import SimpleCookie
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
LOCAL_ADDRS = {"127.0.0.1", "::1"}  # 從這些位址連進來的是這台電腦自己,不用 PIN
COOKIE = "da_session"


def lan_ips() -> list[str]:
    """這台電腦在區域網路上的 IPv4(手機要連的位址)。實際上網用的那張網卡排第一個,
    其他(例如 WSL / Hyper-V 的虛擬網卡 172.x)放後面,手機通常連不到。"""
    primary, others = [], set()
    try:  # 連到一個不會真的送封包的位址,看系統選哪張網卡
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            primary.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        others.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    usable = lambda ip: not ip.startswith(("127.", "169.254.", "0."))  # noqa: E731
    return [ip for ip in primary if usable(ip)] + sorted(ip for ip in others - set(primary) if usable(ip))


class LanAccess:
    """--lan 時給手機用的登入:啟動時產生 6 位數 PIN,輸入正確就發一個 cookie;一分鐘內錯 5 次就先鎖住。"""

    MAX_FAILS, WINDOW = 5, 60

    def __init__(self, ips: list[str], pin: str | None = None):
        self.ips = ips
        self.pin = pin or f"{secrets.randbelow(10**6):06d}"
        self.sessions: set[str] = set()
        self.fails: list[float] = []
        self.lock = threading.Lock()

    def login(self, pin: str) -> str | None:
        """PIN 對了回傳 session id;錯了回傳 None;錯太多次丟 PermissionError。"""
        with self.lock:
            now = time.monotonic()
            self.fails = [t for t in self.fails if now - t < self.WINDOW]
            if len(self.fails) >= self.MAX_FAILS:
                raise PermissionError("PIN 錯太多次,請一分鐘後再試")
            if not secrets.compare_digest(str(pin).strip(), self.pin):
                self.fails.append(now)
                return None
            sid = secrets.token_urlsafe(24)
            self.sessions.add(sid)
            return sid

    def valid(self, sid: str | None) -> bool:
        with self.lock:
            return bool(sid) and sid in self.sessions


LOGIN_PAGE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>小助理 · 登入</title>
<style>
:root{color-scheme:light dark;--bg:#fff;--text:#0d0d0d;--muted:#5d5d5d;--line:#e5e5e5;--btn:#0d0d0d;--btn-ink:#fff;--error:#d93025}
@media (prefers-color-scheme:dark){:root{--bg:#212121;--text:#ececec;--muted:#b4b4b4;--line:#3a3a3a;--btn:#fff;--btn-ink:#0d0d0d;--error:#f87171}}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--text);
font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI","Microsoft JhengHei","PingFang TC",sans-serif}
form{width:min(340px,calc(100vw - 32px));text-align:center}
h1{font-size:24px;margin:0 0 6px}p{color:var(--muted);margin:0 0 20px;font-size:14px}
input{width:100%;box-sizing:border-box;font:600 28px ui-monospace,monospace;letter-spacing:.4em;text-align:center;
padding:10px;border:1px solid var(--line);border-radius:14px;background:transparent;color:var(--text)}
button{margin-top:14px;width:100%;padding:12px;border:0;border-radius:999px;background:var(--btn);color:var(--btn-ink);font:inherit;font-weight:600}
.err{color:var(--error);min-height:1.6em;margin-top:10px;font-size:14px}
</style></head><body>
<form id="f"><h1>小助理</h1><p>輸入電腦上顯示的 6 位數 PIN</p>
<input id="pin" inputmode="numeric" autocomplete="one-time-code" maxlength="6" pattern="[0-9]{6}" required autofocus aria-label="PIN">
<button type="submit">登入</button><div class="err" id="err" role="alert"></div></form>
<script>
document.getElementById("f").onsubmit = async (e) => {
  e.preventDefault();
  const r = await fetch("/api/login", {method: "POST", headers: {"Content-Type": "application/json"},
                                       body: JSON.stringify({pin: document.getElementById("pin").value})});
  if (r.ok) location.reload();
  else document.getElementById("err").textContent = (await r.json().catch(() => ({}))).error || "登入失敗";
};
</script></body></html>"""


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

    def ask(self, text: str, mode: str = "normal", shown: str | None = None) -> bool:
        """shown:畫面上顯示的問題(照計畫執行時,送給模型的是 EXECUTE_TASK,畫面只顯示「照計畫執行」)。"""
        with self.cond:
            if self.busy:
                return False
            self.busy = True
        self.publish("user", {"text": shown or text, "mode": mode})
        threading.Thread(target=self._run, args=(text, mode), daemon=True).start()
        return True

    def execute_plan(self, go: bool) -> bool:
        self.publish("plan_answered", {"go": go})
        return self.ask(agent.EXECUTE_TASK, "execute", shown="照計畫執行") if go else True

    def _run(self, text: str, mode: str = "normal") -> None:
        before, planned = len(toolbox.PATCHES), False
        try:
            r = self.session.ask(self.note + text, agent.file_mentions(text), mode)
            self.note = ""
            self.publish("done", {"role": r.role, "steps": r.steps, "tokens": r.tokens_used,
                                  "total": self.session.tokens_used, "answer": r.answer})
            planned = mode == "plan"
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
        if planned:  # 等不忙了才問,按下「照計畫執行」才不會被當成上一題還沒結束
            self.publish("offer_plan")

    def clear(self) -> bool:
        with self.cond:
            if self.busy:
                return False
            self.session, self.note = self.new_session(self.publish), ""
        if toolbox.DRAFT:  # 新對話不記得草稿,留著會混淆
            toolbox.discard_draft()
        self.publish("clear")
        with self.cond:
            self.view_start = len(self.events)
        return True

    def compact(self) -> str:
        with self.cond:
            if self.busy:
                return "小助理還在回答,等這一輪結束再整理"
            self.busy = True  # 整理摘要要呼叫模型,期間不接新問題
        try:
            return "" if self.session.compact() else "還沒有可以整理的對話"
        except Exception as e:
            return f"整理失敗:{type(e).__name__}: {e}"
        finally:
            with self.cond:
                self.busy = False

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

    def apply(self, index: int, action: str) -> str:
        """action:apply 套用草稿 / keep 先不要(草稿留著繼續改)/ discard 捨棄草稿。"""
        with self.cond:
            if self.busy:
                return "小助理還在回答,等這一輪結束再處理"
        if not 0 <= index < len(toolbox.PATCHES):
            return "找不到這個 patch"
        patch, _ = toolbox.PATCHES[index]
        name = patch.relative_to(toolbox.WORKSPACE).as_posix()
        if action == "apply":
            result = toolbox.apply_patch(patch)
            if result.startswith("已套用"):
                self.note += f"<note>使用者已把 {name} 套用到工作區,相關檔案內容已經改變,需要時請重讀。</note>\n"
        elif action == "discard":
            result = toolbox.discard_draft()
            self.note += "<note>使用者已捨棄你草稿裡的所有修改,檔案回到原本的內容。</note>\n"
        else:
            result = "沒有套用。草稿留著,可以繼續請小助理改,之後再套用。"
        self.publish("applied", {"patch": index, "text": result})
        return result


def make_handler(hub: Hub, token: str, port: int, lan: LanAccess | None = None) -> type[BaseHTTPRequestHandler]:
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"} | {f"{ip}:{port}" for ip in (lan.ips if lan else [])}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # 安靜一點,不要每個請求都印
            pass

        def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict, headers: dict | None = None) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", headers)

        def _host_ok(self) -> bool:
            if self.headers.get("Host") in allowed_hosts:
                return True
            self._json(403, {"error": "只接受本機連線"})
            return False

        def _local(self) -> bool:
            return self.client_address[0] in LOCAL_ADDRS

        def _signed_in(self) -> bool:
            """這台電腦自己連的直接過;手機等其他裝置要先用 PIN 登入(cookie)。"""
            if self._local():
                return True
            morsel = SimpleCookie(self.headers.get("Cookie", "")).get(COOKIE)
            return lan is not None and lan.valid(morsel.value if morsel else None)

        def _token_ok(self, given: str | None) -> bool:
            if not self._signed_in():
                self._json(403, {"error": "請先登入"})
                return False
            if given and secrets.compare_digest(given, token):
                return True
            self._json(403, {"error": "token 不對"})
            return False

        def do_GET(self) -> None:
            if not self._host_ok():
                return
            url = urlsplit(self.path)
            if url.path == "/":
                if not self._signed_in():  # 不給頁面,就拿不到 token
                    return self._send(200, LOGIN_PAGE.encode(), "text/html; charset=utf-8")
                page = PAGE.read_text(encoding="utf-8").replace("__TOKEN__", token)
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                if self._token_ok(self.headers.get("X-Token")):
                    state = {"workspace": str(toolbox.WORKSPACE), "status": hub.status, "busy": hub.busy,
                             "model": hub.model, "draft": sorted(toolbox.DRAFT),
                             "instructions": (toolbox.WORKSPACE / agent.INSTRUCTIONS_FILE).is_file()}
                    if lan and self._local():  # 手機連線方式只給這台電腦看
                        state["lan"] = {"urls": [f"http://{ip}:{port}" for ip in lan.ips], "pin": lan.pin}
                    self._json(200, state)
            elif url.path == "/api/events":
                qs = parse_qs(url.query)
                if self._token_ok((qs.get("token") or [None])[0]):  # EventSource 不能帶標頭,token 放網址
                    self._stream(int(self.headers.get("Last-Event-ID") or 0))
            else:
                self._json(404, {"error": "沒有這個頁面"})

        def _login(self, raw: bytes) -> None:
            if lan is None:
                return self._json(404, {"error": "沒有開手機連線(--lan)"})
            try:
                pin = str(json.loads(raw or b"{}").get("pin", ""))
                sid = lan.login(pin)
            except PermissionError as e:
                return self._json(429, {"error": str(e)})
            except (json.JSONDecodeError, AttributeError):
                return self._json(400, {"error": "格式錯誤"})
            if not sid:
                return self._json(403, {"error": "PIN 不對"})
            self._json(200, {}, {"Set-Cookie": f"{COOKIE}={sid}; HttpOnly; SameSite=Strict; Path=/"})

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
            if not self._host_ok():
                return
            if urlsplit(self.path).path == "/api/login":  # 登入前還沒有 token
                return self._login(raw)
            if not self._token_ok(self.headers.get("X-Token")):
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
                ok = hub.ask(text, "plan" if body.get("mode") == "plan" else "normal")
                self._json(202 if ok else 409, {} if ok else {"error": "上一個問題還在回答中"})
            elif path == "/api/execute":  # 規劃模式交出計畫後:照計畫執行 / 先不要
                ok = hub.execute_plan(bool(body.get("go")))
                self._json(202 if ok else 409, {} if ok else {"error": "上一個問題還在回答中"})
            elif path == "/api/confirm":
                ok = hub.answer(str(body.get("id")), str(body.get("answer")))
                self._json(200 if ok else 404, {} if ok else {"error": "這個確認已經結束了"})
            elif path == "/api/apply":
                self._json(200, {"text": hub.apply(int(body.get("patch", -1)), str(body.get("action", "keep")))})
            elif path == "/api/clear":
                ok = hub.clear()
                self._json(200 if ok else 409, {} if ok else {"error": "還在回答中,等結束再開新對話"})
            elif path == "/api/compact":
                msg = hub.compact()  # 成功時摘要通知會從事件串流送出
                self._json(200, {"text": msg})
            elif path == "/api/remember":
                rule = " ".join(str(body.get("rule", "")).split())
                if not rule:
                    return self._json(400, {"error": "規則是空的"})
                hub.remember(rule)
                self._json(200, {})
            else:
                self._json(404, {"error": "沒有這個 API"})

    return Handler


def serve(hub: Hub, port: int = 8765, token: str | None = None, lan: LanAccess | None = None,
          host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """建立伺服器(還沒開始跑);port=0 會自動挑一個空的(測試用)。"""
    token = token or secrets.token_urlsafe(24)
    server = ThreadingHTTPServer((host, port), make_handler(hub, token, port, lan))
    if port == 0:  # 實際的 port 要等綁定後才知道,Host 檢查要用它
        server.RequestHandlerClass = make_handler(hub, token, server.server_address[1], lan)
    server.daemon_threads = True
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description="小助理的網頁介面")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="不要自動打開瀏覽器")
    ap.add_argument("--lan", action="store_true", help="讓同一個 Wi-Fi 的手機也能開(要輸入啟動時顯示的 PIN)")
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
    lan = LanAccess(lan_ips()) if args.lan else None
    if lan and not lan.ips:
        raise SystemExit("找不到這台電腦的區域網路位址,請確認有連上 Wi-Fi 或網路線")
    server = serve(hub, args.port, lan=lan, host="0.0.0.0" if lan else "127.0.0.1")
    url = f"http://127.0.0.1:{args.port}/"
    print(f"小助理網頁介面:{url}  (工作區 {toolbox.WORKSPACE};按 Ctrl+C 結束)")
    if lan:
        print("\n手機連線(手機要連同一個 Wi-Fi):")
        print(f"  http://{lan.ips[0]}:{args.port}")
        for ip in lan.ips[1:]:
            print(f"  (連不上再試 http://{ip}:{args.port})")
        print(f"  PIN:{lan.pin}")
        print("第一次啟動時 Windows 防火牆可能會詢問,請只勾「私人網路」並允許;公司或公共 Wi-Fi 不建議開。\n")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已結束")


if __name__ == "__main__":
    main()
