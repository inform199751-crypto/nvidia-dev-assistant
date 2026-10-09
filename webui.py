"""網頁介面:在瀏覽器裡跟 HSUAN 聊天(像 Claude 的網頁版)。

    python webui.py            → 開 http://127.0.0.1:8765 並自動打開瀏覽器(只有這台電腦連得到)
    python webui.py --lan      → 同一個 Wi-Fi 的手機也能開:終端機會印出網址和 6 位數 PIN

功能跟終端機的互動模式一樣:回答邊產生邊顯示、工具進度、上網前確認、套用 patch 前確認、新對話、記住規則。
單人使用。安全:API 要帶啟動時隨機產生的 token(其他網站讀不到頁面,拿不到 token),並檢查 Host 標頭
(擋 DNS rebinding);--lan 時,本機以外的裝置要先輸入 PIN 才拿得到頁面。只用標準函式庫。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
import re
import webbrowser
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import cache
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, urlsplit

from langchain_core.messages import messages_from_dict, messages_to_dict

import agent
import guard
import knowledge
import qr
import tools as toolbox
import web

PAGE = Path(__file__).with_name("webui.html")
CONFIRM_TIMEOUT = 300  # 秒;沒人回答就當作不允許
MAX_DIFF_CHARS = 20000
MAX_BODY = 16_000_000  # POST 內容上限(問題文字、規則、上傳的圖片)
MAX_IMAGES = 4  # 一題最多幾張圖
MAX_IMAGE_CHARS = 7_000_000  # 一張圖的 data URL 上限(約 5 MB);瀏覽器會先把大圖縮到 1600px
MAX_THUMB_CHARS = 80_000  # 對話裡顯示的縮圖(會存進對話紀錄)
IMAGE_URL = re.compile(r"data:image/(png|jpeg|webp|gif);base64,[A-Za-z0-9+/=]+")
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
    """手機用的登入:啟動時產生 6 位數 PIN,輸入正確就發一個 cookie;一分鐘內錯 5 次就先鎖住。

    store:記住配對過的手機(只存 cookie 的雜湊),電腦重開、PIN 換了也不用再輸入;None = 只記在記憶體(--lan)。
    open:手機連線關掉時設成 False,已經連著的手機也一起斷開。"""

    MAX_FAILS, WINDOW = 5, 60
    REMEMBER = 365 * 24 * 3600  # 配對的 cookie 留一年

    def __init__(self, ips: list[str], pin: str | None = None, store: Path | None = None, guests: bool = False):
        """guests:另外產生一組訪客 PIN —— 用它配對的手機是訪客(自己的空間,看不到主人的對話和檔案)。
        身分由「輸入哪一組 PIN」決定,手機自己沒辦法升級成主人。"""
        self.ips = ips
        self.pin = pin or f"{secrets.randbelow(10**6):06d}"
        self.guest_pin = None
        while guests and self.guest_pin in (None, self.pin):
            self.guest_pin = f"{secrets.randbelow(10**6):06d}"
        self.sessions: dict[str, str] = {}  # session id → owner / guest
        self.fails: list[float] = []
        self.lock = threading.Lock()
        self.store = store
        self.open = True
        # cookie 的雜湊 → {kind: owner / guest, device: 「iPhone · 主畫面 App」, paired, seen}(存檔的只有雜湊,沒有 cookie)
        self.devices: dict[str, dict] = {}
        if store:
            try:
                saved = json.loads(store.read_text(encoding="utf-8"))
                # 舊格式:雜湊的清單(那時只有主人的手機)、雜湊 → owner / guest(還沒記裝置和時間)
                self.devices = {h: {"kind": "owner"} for h in saved} if isinstance(saved, list) else \
                    {h: r if isinstance(r, dict) else {"kind": r} for h, r in saved.items()}
            except (OSError, ValueError, TypeError, AttributeError):
                pass

    SEEN_EVERY = 300  # 「最後使用」幾秒更新一次存檔(每個請求都會檢查身分,不能每次都寫檔)

    @staticmethod
    def _digest(sid: str) -> str:
        return hashlib.sha256(sid.encode()).hexdigest()

    def _save(self) -> None:
        if self.store:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            self.store.write_text(json.dumps(self.devices, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    def count(self, kind: str) -> int:
        with self.lock:
            return sum(r.get("kind") == kind for r in self.devices.values())

    def device_list(self, kind: str) -> list[dict]:
        """設定頁列出配對過的裝置(最近用過的在前)。舊的配對沒有記裝置和時間,下次連線時補上。"""
        with self.lock:
            rows = [{"id": h[:12], "device": r.get("device", ""), "paired": r.get("paired"), "seen": r.get("seen")}
                    for h, r in self.devices.items() if r.get("kind") == kind]
        return sorted(rows, key=lambda d: -(d["seen"] or d["paired"] or 0))

    def forget(self, kind: str | None = None) -> None:
        """「忘記所有手機」(kind=None)或「忘記所有訪客」(kind="guest"):要重新輸入 PIN。"""
        with self.lock:
            self.sessions = {s: k for s, k in self.sessions.items() if kind and k != kind}
            self.devices = {h: r for h, r in self.devices.items() if kind and r.get("kind") != kind}
            self._save()

    def login(self, pin: str, device: str = "") -> str | None:
        """PIN 對了回傳 session id;錯了回傳 None;錯太多次丟 PermissionError。device:設定頁顯示的裝置類型。"""
        with self.lock:
            now = time.monotonic()
            self.fails = [t for t in self.fails if now - t < self.WINDOW]
            if len(self.fails) >= self.MAX_FAILS:
                raise PermissionError("PIN 錯太多次,請一分鐘後再試")
            pin = str(pin).strip()
            if secrets.compare_digest(pin, self.pin):
                kind = "owner"
            elif self.guest_pin and secrets.compare_digest(pin, self.guest_pin):
                kind = "guest"
            else:
                self.fails.append(now)
                return None
            sid = secrets.token_urlsafe(24)
            self.sessions[sid] = kind
            if self.store:
                self.devices[self._digest(sid)] = {"kind": kind, "device": device, "paired": time.time(), "seen": time.time()}
                self._save()
            return sid

    def guest_ids(self) -> list[str]:
        """配對中的訪客手機代號(identify() 給的 guest:<代號>)。"""
        with self.lock:
            ids = {h[:12] for h, r in self.devices.items() if r.get("kind") == "guest"}
            ids |= {self._digest(s)[:12] for s, k in self.sessions.items() if k == "guest"}
        return sorted(ids)

    def forget_device(self, did: str, kind: str | None = None) -> bool:
        """只取消一個裝置的配對(設定頁每一筆的「取消」);其他手機不受影響。did 是 device_list 的 id(雜湊前 12 碼)。"""
        if not re.fullmatch(r"[0-9a-f]{12}", did or ""):  # 空字串會對到每一支
            return False
        with self.lock:
            sessions = {s: k for s, k in self.sessions.items()
                        if not (self._digest(s).startswith(did) and kind in (None, k))}
            devices = {h: r for h, r in self.devices.items() if not (h.startswith(did) and kind in (None, r.get("kind")))}
            changed = (len(sessions), len(devices)) != (len(self.sessions), len(self.devices))
            self.sessions, self.devices = sessions, devices
            if changed:
                self._save()
        return changed

    def forget_guest(self, gid: str) -> bool:
        return self.forget_device(gid, "guest")

    def logout(self, sid: str | None) -> None:
        """帳號選單的「登出」:只忘記這一支手機,下次要重新輸入 PIN;其他手機不受影響。"""
        if not sid:
            return
        with self.lock:
            self.sessions.pop(sid, None)
            if self.devices.pop(self._digest(sid), None):
                self._save()

    def identify(self, sid: str | None, device: str = "") -> str | None:
        """這個 cookie 是誰:"owner"、"guest:<這支手機的代號>",或 None(沒登入、被忘記、手機連線關了)。
        順便記「最後使用」(每 SEEN_EVERY 秒寫一次檔);舊的配對沒記裝置類型的,用這次的 device 補上。"""
        if not sid:
            return None
        digest = self._digest(sid)
        with self.lock:
            rec = self.devices.get(digest)
            kind = self.sessions.get(sid) or (rec or {}).get("kind")
            if not (self.open and kind):
                return None
            if rec is not None:
                now, dirty = time.time(), False
                if device and not rec.get("device"):
                    rec["device"], dirty = device, True
                if now - (rec.get("seen") or 0) > self.SEEN_EVERY:
                    rec["seen"], dirty = now, True
                if dirty:
                    self._save()
        return "owner" if kind == "owner" else f"guest:{digest[:12]}"

    def valid(self, sid: str | None, device: str = "") -> bool:
        return self.identify(sid, device) is not None


def device_label(ua: str, app: bool = False) -> str:
    """從 User-Agent 猜是什麼裝置,設定頁列配對過的手機用(「iPhone · 主畫面 App」)。
    同一支手機用瀏覽器和主畫面圖示打開,登入資料是分開存的,會算兩筆 —— 列出來才看得出是哪一筆(10/10 使用者問的)。
    app:登入頁回報的「從主畫面打開」;iPhone 主畫面打開的 User-Agent 沒有 Safari/ 字樣,不靠它也認得出來。"""
    ua = ua or ""
    system = next((name for key, name in (("iPad", "iPad"), ("iPhone", "iPhone"), ("Android", "Android"),
                                          ("Windows", "Windows"), ("Macintosh", "Mac"), ("Linux", "Linux")) if key in ua),
                  "其他裝置")
    if "Line/" in ua:  # LINE 裡點開的(iPhone 的也沒有 Safari/,要先判斷)
        browser = "LINE 內建瀏覽器"
    elif app or (system in ("iPhone", "iPad") and "Safari/" not in ua):
        browser = "主畫面 App"
    else:
        browser = next((name for key, name in (("SamsungBrowser", "三星瀏覽器"), ("Edg", "Edge"), ("CriOS", "Chrome"),
                                               ("Chrome/", "Chrome"), ("FxiOS", "Firefox"), ("Firefox/", "Firefox"),
                                               ("Safari/", "Safari")) if key in ua), "瀏覽器")
    return f"{system} · {browser}"


class PhoneLink:
    """桌面版「更多」裡的「手機連線」開關:打開時另外開一個聽 0.0.0.0 的伺服器給手機 —— 同一張 Desk、同一個 token,
    電腦和手機看到的是同一份對話(不要另外開 webui.py --lan,兩個程式共用存檔會互相蓋掉)。
    手機第一次輸入 PIN 配對,之後記住;開關狀態也記著,下次打開小助理自動恢復。只綁需要的時候,
    沒打開就不對區域網路開 port(Windows 防火牆也是打開時才會問)。"""

    def __init__(self, desk: Desk, token: str, port: int, store: Path | None = None, host: str = "0.0.0.0",
                 guests: Callable[[str], Desk] | None = None):
        """guests:給訪客手機的桌面(guest_desks());給了才有訪客 PIN。"""
        self.desk, self.token, self.port, self.store, self.host = desk, token, port, store, host
        self.guests = guests
        self.lan = LanAccess([], store=store / "phones.json" if store else None, guests=guests is not None)
        self.server: ThreadingHTTPServer | None = None
        self.lock = threading.Lock()

    @property
    def guest_listeners(self) -> int:
        """訪客手機連著的頁面數(桌面版判斷「還有沒有人在用」時也算進去)。"""
        return sum(d.listeners for d in getattr(self.guests, "desks", {}).values())

    def _remember(self, on: bool) -> None:
        if self.store:
            self.store.mkdir(parents=True, exist_ok=True)
            (self.store / "phone.json").write_text(json.dumps({"on": on}), encoding="utf-8")

    def resume(self) -> None:
        """上次關掉 HSUAN 時是開著的,就再打開。"""
        try:
            on = json.loads((self.store / "phone.json").read_text(encoding="utf-8")).get("on") if self.store else False
        except (OSError, ValueError, AttributeError):
            on = False
        if on and (err := self.start()):
            logging.warning("手機連線沒有打開:%s", err)

    def start(self) -> str:
        """打開;回傳錯誤訊息,空字串 = 成功。"""
        with self.lock:
            if self.server:
                return ""
            ips = lan_ips()
            if not ips:
                return "找不到這台電腦的區域網路位址,請確認有連上 Wi-Fi"
            self.lan.ips = ips
            try:
                server = serve(self.desk, self.port, token=self.token, lan=self.lan, host=self.host, guests=self.guests)
            except OSError:
                return f"port {self.port} 已經被別的程式占用(例如另外開著的 webui.py --lan),請先關掉它"
            self.port = server.server_address[1]  # port=0(測試)時看實際挑到哪個
            self.lan.open = True
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.server = server
        self._remember(True)
        return ""

    def close(self) -> None:
        """關掉伺服器(HSUAN 結束時用:不改記住的開關狀態)。"""
        with self.lock:
            server, self.server = self.server, None
            self.lan.open = False  # 連著的手機在下一次檢查時斷開
        if server:
            server.shutdown()
            server.server_close()

    def stop(self) -> None:
        self.close()
        self._remember(False)

    def status(self) -> dict:
        urls = [f"http://{ip}:{self.port}" for ip in self.lan.ips] if self.server else []
        on = bool(self.server)
        return {"on": on, "urls": urls, "pin": self.lan.pin if on else "",
                "guest_pin": (self.lan.guest_pin or "") if on else "",
                "phones": self.lan.count("owner"), "guests": self.lan.count("guest"),
                "qr": qr.svg(urls[0]) if urls else "", "guest_usage": self.guest_usage(),
                "devices": self.lan.device_list("owner")}  # 你的手機:一筆一筆列出來,各自可以取消

    def guest_usage(self) -> dict | None:
        """每支訪客手機今天用了多少、幾段對話、最後什麼時候用。訪客的額度跟主人分開,但用的是主人的 NVIDIA 金鑰。
        配對中的都列;取消配對了、但今天用過的也列(用量還是算在這把金鑰上)。"""
        quota, base = getattr(self.guests, "quota", None), getattr(self.guests, "base", None)
        if not quota:
            return None
        used = {u.removeprefix("guest-"): n for u, n in quota.everyone().items() if u.startswith("guest-")}
        paired = set(self.lan.guest_ids())
        devices = {d["id"]: d["device"] for d in self.lan.device_list("guest")}
        rows = []
        for gid in paired | set(used):
            convs = list((base / gid / "state" / "conversations").glob("*.json")) if base else []
            rows.append({"id": gid, "used": used.get(gid, 0), "paired": gid in paired, "conversations": len(convs),
                         "last": max((f.stat().st_mtime for f in convs), default=None), "device": devices.get(gid, "")})
        rows.sort(key=lambda r: (-r["used"], -(r["last"] or 0), r["id"]))
        return {"rows": rows, "limit": quota.daily_tokens, "total": sum(used.values())}


ASSETS = Path(__file__).with_name("assets")
# 不用登入就給的檔案:瀏覽器抓圖示和 manifest 時不帶 cookie(手機還沒輸入 PIN 也要看得到 app 圖示)
PUBLIC_FILES = {"/favicon.ico": ("assistant.ico", "image/x-icon"),
                "/icon-192.png": ("icon-192.png", "image/png"),
                "/icon-512.png": ("icon-512.png", "image/png"),
                "/icon-maskable-512.png": ("icon-maskable-512.png", "image/png"),
                "/apple-touch-icon.png": ("apple-touch-icon.png", "image/png")}
# 手機「加到主畫面」:有這個 manifest,主畫面的圖示點開就是全螢幕的小助理(沒有網址列)
MANIFEST = {"name": "HSUAN", "short_name": "HSUAN", "id": "/", "start_url": "/", "scope": "/",
            "display": "standalone", "background_color": "#121212", "theme_color": "#0a0a0a",
            "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                      {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
                      {"src": "/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}]}


LOGIN_PAGE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#121212">
<link rel="manifest" href="/manifest.webmanifest"><link rel="apple-touch-icon" href="/apple-touch-icon.png">
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-title" content="HSUAN">
<title>HSUAN · 登入</title>
<style>
:root{color-scheme:dark;--bg:#121212;--text:#ececec;--muted:#b4b4b4;--line:#2a2a2a;--btn:#fff;--btn-ink:#0d0d0d;--error:#f87171}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--text);
font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI","Microsoft JhengHei","PingFang TC",sans-serif}
form{width:min(340px,calc(100vw - 32px));text-align:center}
h1{font-size:24px;margin:0 0 6px}p{color:var(--muted);margin:0 0 20px;font-size:14px}
input{width:100%;box-sizing:border-box;font:600 28px ui-monospace,monospace;letter-spacing:.4em;text-align:center;
padding:10px;border:1px solid var(--line);border-radius:14px;background:transparent;color:var(--text)}
button{margin-top:14px;width:100%;padding:12px;border:0;border-radius:999px;background:var(--btn);color:var(--btn-ink);font:inherit;font-weight:600}
.err{color:var(--error);min-height:1.6em;margin-top:10px;font-size:14px}
.note{margin-top:18px;font-size:12px;line-height:1.6}
</style></head><body>
<form id="f"><h1>HSUAN</h1><p>輸入電腦上顯示的 6 位數 PIN</p>
<input id="pin" inputmode="numeric" autocomplete="one-time-code" maxlength="6" pattern="[0-9]{6}" required autofocus aria-label="PIN">
<button type="submit">登入</button><div class="err" id="err" role="alert"></div>
<p class="note">用訪客 PIN 登入的話,你的對話只有你看得到,但會存在這台電腦上。</p></form>
<script>
document.getElementById("f").onsubmit = async (e) => {
  e.preventDefault();
  const r = await fetch("/api/login", {method: "POST", headers: {"Content-Type": "application/json"},
                                       body: JSON.stringify({pin: document.getElementById("pin").value,
                                                             app: matchMedia("(display-mode: standalone)").matches || navigator.standalone === true})});
  if (r.ok) location.reload();
  else document.getElementById("err").textContent = (await r.json().catch(() => ({}))).error || "登入失敗";
};
</script></body></html>"""


class Hub:
    """一段對話的狀態 + 事件紀錄。agent 在背景執行緒跑,事件記下來給瀏覽器(重新整理時從頭重播這段對話)。"""

    def __init__(self, new_session: Callable[[Callable], agent.AgentSession], status: str = "", model: str = "",
                 team: bool = False, gate: Gate | None = None, patch_dir: Path | None = None,
                 output_dir: Path | None = None, workspace: Path | None = None, screen: bool = False):
        """screen:問題先過安全模型(guard.py),不安全就不回答 —— 訪客用的是主人的金鑰,主人自己的問題不檢查。"""
        self.new_session = new_session
        self.screen = screen
        self.status = status
        self.model = model  # 頁面上方顯示的模型名稱
        # team:多人共用伺服器。工作區是共用的(唯讀),所以不能套用修改、不能改 ASSISTANT.md,改成下載 patch
        self.team = team
        self.gate = gate  # 全部使用者共用的同時執行上限(保護 GPU);None = 不限
        self.bench = toolbox.Workbench(patch_dir=patch_dir, output_dir=output_dir,  # 這段對話自己的草稿 / patch / 文件
                                       confirm=self.confirm, workspace=workspace)
        self.plan_waiting = False  # 交出計畫、還沒按「照計畫執行 / 先不要」
        self.escalation: dict | None = None  # 做到步數上限、還沒按「換模型接著做 / 先不要」(offer_escalate 的內容)
        self.cond = threading.Condition()
        self.events: list[dict] = []
        self.view_start = 0  # 最近一次「新對話」之後的第一個事件
        self.busy = False
        self.note = ""  # 下一個問題要先告訴模型的事(例如已經套用 patch)
        self.pending: dict[str, list] = {}  # 等使用者回答的確認:id → [threading.Event, 答案]
        self.listeners = 0  # 現在有幾個頁面連著事件串流(桌面版用來判斷視窗是不是都關了)
        self.always: set[str] = set()
        self.on_event: Callable[[str], None] | None = None  # Desk 用來在重要事件後存檔
        self.chat = False  # 「對話」模式(側邊欄切到「對話」開的):不綁專案,見 use_chat()
        self.session = self._new_session()

    def _new_session(self) -> agent.AgentSession:
        with toolbox.using(self.bench):  # 系統提示要讀這個專案的 ASSISTANT.md
            session = self.new_session(self.publish)
            if self.chat:
                session.to_chat()
            self._knowledge_tool(session)
        return session

    def _knowledge_tool(self, session: agent.AgentSession) -> None:
        """有知識庫才給 search_knowledge(工具定義每一步都要送,沒有知識庫就不要佔 token)。"""
        tools = [t for t in session.tools if t.name != knowledge.search_knowledge.name]
        if self.bench.knowledge and not session.whole:
            tools.append(knowledge.search_knowledge)
        session.tools = tools
        session.refresh_system_prompt()

    def use_knowledge(self, kbs: list) -> None:
        """Desk 的知識庫變了(加了、拿掉、更新完):工具和系統提示裡的知識庫清單跟著換,下一輪生效。"""
        self.bench.knowledge = list(kbs)
        with toolbox.using(self.bench):
            self._knowledge_tool(self.session)

    @property
    def workspace(self) -> Path:
        return self.bench.workspace or toolbox.WORKSPACE

    @property
    def rules_file(self) -> Path:
        return self.bench.rules or self.workspace / agent.INSTRUCTIONS_FILE

    def use_rules(self, path: Path) -> None:
        """改用 path 當專案規則檔(從 GitHub 下載的專案:不信任 repo 自帶的 ASSISTANT.md)。"""
        self.bench.rules = path
        with toolbox.using(self.bench):
            self.session.refresh_system_prompt()

    def use_chat(self) -> None:
        """換成「對話」模式:短的系統提示、只有上網和存文件的工具,不附加 @檔案、不能改專案規則。"""
        self.chat = True
        with toolbox.using(self.bench):
            self.session.to_chat()

    def use_lang(self, lang: str | None) -> None:
        """帳號選單的「回答語言」:改了馬上生效(下一步送出的系統提示就換了)。"""
        self.bench.lang = lang
        with toolbox.using(self.bench):
            self.session.refresh_system_prompt()

    # ---------- 事件 ----------

    def publish(self, kind: str, data: dict | None = None) -> None:
        with self.cond:
            self.events.append({"seq": len(self.events) + 1, "kind": kind, **(data or {})})
            self.cond.notify_all()
        if self.on_event:
            self.on_event(kind)

    # ---------- 輸入框的「選模型」 ----------

    def roles(self) -> list[dict]:
        """選項:自動(依難度和長度選)+ 這個後端有、能用工具的角色。"""
        reg = self.session.registry
        out = [{"role": None, "name": "自動", "desc": "依難度和長度自動選"}]
        for role, desc in ROLE_CHOICES.items():
            spec = reg.specs.get(role)
            if spec and spec.supports_tools and reg.backend in spec.available_on:
                out.append({"role": role, "name": spec.model.split("/")[-1], "desc": desc})
        return out

    def set_role(self, role: str | None) -> str:
        """這段對話之後都用 role(None = 自動);回傳錯誤訊息,空字串 = 成功。"""
        if role not in {r["role"] for r in self.roles()}:
            return "沒有這個模型"
        with self.cond:
            if self.busy:
                return "HSUAN 還在回答,等這一輪結束再換"
            self.session.requested_role = role
        return ""

    def set_effort(self, effort: str) -> str:
        """「選模型」裡的努力程度(agent.EFFORTS),這段對話之後都用它;回傳錯誤訊息,空字串 = 成功。"""
        if effort not in agent.EFFORTS:
            return "沒有這個努力程度"
        with self.cond:
            if self.busy:
                return "HSUAN 還在回答,等這一輪結束再換"
            self.session.effort = effort
        return ""

    # ---------- 存檔 / 讀檔(Desk 用:關掉程式再打開,對話還在) ----------

    def export(self) -> dict:
        """要存到硬碟的內容。事件裡一個字一個字的 text 合併、thinking 丟掉,不然檔案會很大。"""
        with self.cond:
            events = []
            for ev in self.events[self.view_start:]:
                if ev["kind"] == "thinking":
                    continue
                if ev["kind"] == "text" and events and events[-1]["kind"] == "text":
                    events[-1] = {**events[-1], "text": events[-1]["text"] + ev["text"]}
                else:
                    events.append(dict(ev))
        return {"messages": messages_to_dict(self.session.messages), "events": events,
                "tokens_used": self.session.tokens_used, "draft": dict(self.bench.draft),
                "patches": [[str(p), ok] for p, ok in self.bench.patches], "role": self.session.requested_role,
                "effort": self.session.effort}

    def restore(self, data: dict) -> None:
        events = [dict(ev) for ev in data.get("events", [])]
        # 上次關掉時沒收尾的:答到一半的這一輪、等回答的確認(那個執行緒已經不在了),補上結尾,畫面才不會卡在「回答中」
        asked = {ev["id"] for ev in events if ev["kind"] == "confirm"} - {ev["id"] for ev in events if ev["kind"] == "confirm_done"}
        events += [{"kind": "confirm_done", "id": cid, "answer": "n"} for cid in sorted(asked)]
        last = next((ev["kind"] for ev in reversed(events) if ev["kind"] in ("user", "done", "error")), None)
        if last == "user":
            events.append({"kind": "error", "text": "上次關掉程式時這一輪還沒答完,請再問一次"})
        turn = next((ev["kind"] for ev in reversed(events) if ev["kind"] in ("user", "offer_plan", "plan_answered")), None)
        self.plan_waiting = turn == "offer_plan"
        offer = next((ev for ev in reversed(events) if ev["kind"] in ("user", "offer_escalate", "escalate_answered")), {})
        self.escalation = {k: v for k, v in offer.items() if k not in ("kind", "seq")} \
            if offer.get("kind") == "offer_escalate" else None
        with self.cond:
            self.events = [{**ev, "seq": i} for i, ev in enumerate(events, 1)]
        if data.get("messages"):
            self.session.messages = messages_from_dict(data["messages"])
            with toolbox.using(self.bench):
                self.session.refresh_system_prompt()  # 今天日期、專案規則用現在的
        self.session.tokens_used = data.get("tokens_used", 0)
        if "role" in data:  # 輸入框選的模型;模型清單改過、已經沒有這個角色就回到自動
            self.session.requested_role = data["role"] if data["role"] in {r["role"] for r in self.roles()} else None
        if data.get("effort") in agent.EFFORTS:
            self.session.effort = data["effort"]
        self.bench.draft.update(data.get("draft", {}))
        self.bench.patches.extend((Path(p), ok) for p, ok in data.get("patches", []) if Path(p).is_file())

    def events_since(self, seq: int, timeout: float) -> list[dict]:
        with self.cond:
            start = max(seq, self.view_start)
            if len(self.events) <= start:
                self.cond.wait(timeout)
            return self.events[max(seq, self.view_start):]

    # ---------- 對話 ----------

    def ask(self, text: str, mode: str = "normal", shown: str | None = None, role: str | None = None,
            images: list[dict] | None = None) -> bool:
        """shown:畫面上顯示的問題(照計畫執行時,送給模型的是 EXECUTE_TASK,畫面只顯示「照計畫執行」)。
        role:只有這一輪換模型(做到步數上限後的「換模型接著做」)。
        images:上傳的圖片 [{"name","data","thumb"}](parse_images 檢查過);畫面和存檔只留縮圖,原圖只送給看圖模型。"""
        with self.cond:
            if self.busy:
                return False
            self.busy = True
            self.plan_waiting = False  # 問了別的,或已經照計畫執行:之前的計畫不再等
            stale, self.escalation = self.escalation, None
        if stale:  # 沒按「換模型接著做」就問了別的:那張卡片收起來
            self.publish("escalate_answered", {"go": False})
        event = {"text": shown or text, "mode": mode}
        if images:
            event["images"] = [im["thumb"] for im in images]
        self.publish("user", event)
        pictures = [(im["name"], im["data"]) for im in images or []]
        threading.Thread(target=self._run, args=(text, mode, role, pictures), daemon=True).start()
        return True

    def execute_plan(self, go: bool) -> bool:
        self.plan_waiting = False
        self.publish("plan_answered", {"go": go})
        return self.ask(agent.EXECUTE_TASK, "execute", shown="照計畫執行") if go else True

    def escalate(self, go: bool) -> str:
        """做到步數上限後:換模型接著做 / 先不要。回傳錯誤訊息,空字串 = 成功。"""
        with self.cond:
            if self.busy:
                return "上一個問題還在回答中"
            offer, self.escalation = self.escalation, None
        if not offer:
            return "這個提議已經過期了"
        self.publish("escalate_answered", {"go": go})
        if go and not self.ask(agent.CONTINUE_TASK, offer["mode"], shown=f"換 {offer['model']} 接著做", role=offer["role"]):
            return "上一個問題還在回答中"
        return ""

    def _run(self, text: str, mode: str = "normal", role: str | None = None, images: list | None = None) -> None:
        patches = self.bench.patches
        before, planned, offer = len(patches), False, None
        try:
            if self.screen and (v := guard.screen(text)).safe is False:
                raise guard.Blocked(f"這個問題沒有通過安全檢查({v.categories or '不當內容'}),HSUAN 不會回答。")
            with self.gate.turn(self.publish) if self.gate else nullcontext(), toolbox.using(self.bench):
                files = [] if self.chat else agent.file_mentions(text)  # 對話模式讀不到電腦上的檔案
                r = self.session.ask(self.note + text, files, mode, role, images or [])
            self.note = ""
            self.publish("done", {"role": r.role, "steps": r.steps, "tokens": r.tokens_used,
                                  "total": self.session.tokens_used, "answer": r.answer})
            planned = mode == "plan" and not r.hit_limit  # 做到上限 = 沒交出計畫
            if hard := self.session.escalation(r):
                offer = {"role": hard, "model": self.session.registry.spec(hard).model.split("/")[-1],
                         "mode": mode, "used": r.role, "steps": r.steps}
            if len(patches) > before:  # 這一輪提了 patch:問要不要套用(沒通過測試也給看,但會警告)
                patch, passed = patches[-1]
                self.publish("offer_apply", {"patch": len(patches) - 1, "passed": passed, "team": self.team,
                                             "name": toolbox.patch_label(patch),
                                             "diff": patch.read_text(encoding="utf-8")[:MAX_DIFF_CHARS]})
        except guard.Blocked as e:  # 訪客的問題沒通過安全檢查:模型沒有收到,對話照常
            self.publish("error", {"text": str(e)})
        except Exception as e:  # 額度用完、模型全掛:這輪已經退掉,對話還能繼續
            self.publish("error", {"text": f"{type(e).__name__}: {e}"})
        finally:
            with self.cond:
                self.busy = False
                self.cond.notify_all()
        if planned:  # 等不忙了才問,按下「照計畫執行」才不會被當成上一題還沒結束
            self.plan_waiting = True
            self.publish("offer_plan")
        if offer:
            self.escalation = offer
            self.publish("offer_escalate", offer)

    def clear(self) -> bool:
        with self.cond:
            if self.busy:
                return False
            role, effort = self.session.requested_role, self.session.effort  # 選好的模型和努力程度留著
            self.session = self._new_session()
            self.session.requested_role, self.session.effort = role, effort
            self.note, self.plan_waiting, self.escalation = "", False, None
        self.bench.draft.clear()  # 新對話不記得草稿,留著會混淆
        self.publish("clear")
        with self.cond:
            self.view_start = len(self.events)
        return True

    def compact(self) -> str:
        with self.cond:
            if self.busy:
                return "HSUAN 還在回答,等這一輪結束再整理"
            self.busy = True  # 整理摘要要呼叫模型,期間不接新問題
        try:
            return "" if self.session.compact() else "還沒有可以整理的對話"
        except Exception as e:
            return f"整理失敗:{type(e).__name__}: {e}"
        finally:
            with self.cond:
                self.busy = False

    def remember(self, rule: str) -> str:
        if self.team:
            return "多人模式的專案規則由管理者維護(工作區的 ASSISTANT.md),不能從這裡改"
        if self.chat:
            return "對話模式不綁專案,沒有專案規則;請切到「程式」,在那個專案的對話裡加"
        with toolbox.using(self.bench):  # 寫進這段對話所在專案的 ASSISTANT.md
            path = agent.remember(rule)
            self.session.refresh_system_prompt()
        self.publish("notice", {"text": f"已記住,寫進 {path.name}:{rule}"})
        return ""

    # ---------- 需要使用者確認的動作 ----------

    def confirm(self, action: str, detail: str) -> bool:
        """給 web.CONFIRM 用:在網頁上跳出確認卡片,等使用者按按鈕(agent 執行緒會在這裡等)。"""
        if action in self.always:
            self.publish("notice", {"text": f"已允許{action}:{detail}"})
            return True
        cid = secrets.token_hex(4)
        slot = [threading.Event(), "n", action, detail]  # 後兩個給「待命」頁面列出來
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
                return "HSUAN 還在回答,等這一輪結束再處理"
        if index == -1:  # 「待命」頁面:最新的那份(包含草稿全部修改)
            index = len(self.bench.patches) - 1
        if not 0 <= index < len(self.bench.patches):
            return "找不到這個 patch"
        patch, _ = self.bench.patches[index]
        name = toolbox.patch_label(patch)
        if action == "apply" and self.team:
            return "多人模式不能直接改共用的工作區,請下載 patch 自己套用"
        with toolbox.using(self.bench):
            if action == "apply":
                result = toolbox.apply_patch(patch)
                if result.startswith("已套用"):
                    self.note += f"<note>使用者已把 {name} 套用到工作區,相關檔案內容已經改變,需要時請重讀。</note>\n"
            elif action == "discard":
                result = toolbox.discard_draft()
                self.note += "<note>使用者已捨棄你草稿裡的所有修改,檔案回到原本的內容。</note>\n"
            else:
                result = "沒有套用。草稿留著,可以繼續請 HSUAN 改,之後再套用。"
        self.publish("applied", {"patch": index, "text": result})
        return result

    def patch_file(self, index: int) -> Path | None:
        if index == -1:  # 最新的那份
            index = len(self.bench.patches) - 1
        return self.bench.patches[index][0] if 0 <= index < len(self.bench.patches) else None

    # ---------- 側邊欄的「成品」「待命」頁面 ----------

    def _docs_dir(self) -> Path:
        return self.bench.output_dir or toolbox.OUTPUT_DIR

    def outputs(self) -> dict:
        """成品:存成檔案的文件(新的在前)+ 這段對話提過的 patch(新的在前,最新的那份才包含全部修改)。"""
        patches = self.bench.patches
        return {"docs": docs_in(self._docs_dir()),
                "patches": [{"i": i, "name": p.name, "passed": ok, "latest": i == len(patches) - 1}
                            for i, (p, ok) in reversed(list(enumerate(patches)))]}

    def doc_file(self, name: str) -> Path | None:
        return doc_path(self._docs_dir(), name)

    def pending_state(self) -> dict:
        """待命:等使用者處理的事 —— 還沒套用的草稿、等回答的上網確認、等核准的計畫。"""
        with self.cond:
            confirms = [{"id": cid, "action": s[2], "detail": s[3]} for cid, s in self.pending.items()]
        patches = self.bench.patches
        return {"draft": sorted(self.bench.draft), "confirms": confirms, "plan": self.plan_waiting,
                "patch": len(patches) - 1 if self.bench.draft and patches else None,
                "passed": bool(patches and patches[-1][1]), "team": self.team, "busy": self.busy}


class Gate:
    """全部使用者共用的同時執行上限:GPU 一次只能服務這麼多人,其他人排隊(畫面上會顯示排第幾)。"""

    def __init__(self, limit: int):
        self.sem = threading.BoundedSemaphore(limit)
        self.lock = threading.Lock()
        self.waiting = 0

    @contextmanager
    def turn(self, publish: Callable):
        if not self.sem.acquire(blocking=False):
            with self.lock:
                self.waiting += 1
                ahead = self.waiting
            publish("notice", {"text": f"使用的人比較多,排隊中(前面約 {ahead} 位)…"})
            self.sem.acquire()
            with self.lock:
                self.waiting -= 1
        try:
            yield
        finally:
            self.sem.release()


SAVE_ON = {"user", "done", "error", "applied", "offer_apply", "offer_plan", "plan_answered", "offer_escalate",
           "escalate_answered", "confirm_done", "clear"}
# 輸入框「選模型」列出的角色(models.yaml)和說明;備援角色不列(主模型掛了才會自動用到)
ROLE_CHOICES = {"code": "日常,快", "code_deep": "難題,慢但推理強", "long_context": "超長內容(整份文件、很多檔案)"}


def doc_path(d: Path, name: str) -> Path | None:
    """只給輸出資料夾裡、文件類副檔名的檔案(擋 ../、隱藏檔、log 之類)。"""
    p = d / name
    if "/" in name or "\\" in name or name.startswith(".") or p.suffix.lower() not in toolbox.DOC_SUFFIXES:
        return None
    return p if p.is_file() else None


def docs_in(d: Path) -> list[dict]:
    docs = [p for p in d.glob("*") if doc_path(d, p.name)] if d.is_dir() else []
    docs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in docs]


GITHUB_REPO = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9-]+)/([A-Za-z0-9._-]+?)(?:\.git)?(?:[/?#].*)?")
CLONE_DEPTH = 200    # git_log / git_blame 看得到最近的歷史,又不用整包下載
CLONE_TIMEOUT = 300  # 秒


def github_repo(text: str) -> tuple[str, str] | None:
    """「新增專案」貼的是 GitHub 網址的話,回傳 (作者, 專案);github.com/作者/專案/tree/main/... 也認得。"""
    m = GITHUB_REPO.fullmatch(text.strip())
    return (m[1], m[2]) if m and m[2] not in (".", "..") else None


def _rmtree(p: Path) -> None:
    """git 的物件檔是唯讀的,Windows 上要先改權限才刪得掉;刪不掉(還有行程開著)就算了。"""
    def writable(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    try:
        if p.exists():
            shutil.rmtree(p, onexc=writable)
    except OSError:
        logging.warning("刪不掉下載到一半的 %s", p)


def clone_public(url: str, dest: Path) -> str:
    """下載公開的 repo;成功回空字串,不然回錯誤訊息。
    不用任何登入資訊:清掉 credential helper(含系統的 Git Credential Manager、gh 設的)和 askpass、關掉終端機提示,
    私人 repo 直接失敗 —— 公司程式碼不會被抓下來、再送到 build.nvidia.com。LFS 大檔不下載(留指標檔)。"""
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_ASKPASS", "SSH_ASKPASS")}
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never", GIT_LFS_SKIP_SMUDGE="1")
    cmd = ["git", "-c", "credential.helper=", "-c", "core.askPass=", "clone", "--quiet", "--depth", str(CLONE_DEPTH),
           "--single-branch", "--no-tags", url, str(dest)]
    try:
        r = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=CLONE_TIMEOUT)
    except FileNotFoundError:
        return "找不到 git,請先安裝 Git for Windows"
    except subprocess.TimeoutExpired:
        _rmtree(dest)
        return f"下載超過 {CLONE_TIMEOUT // 60} 分鐘,專案可能太大"
    if r.returncode == 0:
        return ""
    _rmtree(dest)
    err = r.stderr.strip()
    if re.search(r"could not read Username|Authentication failed|Repository not found|not found", err, re.I):
        return "找不到這個公開的 repo。私人 repo 不能下載(模型現在在 build.nvidia.com,公司程式碼不能送過去)"
    return "下載失敗:" + (err.splitlines()[-1] if err else f"git 結束代碼 {r.returncode}")


@cache
def version() -> str:
    """「了解更多」顯示的版本:最近一次 commit 的代碼和日期;不是用 git 拿到的(例如 ZIP)就空白。"""
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).parent), "log", "-1", "--format=%h · %cd", "--date=short"],
                           capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def parse_images(raw) -> list[dict]:
    """/api/ask 帶的圖片:只收 png / jpeg / webp / gif 的 data URL,限張數和大小;不合格丟 ValueError。"""
    if not raw:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_IMAGES:
        raise ValueError(f"一題最多 {MAX_IMAGES} 張圖")
    out = []
    for i, im in enumerate(raw, 1):
        data, thumb = (str(im.get("data", "")), str(im.get("thumb", ""))) if isinstance(im, dict) else ("", "")
        if len(data) > MAX_IMAGE_CHARS:
            raise ValueError("圖片太大,請縮小後再傳")
        if not IMAGE_URL.fullmatch(data) or not IMAGE_URL.fullmatch(thumb) or len(thumb) > MAX_THUMB_CHARS:
            raise ValueError("只能傳 PNG、JPEG、WebP、GIF 圖片")
        name = re.sub(r'[<>"\r\n]', "_", str(im.get("name") or f"圖片{i}"))[:80]  # 會放進 <image name="…">,不能有引號和換行
        out.append({"name": name, "data": data, "thumb": thumb})
    return out


def _title(text: str) -> str:
    line = " ".join(text.split())
    return line[:28] + ("…" if len(line) > 28 else "") or "新對話"


class Desk:
    """一個使用者的桌面(像 Claude 的側邊欄):好幾個專案(工作區),每個專案底下好幾段對話,每段對話一個 Hub。

    對話存在 store/conversations/<id>.json,專案清單存在 store/projects.json:關掉程式再打開,對話還在、也能接著聊。
    store=None 就不存(測試、只有一段對話的舊用法)。
    """

    def __init__(self, make_hub: Callable[[Path], Hub], default_workspace: Path, store: Path | None = None,
                 can_add_projects: bool = True, docs_dir: Path | None = None, repos_dir: Path | None = None,
                 embed: knowledge.Embedder | None = None):
        self.make_hub = make_hub
        self.store = store
        self.can_add_projects = can_add_projects
        self.docs_dir = docs_dir  # 這個人的文件放哪;None = OUTPUT_DIR
        self.repos_dir = repos_dir  # 從 GitHub 下載的專案放哪;None = 不能下載
        self.embed = embed  # 知識庫算向量用的(registry.embedder());None = 只能用關鍵字搜尋
        self.kb_meta: list[dict] = []  # 知識庫:[{id, name, path}],存在 store/knowledge.json
        self.kbs: dict[str, knowledge.KnowledgeBase] = {}  # 用到才載入索引
        self.lock = threading.RLock()
        self.hubs: dict[str, Hub] = {}
        # 對話 id → {"id","title","project","kind","created","updated"};kind:code(綁專案)/ chat(對話模式,project 是 None)
        self.meta: dict[str, dict] = {}
        self.projects: list[dict] = []
        self.settings: dict = {}  # 這個人的設定(帳號選單的回答語言),存在 store/settings.json
        self.focus: str | None = None  # 畫面上最近打開的對話(開事件串流時記下)
        if store:
            self._load()
        # 預設專案(.env 的 AGENT_WORKSPACE;VS Code 外掛是它開著的資料夾)。桌面版和外掛共用同一個 store,
        # 預設專案可能不一樣,所以各自記住自己的,不能拿 projects[0]
        default = next((p for p in self.projects if Path(p["path"]) == default_workspace), None)
        if not default:
            taken = {p["id"] for p in self.projects}
            default = {"id": "p0" if "p0" not in taken else f"p{secrets.token_hex(3)}",
                       "name": default_workspace.name, "path": str(default_workspace)}
            self.projects.append(default)
            self._save_projects()
        self.default_pid = default["id"]

    @classmethod
    def single(cls, hub: Hub) -> "Desk":
        """把一個現成的 Hub 包成只有一段對話的桌面(舊用法、測試)。"""
        desk = cls(lambda ws: hub, hub.workspace)
        desk.hubs["main"] = hub
        desk.meta["main"] = {"id": "main", "title": "對話", "project": "p0", "created": time.time(), "updated": time.time()}
        return desk

    # ---------- 存檔 ----------

    def _conv_dir(self) -> Path:
        return self.store / "conversations"

    def _load(self) -> None:
        try:
            self.projects = json.loads((self.store / "projects.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.projects = []
        for f in self._conv_dir().glob("*.json") if self._conv_dir().is_dir() else []:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                self.meta[data["id"]] = {k: data[k] for k in ("id", "title", "project", "created", "updated")}
                self.meta[data["id"]]["kind"] = data.get("kind", "code")         # 以前的對話都是綁專案的
                self.meta[data["id"]]["draft"] = sorted(data.get("draft", {}))  # 還沒套用的草稿,「待命」要列
                self.meta[data["id"]]["plan"] = bool(data.get("plan"))           # 等核准的計畫
            except (OSError, ValueError, KeyError):
                continue  # 壞掉的檔案略過,不擋其他對話
        self._read_settings()
        try:
            self.kb_meta = json.loads((self.store / "knowledge.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.kb_meta = []

    def _read_settings(self) -> None:
        """桌面版和 VS Code 外掛共用同一份:每次開對話都重讀,另一邊改的語言新對話就會用到。"""
        try:
            self.settings = json.loads((self.store / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            pass

    @property
    def lang(self) -> str:
        lang = self.settings.get("lang")
        return lang if lang in agent.ANSWER_LANGS else "zh-TW"

    def set_lang(self, lang: str) -> None:
        """帳號選單的「回答語言」:存起來,已經打開的對話也馬上換。"""
        if lang not in agent.ANSWER_LANGS:
            raise ValueError("沒有這個語言")
        with self.lock:
            self.settings["lang"] = lang
            if self.store:
                self.store.mkdir(parents=True, exist_ok=True)
                (self.store / "settings.json").write_text(json.dumps(self.settings, ensure_ascii=False), encoding="utf-8")
            for hub in self.hubs.values():
                hub.use_lang(lang)

    def _save_projects(self) -> None:
        if self.store:
            try:  # 另一個行程(桌面版 / VS Code 外掛)可能剛加了專案:合併進來,不要蓋掉
                on_disk = json.loads((self.store / "projects.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                on_disk = []
            known = {p["id"] for p in self.projects} | {p["path"] for p in self.projects}
            self.projects += [p for p in on_disk if p.get("id") not in known and p.get("path") not in known]
            self.store.mkdir(parents=True, exist_ok=True)
            (self.store / "projects.json").write_text(json.dumps(self.projects, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")

    def save(self, cid: str) -> None:
        with self.lock:
            hub, meta = self.hubs.get(cid), self.meta.get(cid)
            if not hub or not meta:
                return
            if meta["title"] == "新對話":
                first = next((e["text"] for e in hub.events if e["kind"] == "user"), "")
                if first:
                    meta["title"] = _title(first)
            meta["updated"] = time.time()
            meta["draft"] = sorted(hub.bench.draft)
            meta["plan"] = hub.plan_waiting
            if not self.store:
                return
            self._conv_dir().mkdir(parents=True, exist_ok=True)
            tmp = self._conv_dir() / f"{cid}.json.tmp"  # 先寫暫存檔再改名:寫到一半當掉不會留下壞檔
            tmp.write_text(json.dumps({**meta, **hub.export()}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._conv_dir() / f"{cid}.json")

    # ---------- 專案 / 對話 ----------

    def project(self, pid: str) -> dict | None:
        return next((p for p in self.projects if p["id"] == pid), None)

    def add_project(self, path: str, name: str = "") -> dict:
        """加一個專案:電腦上的資料夾,或公開的 GitHub 網址(下載到 repos_dir)。"""
        if not self.can_add_projects:
            raise PermissionError("多人模式的專案由管理者設定")
        if gh := github_repo(path):
            return self._clone(*gh, name)
        p = Path(path.strip().strip('"')).expanduser()
        if not p.is_dir():
            raise ValueError(f"找不到資料夾:{path}")
        p = p.resolve()
        with self.lock:
            if existing := next((x for x in self.projects if Path(x["path"]) == p), None):
                return existing
            proj = {"id": f"p{secrets.token_hex(3)}", "name": name.strip() or p.name, "path": str(p)}
            self.projects.append(proj)
            self._save_projects()
            return proj

    def _clone(self, owner: str, repo: str, name: str = "") -> dict:
        if not self.repos_dir:
            raise ValueError("這裡不能從 GitHub 下載專案")
        url = f"https://github.com/{owner}/{repo}"
        with self.lock:
            if existing := next((x for x in self.projects if (x.get("source") or "").lower() == url.lower()), None):
                return existing
        dest, n = self.repos_dir / f"{owner}-{repo}", 2
        while dest.exists():
            dest, n = self.repos_dir / f"{owner}-{repo}-{n}", n + 1
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        if err := clone_public(url + ".git", dest):  # 不拿著鎖下載:大專案要好幾分鐘,其他對話不能卡住
            raise ValueError(err)
        dest = dest.resolve()
        with self.lock:
            # 規則檔放在 repo 外面:repo 自帶的 ASSISTANT.md 是別人寫的,不能直接進系統提示
            proj = {"id": f"p{secrets.token_hex(3)}", "name": name.strip() or repo, "path": str(dest),
                    "source": url, "rules": str(dest.with_name(dest.name + ".rules.md"))}
            self.projects.append(proj)
            self._save_projects()
            return proj

    def _make(self, meta: dict) -> Hub:
        if meta.get("kind") == "chat":  # 不綁專案:工具裡沒有讀檔的,工作區只是佔位
            hub = self.make_hub(Path(self.project(self.default_pid)["path"]))
            hub.use_chat()
            return hub
        proj = self.project(meta["project"]) or self.project(self.default_pid)  # 專案被拿掉了:放回預設專案
        hub = self.make_hub(Path(proj["path"]))
        if proj.get("rules"):
            hub.use_rules(Path(proj["rules"]))
        return hub

    def _attach(self, cid: str, hub: Hub) -> Hub:
        hub.on_event = lambda kind: self.save(cid) if kind in SAVE_ON else None
        if self.store:
            self._read_settings()
        if self.lang != "zh-TW":
            hub.use_lang(self.lang)
        if kbs := self.ready_knowledge():
            hub.use_knowledge(kbs)
        self.hubs[cid] = hub
        return hub

    # ---------- 知識庫(側邊欄的「知識庫」頁面) ----------

    @property
    def can_add_knowledge(self) -> bool:
        return self.can_add_projects and self.store is not None  # 訪客、多人模式不能加;沒有存檔的地方也不能

    def _kb(self, meta: dict) -> knowledge.KnowledgeBase:
        with self.lock:
            if meta["id"] not in self.kbs:
                self.kbs[meta["id"]] = knowledge.KnowledgeBase(Path(meta["path"]), self.store / "knowledge" / meta["id"],
                                                               self.embed, meta["name"])
            return self.kbs[meta["id"]]

    def ready_knowledge(self) -> list[knowledge.KnowledgeBase]:
        """建好索引、有內容的知識庫(給 search_knowledge 搜)。"""
        return [kb for kb in map(self._kb, list(self.kb_meta)) if kb.chunks]

    def _share_knowledge(self) -> None:
        kbs = self.ready_knowledge()
        with self.lock:
            hubs = list(self.hubs.values())
        for hub in hubs:
            hub.use_knowledge(kbs)

    def _save_knowledge(self) -> None:
        self.store.mkdir(parents=True, exist_ok=True)
        (self.store / "knowledge.json").write_text(json.dumps(self.kb_meta, ensure_ascii=False, indent=1), encoding="utf-8")

    def add_knowledge(self, path: str, name: str = "") -> dict:
        """加一個資料夾當知識庫,馬上在背景建索引。"""
        if not self.can_add_knowledge:
            raise PermissionError("這裡不能加知識庫")
        p = Path(path.strip().strip('"')).expanduser()
        if not p.is_dir():
            raise ValueError(f"找不到資料夾:{path}")
        p = p.resolve()
        with self.lock:
            if existing := next((m for m in self.kb_meta if Path(m["path"]) == p), None):
                return existing
            meta = {"id": f"k{secrets.token_hex(3)}", "name": name.strip() or p.name, "path": str(p)}
            self.kb_meta.append(meta)
            self._save_knowledge()
        self.update_knowledge(meta["id"])
        return meta

    def update_knowledge(self, kid: str) -> bool:
        """在背景更新索引(只重算改過的檔案);已經在更新就回 False。"""
        meta = next((m for m in self.kb_meta if m["id"] == kid), None)
        if not meta:
            return False
        kb = self._kb(meta)
        if kb.state == "indexing":
            return False
        kb.state = "indexing"  # 先標起來:畫面馬上看得到,也不會按兩次跑兩個

        def run() -> None:
            kb.update()
            self._share_knowledge()  # 段落、檔案數變了:系統提示裡的知識庫清單跟著換

        threading.Thread(target=run, daemon=True).start()
        return True

    def remove_knowledge(self, kid: str) -> bool:
        with self.lock:
            meta, kb = next((m for m in self.kb_meta if m["id"] == kid), None), self.kbs.get(kid)
            if not meta or (kb and kb.state == "indexing"):  # 建索引中不能拿掉(背景執行緒還在寫)
                return False
            self.kb_meta.remove(meta)
            self.kbs.pop(kid, None)
            self._save_knowledge()
        _rmtree(self.store / "knowledge" / kid)  # 只刪索引,資料夾裡的文件不動
        self._share_knowledge()
        return True

    def knowledge_listing(self, paths: bool = True) -> list[dict]:
        out = []
        for meta in list(self.kb_meta):
            kb = self._kb(meta)
            item = {"id": meta["id"], **kb.summary(), "path": meta["path"] if paths else meta["name"],
                    "missing": not Path(meta["path"]).is_dir()}
            if kb.state != "indexing" and not item["missing"]:
                item["changes"] = kb.changes()  # 有幾個檔案改過、還沒更新
            out.append(item)
        return out

    def new_conversation(self, pid: str | None = None, kind: str = "code") -> str:
        """kind:code = 綁專案(pid,沒給用預設專案);chat = 對話模式,不綁專案。"""
        with self.lock:
            chat = kind == "chat"
            proj = None if chat else self.project(pid or self.default_pid) or self.project(self.default_pid)
            cid = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
            self.meta[cid] = {"id": cid, "title": "新對話", "project": proj["id"] if proj else None,
                              "kind": "chat" if chat else "code", "created": time.time(), "updated": time.time()}
            self._attach(cid, self._make(self.meta[cid]))
            return cid

    def hub(self, cid: str | None) -> Hub:
        """拿這段對話的 Hub;還沒載入就從硬碟讀。沒給 id(VS Code 外掛的「問選取的程式碼」)就用畫面上最近打開的那段,
        再沒有就用最近更新的(都沒有就開一段新的)。"""
        with self.lock:
            if not cid or cid not in self.meta:
                cid = (self.focus if self.focus in self.meta else None) or \
                    max(self.meta, key=lambda c: self.meta[c]["updated"], default=None) or self.new_conversation()
            if cid in self.hubs:
                return self.hubs[cid]
            hub = self._attach(cid, self._make(self.meta[cid]))
            try:
                hub.restore(json.loads((self._conv_dir() / f"{cid}.json").read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError) as e:
                hub.publish("notice", {"text": f"這段對話的紀錄讀不出來,從頭開始:{e}"})
            return hub

    @property
    def listeners(self) -> int:
        """全部對話加起來有幾個頁面連著(桌面版看這個判斷視窗是不是都關了)。"""
        with self.lock:
            return sum(h.listeners for h in self.hubs.values())

    def cid_of(self, hub: Hub) -> str | None:
        return next((c for c, h in self.hubs.items() if h is hub), None)

    def rename(self, cid: str, title: str) -> None:
        with self.lock:
            if cid in self.meta and title.strip():
                self.meta[cid]["title"] = _title(title)
                if cid in self.hubs:
                    self.save(cid)

    def delete(self, cid: str) -> bool:
        with self.lock:
            hub = self.hubs.get(cid)
            if hub and hub.busy:
                return False
            self.hubs.pop(cid, None)
            self.meta.pop(cid, None)
            if self.store:
                (self._conv_dir() / f"{cid}.json").unlink(missing_ok=True)
            return True

    def listing(self) -> dict:
        with self.lock:
            convs = sorted(self.meta.values(), key=lambda m: m["updated"], reverse=True)
            return {"projects": self.projects, "can_add_projects": self.can_add_projects, "default_project": self.default_pid,
                    "conversations": [{**m, "busy": bool(self.hubs.get(m["id"]) and self.hubs[m["id"]].busy)}
                                      for m in convs]}

    # ---------- 側邊欄「成品」「待命」:跨所有對話 ----------

    def outputs(self) -> dict:
        with self.lock:
            loaded = list(self.hubs.items())
        patches = []
        for cid, h in loaded:
            for p in h.outputs()["patches"]:
                patches.append({**p, "conv": cid, "title": self.meta.get(cid, {}).get("title", "")})
        return {"docs": docs_in(self.docs_dir or toolbox.OUTPUT_DIR), "patches": patches}

    def doc_file(self, name: str) -> Path | None:
        return doc_path(self.docs_dir or toolbox.OUTPUT_DIR, name)

    def pending(self) -> list[dict]:
        """還沒載入的對話,看存檔時記下的草稿;載入的看現在的狀態(含等回答的確認、等核准的計畫)。"""
        with self.lock:
            loaded = dict(self.hubs)
            metas = list(self.meta.values())
        items = []
        for m in metas:
            cid = m["id"]
            if cid in loaded:
                st = loaded[cid].pending_state()
            else:
                st = {"draft": m.get("draft", []), "confirms": [], "plan": m.get("plan", False),
                      "patch": -1 if m.get("draft") else None,
                      "passed": None, "team": not self.can_add_projects, "busy": False}
            if st["draft"] or st["confirms"] or st["plan"]:
                items.append({**st, "conv": cid, "title": m.get("title", "")})
        return items


class Hubs:
    """每個使用者一張 Desk(各自的專案、對話、草稿、確認、額度)。make 回傳 Hub 的話包成只有一段對話的 Desk。"""

    def __init__(self, make: Callable[[str], Desk | Hub]):
        self.make = make
        self.lock = threading.Lock()
        self.desks: dict[str, Desk] = {}

    def get(self, user: str) -> Desk:
        with self.lock:
            if user not in self.desks:
                made = self.make(user)
                self.desks[user] = Desk.single(made) if isinstance(made, Hub) else made
            return self.desks[user]


@dataclass
class TeamAccess:
    """多人模式:使用者身分由前面的登入代理(公司 SSO / oauth2-proxy)放進標頭,伺服器只認這個標頭。
    一定要讓使用者只能經過代理連進來(容器不對外開 port),不然標頭可以自己偽造。"""
    user_header: str = "X-Forwarded-Email"
    allowed_hosts: set[str] = field(default_factory=set)  # 代理對外的網址(Host 標頭);空的 = 不檢查


def make_handler(hubs: Hubs | Desk | Hub, token: str, port: int, lan: LanAccess | None = None,
                 team: TeamAccess | None = None, phone: PhoneLink | None = None,
                 guests: Callable[[str], Desk] | None = None) -> type[BaseHTTPRequestHandler]:
    if isinstance(hubs, (Hub, Desk)):  # 單人:不管誰連進來都是同一張桌面
        single = hubs if isinstance(hubs, Desk) else Desk.single(hubs)
        hubs = Hubs(lambda user: single)
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"} | {f"{ip}:{port}" for ip in (lan.ips if lan else [])}
    if team:
        allowed_hosts |= team.allowed_hosts

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
            if (team and not team.allowed_hosts) or self.headers.get("Host") in allowed_hosts:
                return True
            self._json(403, {"error": "只接受本機連線"})
            return False

        def _local(self) -> bool:
            return self.client_address[0] in LOCAL_ADDRS

        def _user(self) -> str | None:
            if not team:
                return "local"
            return self.headers.get(team.user_header, "").strip() or None

        def _device(self, app: bool = False) -> str:
            """設定頁顯示的裝置類型(從 User-Agent 猜)。"""
            return device_label(self.headers.get("User-Agent", ""), app)

        def _sid(self) -> str | None:
            """手機登入的 cookie。"""
            morsel = SimpleCookie(self.headers.get("Cookie", "")).get(COOKIE)
            return morsel.value if morsel else None

        def _guest(self) -> str | None:
            """用訪客 PIN 配對的手機:回傳它的代號(每支手機一個自己的空間);其他人 None。"""
            if team or lan is None or guests is None or self._local():
                return None
            who = lan.identify(self._sid(), self._device()) or ""
            return who.split(":", 1)[1] if who.startswith("guest:") else None

        def _label(self) -> str:
            """側邊欄左下角顯示的身分。"""
            return self._user() if team else ("訪客" if self._guest() else "")

        def _desk(self) -> Desk:
            gid = self._guest()
            return guests(gid) if gid else hubs.get(self._user())

        def _hub(self) -> Hub:
            """這個請求是哪段對話:POST 看 X-Conv 標頭,GET(事件串流、下載)看網址的 c=;沒給就用最近的那段。"""
            cid = self.headers.get("X-Conv") or (parse_qs(urlsplit(self.path).query).get("c") or [None])[0]
            return self._desk().hub(cid)

        def _signed_in(self) -> bool:
            """多人模式看登入代理給的身分;單人模式這台電腦自己連的直接過,手機等其他裝置要先用 PIN 登入(cookie)。"""
            if team:
                return self._user() is not None
            if self._local():
                return True
            return lan is not None and lan.valid(self._sid(), self._device())

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
            qs = parse_qs(url.query)
            if url.path in PUBLIC_FILES:  # 視窗、工作列、手機主畫面的圖示
                name, ctype = PUBLIC_FILES[url.path]
                if (ASSETS / name).is_file():
                    return self._send(200, (ASSETS / name).read_bytes(), ctype)
                return self._json(404, {"error": "沒有圖示"})
            if url.path == "/manifest.webmanifest":
                return self._send(200, json.dumps(MANIFEST, ensure_ascii=False).encode(), "application/manifest+json")
            if url.path == "/":
                if team and not self._signed_in():
                    return self._json(401, {"error": f"沒有登入身分(登入代理沒有送 {team.user_header})"})
                if not self._signed_in():  # 不給頁面,就拿不到 token
                    return self._send(200, LOGIN_PAGE.encode(), "text/html; charset=utf-8")
                page = PAGE.read_text(encoding="utf-8").replace("__TOKEN__", token)
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                if self._token_ok(self.headers.get("X-Token")):
                    hub, guest, desk = self._hub(), self._guest(), self._desk()
                    quota, who = hub.session.quota, hub.session.user
                    state = {"workspace": "" if hub.chat else hub.workspace.name if guest else str(hub.workspace),  # 訪客不給電腦上的路徑
                             "status": hub.status, "busy": hub.busy, "chat": hub.chat,
                             "model": hub.model, "draft": sorted(hub.bench.draft), "team": hub.team,
                             "user": self._label(), "guest": bool(guest), "conv": desk.cid_of(hub),
                             "instructions": not hub.chat and hub.rules_file.is_file(),
                             "role": hub.session.requested_role, "roles": hub.roles(), "effort": hub.session.effort,
                             # 今天的 token 用量(側邊欄左下角);訪客看到的是自己的額度
                             "usage": {"used": quota.used(who), "limit": quota.daily_tokens}
                             if hasattr(quota, "used") else None,
                             # 帳號選單:回答語言、有沒有「手機與訪客」(只有桌面版、電腦自己看)、「登出」(只有手機)
                             "lang": desk.lang, "phone": bool(phone and self._local()),
                             "can_logout": bool(lan and not self._local()),
                             "backend": hub.session.registry.backend, "version": version()}
                    if lan and self._local():  # 手機連線方式只給這台電腦看
                        state["lan"] = {"urls": [f"http://{ip}:{port}" for ip in lan.ips], "pin": lan.pin}
                    self._json(200, state)
            elif url.path == "/api/phone":  # 手機連線的狀態(含 PIN):只給這台電腦自己看
                if self._token_ok(self.headers.get("X-Token")):
                    if not phone or not self._local():
                        return self._json(404, {"error": "沒有手機連線功能"})
                    self._json(200, phone.status())
            elif url.path == "/api/desk":  # 側邊欄:專案和對話清單
                if self._token_ok(self.headers.get("X-Token")):
                    listing = self._desk().listing()
                    if self._guest():  # 訪客不給電腦上的路徑
                        listing["projects"] = [{**p, "path": p["name"]} for p in listing["projects"]]
                    self._json(200, {**listing, "user": self._label()})
            elif url.path == "/api/knowledge":  # 知識庫頁面:清單、建索引的進度
                if self._token_ok(self.headers.get("X-Token")):
                    desk = self._desk()
                    self._json(200, {"items": desk.knowledge_listing(paths=not self._guest()),  # 訪客不給電腦上的路徑
                                     "can_add": desk.can_add_knowledge, "semantic": desk.embed is not None,
                                     "backend": self._hub().session.registry.backend})
            elif url.path == "/api/outputs":  # 成品
                if self._token_ok(self.headers.get("X-Token")):
                    self._json(200, self._desk().outputs())
            elif url.path == "/api/pending":  # 待命
                if self._token_ok(self.headers.get("X-Token")):
                    self._json(200, {"items": self._desk().pending()})
            elif url.path == "/api/instructions":  # 自訂:這段對話所在專案的 ASSISTANT.md
                if self._token_ok(self.headers.get("X-Token")):
                    hub = self._hub()
                    f = hub.rules_file
                    if hub.chat:  # 對話模式不綁專案,沒有專案規則
                        return self._json(200, {"chat": True, "text": "", "project": "", "team": hub.team})
                    self._json(200, {"text": f.read_text(encoding="utf-8", errors="replace")[:20000] if f.is_file() else "",
                                     "project": hub.workspace.name, "team": hub.team,
                                     "downloaded": hub.bench.rules is not None})
            elif url.path == "/api/events":
                if self._token_ok((qs.get("token") or [None])[0]):  # EventSource 不能帶標頭,token 放網址
                    hub, desk = self._hub(), self._desk()
                    desk.focus = desk.cid_of(hub)
                    self._stream(hub, int(self.headers.get("Last-Event-ID") or 0))
            elif url.path == "/api/patch":  # 下載 patch(多人模式不能直接套用,自己下載回去套)
                if self._token_ok((qs.get("token") or [None])[0]):
                    p = self._hub().patch_file(int((qs.get("i") or ["-1"])[0]))
                    if not p:
                        return self._json(404, {"error": "找不到這個 patch"})
                    self._send(200, p.read_bytes(), "text/x-diff; charset=utf-8",
                               {"Content-Disposition": f'attachment; filename="{p.name}"'})
            elif url.path == "/api/file":  # 下載「成品」裡的文件
                if self._token_ok((qs.get("token") or [None])[0]):
                    p = self._desk().doc_file((qs.get("name") or [""])[0])
                    if not p:
                        return self._json(404, {"error": "找不到這個檔案"})
                    self._send(200, p.read_bytes(), "application/octet-stream",
                               {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(p.name)}"})
            else:
                self._json(404, {"error": "沒有這個頁面"})

        def _login(self, raw: bytes) -> None:
            if lan is None:
                return self._json(404, {"error": "沒有開手機連線(--lan)"})
            try:
                body = json.loads(raw or b"{}")
                sid = lan.login(str(body.get("pin", "")), self._device(bool(body.get("app"))))
            except PermissionError as e:
                return self._json(429, {"error": str(e)})
            except (json.JSONDecodeError, AttributeError):
                return self._json(400, {"error": "格式錯誤"})
            if not sid:
                return self._json(403, {"error": "PIN 不對"})
            # 記住配對的話 cookie 留一年(手機主畫面的 app 每次打開不用再輸入 PIN);Lax:從主畫面打開也會帶上
            keep = f"; Max-Age={LanAccess.REMEMBER}" if lan.store else ""
            self._json(200, {}, {"Set-Cookie": f"{COOKIE}={sid}; HttpOnly; SameSite=Lax; Path=/{keep}"})

        def _stream(self, hub: Hub, since: int) -> None:
            """Server-Sent Events:有新事件就送,沒有就每 15 秒送一次 ping 保持連線。"""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with hub.cond:
                hub.listeners += 1
            try:
                while True:
                    batch = hub.events_since(since, timeout=15)
                    if not batch:
                        self.wfile.write(b": ping\n\n")
                    for ev in batch:
                        self.wfile.write(f"id: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                        since = ev["seq"]
                    self.wfile.flush()
                    if not self._signed_in():  # 手機連線關了、或按了「忘記所有手機」:連著的也斷開
                        return
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return  # 瀏覽器關掉或重新整理
            finally:
                with hub.cond:
                    hub.listeners -= 1

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
            desk = self._desk()
            if path in ("/api/phone", "/api/phone/forget"):  # 「更多」的手機連線開關:只有這台電腦自己能按
                if not phone or not self._local():
                    return self._json(404, {"error": "沒有手機連線功能"})
                if path == "/api/phone/forget" and body.get("id"):  # 只取消一支訪客手機
                    if not phone.lan.forget_device(str(body["id"])):
                        return self._json(404, {"error": "找不到這支手機"})
                elif path == "/api/phone/forget":
                    phone.lan.forget("guest" if body.get("who") == "guests" else None)
                elif err := (phone.start() if body.get("on") else phone.stop()):
                    return self._json(409, {"error": err})
                return self._json(200, phone.status())
            if path == "/api/logout":  # 帳號選單的「登出」:只有用 PIN 登入的手機有
                if not lan or self._local():
                    return self._json(404, {"error": "這台電腦不用登出"})
                lan.logout(self._sid())
                return self._json(200, {}, {"Set-Cookie": f"{COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0"})
            if path == "/api/settings":  # 帳號選單的「回答語言」
                try:
                    desk.set_lang(str(body.get("lang", "")))
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                return self._json(200, {"lang": desk.lang})
            if path == "/api/conv/new":  # 側邊欄的「新增」、專案旁邊的「+」;kind=chat 是「對話」模式
                kind = "chat" if body.get("kind") == "chat" else "code"
                return self._json(200, {"id": desk.new_conversation(body.get("project"), kind)})
            if path == "/api/conv/rename":
                desk.rename(str(body.get("id")), str(body.get("title", "")))
                return self._json(200, {})
            if path == "/api/conv/delete":
                ok = desk.delete(str(body.get("id")))
                return self._json(200 if ok else 409, {} if ok else {"error": "這段對話還在回答中,等結束再刪"})
            if path == "/api/knowledge/add":
                try:
                    return self._json(200, desk.add_knowledge(str(body.get("path", "")), str(body.get("name", ""))))
                except (ValueError, PermissionError) as e:
                    return self._json(400, {"error": str(e)})
            if path == "/api/knowledge/update":
                ok = desk.update_knowledge(str(body.get("id")))
                return self._json(202 if ok else 409, {} if ok else {"error": "找不到這個知識庫,或正在建索引"})
            if path == "/api/knowledge/remove":
                ok = desk.remove_knowledge(str(body.get("id")))
                return self._json(200 if ok else 409, {} if ok else {"error": "找不到這個知識庫,或正在建索引"})
            if path == "/api/knowledge/search":  # 知識庫頁面的「試搜尋」:看找到哪幾段(不經過模型)
                q = " ".join(str(body.get("q", "")).split())
                if not q:
                    return self._json(400, {"error": "要搜尋什麼?"})
                hits, how = knowledge.search_all(desk.ready_knowledge(), q)
                return self._json(200, {"how": how, "hits": [{**h, "where": knowledge.label(h)} for h in hits]})
            if path == "/api/project/add":
                try:
                    return self._json(200, desk.add_project(str(body.get("path", "")), str(body.get("name", ""))))
                except (ValueError, PermissionError) as e:
                    return self._json(400, {"error": str(e)})
            hub = self._hub()
            if path == "/api/ask":
                text = str(body.get("text", "")).strip()
                try:
                    images = parse_images(body.get("images"))
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                if not text and not images:
                    return self._json(400, {"error": "問題是空的"})
                text = text or "請說明這張圖。"
                ok = hub.ask(text, "plan" if body.get("mode") == "plan" and not hub.chat else "normal", images=images)
                self._json(202 if ok else 409, {} if ok else {"error": "上一個問題還在回答中"})
            elif path == "/api/execute":  # 規劃模式交出計畫後:照計畫執行 / 先不要
                ok = hub.execute_plan(bool(body.get("go")))
                self._json(202 if ok else 409, {} if ok else {"error": "上一個問題還在回答中"})
            elif path == "/api/escalate":  # 做到步數上限後:換模型接著做 / 先不要
                msg = hub.escalate(bool(body.get("go")))
                self._json(409 if msg else 202, {"error": msg} if msg else {})
            elif path == "/api/confirm":
                ok = hub.answer(str(body.get("id")), str(body.get("answer")))
                self._json(200 if ok else 404, {} if ok else {"error": "這個確認已經結束了"})
            elif path == "/api/apply":
                self._json(200, {"text": hub.apply(int(body.get("patch", -1)), str(body.get("action", "keep")))})
            elif path == "/api/clear":
                ok = hub.clear()
                self._json(200 if ok else 409, {} if ok else {"error": "還在回答中,等結束再開新對話"})
            elif path == "/api/role":  # 輸入框的「選模型」:這段對話之後都用它(null = 自動)
                role = body.get("role")
                if msg := hub.set_role(str(role) if role else None):
                    return self._json(409, {"error": msg})
                desk.save(desk.cid_of(hub) or "")
                self._json(200, {"role": hub.session.requested_role})
            elif path == "/api/effort":  # 「選模型」裡的努力程度滑桿:這段對話之後都用它
                if msg := hub.set_effort(str(body.get("effort", ""))):
                    return self._json(409 if hub.busy else 400, {"error": msg})
                desk.save(desk.cid_of(hub) or "")
                self._json(200, {"effort": hub.session.effort})
            elif path == "/api/compact":
                msg = hub.compact()  # 成功時摘要通知會從事件串流送出
                self._json(200, {"text": msg})
            elif path == "/api/remember":
                rule = " ".join(str(body.get("rule", "")).split())
                if not rule:
                    return self._json(400, {"error": "規則是空的"})
                msg = hub.remember(rule)
                self._json(403 if msg else 200, {"error": msg} if msg else {})
            else:
                self._json(404, {"error": "沒有這個 API"})

    return Handler


class _Server(ThreadingHTTPServer):
    # Windows 的 SO_REUSEADDR 會讓第二個程式也綁上同一個 port,兩個一起收請求、由誰回應不一定
    # (10/9 實測:舊版和新版同時開在 8765)。Windows 上關掉,第二個直接報錯
    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True


def serve(hubs: Hubs | Hub, port: int = 8765, token: str | None = None, lan: LanAccess | None = None,
          host: str = "127.0.0.1", team: TeamAccess | None = None, phone: PhoneLink | None = None,
          guests: Callable[[str], Desk] | None = None) -> ThreadingHTTPServer:
    """建立伺服器(還沒開始跑);port=0 會自動挑一個空的(測試用)。port 被占用時丟 OSError。"""
    token = token or secrets.token_urlsafe(24)
    server = _Server((host, port), make_handler(hubs, token, port, lan, team, phone, guests))
    if port == 0:  # 實際的 port 要等綁定後才知道,Host 檢查要用它
        server.RequestHandlerClass = make_handler(hubs, token, server.server_address[1], lan, team, phone, guests)
    return server


def local_desk(registry, quota, tools: list, status: str, model: str, role: str | None = None,
               thinking: bool | None = None) -> Desk:
    """自己電腦用的桌面(網頁、桌面版、VS Code 外掛共用):對話存在 OUTPUT_DIR/state,下次打開還在。
    預設專案是 .env 的 AGENT_WORKSPACE(VS Code 外掛會換成它開著的資料夾),其他專案可以在側邊欄加。"""
    def make_hub(workspace: Path) -> Hub:
        return Hub(lambda on_event: agent.AgentSession(registry, quota, role=role, thinking=thinking, tools=tools,
                                                       on_event=on_event),
                   status=status, model=model, workspace=workspace)
    return Desk(make_hub, toolbox.WORKSPACE, store=toolbox.OUTPUT_DIR / "state", repos_dir=toolbox.OUTPUT_DIR / "repos",
                embed=registry.embedder())


GUEST_DAILY_TOKENS = 200_000  # 每支訪客手機每天的額度(用的是主人的 API 金鑰)
# 訪客用不到的工具:空資料夾沒有程式可以測、可以改、可以看 git;也不讓訪客在主人電腦上跑沙箱
GUEST_BLOCKED_TOOLS = {"run_tests", "run_command", "propose_patch", "git_log", "git_blame"}


def guest_desks(registry, tools: list, status: str, model: str, base: Path | None = None) -> Callable[[str], Desk]:
    """訪客(用訪客 PIN 配對的手機)的桌面:每支手機一個 —— 工作區是自己的空資料夾(讀不到主人的專案)、
    自己的對話和文件;跟多人模式一樣不能套用修改、改規則、加專案;另外一份比較小的每日額度。
    都存在 OUTPUT_DIR/guests/<代號>/,主人的電腦上看得到(登入頁有告訴訪客)。"""
    from token_guard import make_quota

    base = base or toolbox.OUTPUT_DIR / "guests"
    quota = make_quota(GUEST_DAILY_TOKENS, 1, store=base / "usage.json")
    allowed = [t for t in tools if t.name not in GUEST_BLOCKED_TOOLS]
    desks: dict[str, Desk] = {}
    lock = threading.Lock()

    def get(gid: str) -> Desk:
        with lock:
            if gid not in desks:
                home = base / gid
                space = home / "訪客空間"
                space.mkdir(parents=True, exist_ok=True)
                desks[gid] = Desk(lambda ws: Hub(lambda on_event: agent.AgentSession(
                    registry, quota, user=f"guest-{gid}", tools=allowed, on_event=on_event),
                    status=status, model=model, team=True, patch_dir=home / "patches", output_dir=home / "docs",
                    workspace=ws, screen=True), space, store=home / "state", can_add_projects=False, docs_dir=home / "docs")
            return desks[gid]

    get.desks = desks  # PhoneLink.guest_listeners 用
    get.quota, get.base = quota, base  # PhoneLink.guest_usage 用:設定頁列出每支訪客手機的用量
    return get


def main() -> None:
    ap = argparse.ArgumentParser(description="HSUAN 的網頁介面")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="不要自動打開瀏覽器")
    ap.add_argument("--lan", action="store_true", help="讓同一個 Wi-Fi 的手機也能開(要輸入啟動時顯示的 PIN)")
    ap.add_argument("--token", help="指定 API token(VS Code 外掛用:它自己產生,才能直接送問題);不給就隨機產生")
    ap.add_argument("--role", help="強制指定模型角色:code(日常)/ code_deep(難題)/ long_context")
    ap.add_argument("--think", choices=["on", "off"], help="思考模式;不給就用伺服器預設")
    team_args = ap.add_argument_group("多人模式(部署在伺服器、前面接公司登入代理;見 deploy/README.md)")
    team_args.add_argument("--team", action="store_true", help="每人各自的對話 / 草稿 / 額度;工作區唯讀,改成下載 patch")
    team_args.add_argument("--user-header", default="X-Forwarded-Email", help="登入代理放使用者身分的標頭")
    team_args.add_argument("--allowed-host", action="append", default=[],
                           help="登入代理對外的網址(Host 標頭,可重複);不給就不檢查 Host")
    team_args.add_argument("--max-concurrent", type=int, default=4, help="全部使用者同時執行的上限,其他人排隊")
    team_args.add_argument("--host", help="監聽位址(多人模式預設 0.0.0.0,單人預設 127.0.0.1)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    registry, quota, tools, allow_web = agent.prepare()
    thinking = {"on": True, "off": False}.get(args.think)
    status = agent.web_status(allow_web)
    model = registry.spec(args.role or registry.routing["default_role"]).model.split("/")[-1]

    if args.team:
        if args.lan:
            raise SystemExit("--team 和 --lan 不能一起用")
        gate = Gate(args.max_concurrent)

        def team_desk(user: str) -> Desk:
            # 每人一個資料夾:對話、patch、文件都放這裡(共用的工作區是唯讀的);專案由管理者決定,不能自己加
            base = toolbox.OUTPUT_DIR / "users" / re.sub(r"[^\w.@-]", "_", user)[:80]
            return Desk(lambda ws: Hub(lambda on_event: agent.AgentSession(
                registry, quota, user=user, role=args.role, thinking=thinking, tools=tools, on_event=on_event),
                status=status, model=model, team=True, gate=gate, patch_dir=base / "patches",
                output_dir=base / "docs", workspace=ws),
                toolbox.WORKSPACE, store=base / "state", can_add_projects=False, docs_dir=base / "docs")

        hubs: Hubs | Desk = Hubs(team_desk)
        team = TeamAccess(args.user_header, set(args.allowed_host))
    else:
        hubs, team = local_desk(registry, quota, tools, status, model, args.role, thinking), None
    lan = LanAccess(lan_ips()) if args.lan else None
    if lan and not lan.ips:
        raise SystemExit("找不到這台電腦的區域網路位址,請確認有連上 Wi-Fi 或網路線")
    host = args.host or ("0.0.0.0" if lan or team else "127.0.0.1")
    try:
        server = serve(hubs, args.port, token=args.token, lan=lan, host=host, team=team)
    except OSError:
        raise SystemExit(f"port {args.port} 已經被占用:可能已經有一個 HSUAN 在跑了(請先關掉它),"
                         f"或用 --port 換一個")
    url = f"http://127.0.0.1:{args.port}/"
    if team:
        print(f"HSUAN 多人模式:{host}:{args.port}  (工作區 {toolbox.WORKSPACE},身分標頭 {team.user_header},"
              f"同時執行上限 {args.max_concurrent})")
        server.serve_forever()
        return
    print(f"HSUAN 網頁介面:{url}  (工作區 {toolbox.WORKSPACE};按 Ctrl+C 結束)")
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
