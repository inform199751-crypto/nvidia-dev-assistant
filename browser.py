"""操作瀏覽器(10/15):HSUAN 用無頭的 Edge / Chrome 測試使用者自己開發中的網站(像 Claude 的「操作瀏覽器」)。

不裝 Playwright(還要另外下載約 150 MB 的瀏覽器):直接用電腦上的 Edge,透過 DevTools 協定(CDP)控制,
連線用已經裝好的 websockets。每段對話一個分頁(Workbench.browser),整個 HSUAN 共用一個瀏覽器行程。

安全:
- 只開開發用的位址(is_dev_url):localhost、127.0.0.1、[::1]、*.localhost、*.test;點了導到別的網站就停下來、回到空白頁。
- 獨立的暫存設定檔:沒有使用者的登入、cookie、擴充功能;HSUAN 結束時刪掉。
- 打開網址前先問使用者(跟上網一樣可以「之後都允許」);試用平台上提醒畫面內容會送到 NVIDIA。
- 工具說明要求只用測試資料,不輸入真的密碼或個資。
- 畫面內容是使用者自己的開發網站,不另外過安全模型(每個動作都檢查太慢);導到外部網站已經擋了。
"""
from __future__ import annotations

import atexit
import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from langchain_core.tools import tool

import guard
import tools as toolbox
import web

DEV_SUFFIXES = (".localhost", ".test")
DEV_HOSTS = {"localhost", "127.0.0.1", "::1"}
LOAD_TIMEOUT = 15     # 秒:等頁面載入
SETTLE = 0.4          # 秒:動作之後等畫面更新(前端框架重畫)
MAX_ITEMS = 150       # 一次最多列幾個可以操作的元素
MAX_TEXT = 6000       # 畫面文字最多幾個字
DENIED = "使用者沒有允許打開這個網址。請說明你想測什麼,或請使用者自己打開。"


def find_browser() -> str | None:
    """電腦上的 Edge(Windows 都有)或 Chrome。桌面版開視窗也用這個。"""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("chromium"),
    ]
    return next((c for c in candidates if c and Path(c).is_file()), None)


def is_dev_url(url: str) -> bool:
    """開發用的位址才給開:這個工具是拿來測自己的網站,不是上網(上網走 web.py,有自己的檢查)。"""
    if url == "about:blank":
        return True
    try:
        u = urlsplit(url)
        host = (u.hostname or "").lower()
    except ValueError:
        return False
    return u.scheme in ("http", "https") and (host in DEV_HOSTS or host.endswith(DEV_SUFFIXES))


class BrowserError(RuntimeError):
    pass


class Browser:
    """一個無頭瀏覽器行程(暫存設定檔),用到才啟動。"""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.profile: Path | None = None
        self.port, self.ws_path = 0, ""
        self.lock = threading.Lock()
        atexit.register(self.close)

    def _alive(self) -> bool:
        import httpx
        try:
            return bool(self.port) and httpx.get(f"http://127.0.0.1:{self.port}/json/version", timeout=3).is_success
        except httpx.HTTPError:
            return False

    def _start(self) -> None:
        """啟動並等它寫出 DevToolsActivePort。Edge 的啟動行程把工作交給另一個行程後就先結束(10/9 桌面版也遇過),
        所以不能用「行程結束了」判斷失敗,只看有沒有等到 port。"""
        exe = find_browser()
        if not exe:
            raise BrowserError("這台電腦找不到 Edge 或 Chrome")
        self.profile = Path(tempfile.mkdtemp(prefix="hsuan-browser-"))
        self.proc = subprocess.Popen(
            [exe, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={self.profile}", "--no-first-run",
             "--no-default-browser-check", "--disable-extensions", "--disable-sync", "--window-size=1280,900",
             "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        port_file, began = self.profile / "DevToolsActivePort", time.monotonic()
        while time.monotonic() - began < 20:
            try:
                lines = port_file.read_text().splitlines() if port_file.is_file() else []
            except OSError:
                lines = []
            if len(lines) >= 2:
                self.port, self.ws_path = int(lines[0]), lines[1]
                return
            time.sleep(0.1)
        self.close()
        raise BrowserError("瀏覽器沒有啟動")

    def new_tab(self) -> "Tab":
        import httpx
        with self.lock:
            if not self._alive():
                self.close()
                self._start()
            target = httpx.put(f"http://127.0.0.1:{self.port}/json/new?about:blank", timeout=10).json()
        return Tab(self, target)

    def close(self) -> None:
        """請瀏覽器自己關掉(真正的瀏覽器行程不是我們啟動的那個),再刪掉暫存設定檔。"""
        port, path, self.port = self.port, self.ws_path, 0
        if port and path:
            try:
                from websockets.sync.client import connect
                with connect(f"ws://127.0.0.1:{port}{path}", open_timeout=3) as ws:
                    ws.send(json.dumps({"id": 1, "method": "Browser.close"}))
                    ws.recv(timeout=3)
            except Exception:
                pass
        proc, self.proc = self.proc, None
        if proc and proc.poll() is None:
            proc.kill()
        if self.profile:
            for _ in range(20):  # 瀏覽器結束後才放開檔案
                shutil.rmtree(self.profile, ignore_errors=True)  # 暫存設定檔:這次的 cookie、快取都不留
                if not self.profile.exists():
                    break
                time.sleep(0.1)
            self.profile = None


_BROWSER = Browser()

# 列出畫面上可以操作的元素,每個標上編號(data-hsuan-ref),模型用編號點、輸入;密碼欄不回傳內容
SNAPSHOT_JS = r"""(() => {
  document.querySelectorAll('[data-hsuan-ref]').forEach((e) => e.removeAttribute('data-hsuan-ref'));
  const visible = (e) => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const clean = (t) => String(t || '').replace(/\s+/g, ' ').trim().slice(0, 80);
  const items = []; let n = 0;
  const sel = 'a[href],button,input,select,textarea,summary,[role=button],[role=link],[role=checkbox],[role=tab],[role=menuitem],[onclick],[contenteditable=true]';
  for (const e of document.querySelectorAll(sel)) {
    if (!visible(e) || (e.type === 'hidden')) continue;
    if (++n > %(max)d) break;
    e.setAttribute('data-hsuan-ref', n);
    let kind = e.tagName.toLowerCase();
    if (kind === 'input') kind += '(' + (e.type || 'text') + ')';
    const name = clean(e.getAttribute('aria-label') || (kind.startsWith('input') || kind === 'textarea' ? '' : e.innerText)
                       || e.placeholder || e.title || e.name || e.id);
    const extra = [];
    if (['checkbox', 'radio'].includes(e.type)) extra.push(e.checked ? '已勾選' : '沒勾');
    else if (e.type === 'password') extra.push(e.value ? '已填' : '空的');
    else if ('value' in e && e.value && kind !== 'button') extra.push('值=' + clean(e.value));
    if (e.placeholder && name !== clean(e.placeholder)) extra.push('提示=' + clean(e.placeholder));
    if (e.disabled) extra.push('停用');
    if (e.tagName === 'A') extra.push('→ ' + (e.getAttribute('href') || '').slice(0, 80));
    items.push('[' + n + '] ' + kind + ' 「' + name + '」' + (extra.length ? '(' + extra.join(',') + ')' : ''));
  }
  const text = document.body ? document.body.innerText.replace(/\n{3,}/g, '\n\n').trim() : '';
  return JSON.stringify({ title: document.title, url: location.href, items, more: n > %(max)d, text: text.slice(0, %(text)d) });
})()""" % {"max": MAX_ITEMS, "text": MAX_TEXT}

KEYS = {"Enter": 13, "Tab": 9, "Escape": 27, "Backspace": 8, "ArrowDown": 40, "ArrowUp": 38, "ArrowLeft": 37,
        "ArrowRight": 39, "Space": 32}


class Tab:
    """一個分頁(一段對話一個):CDP 的 websocket 連線,順便收主控台的錯誤。"""

    def __init__(self, browser: Browser, target: dict):
        from websockets.sync.client import connect
        self.browser, self.id = browser, target["id"]
        self.ws = connect(target["webSocketDebuggerUrl"], max_size=50_000_000, open_timeout=10)
        self.n = 0
        self.errors: list[str] = []  # 這次動作之後主控台的錯誤(給模型看:測網站最常要的就是這個)
        self.lock = threading.Lock()
        for domain in ("Page.enable", "Runtime.enable", "Log.enable"):
            self.cdp(domain)

    def cdp(self, method: str, timeout: float = 20, **params) -> dict:
        self.n += 1
        rid = self.n
        self.ws.send(json.dumps({"id": rid, "method": method, "params": params}))
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise BrowserError(f"瀏覽器超過 {timeout:.0f} 秒沒有回應({method})")
            msg = json.loads(self.ws.recv(timeout=left))
            if msg.get("id") == rid:
                if "error" in msg:
                    raise BrowserError(msg["error"].get("message", str(msg["error"])))
                return msg.get("result") or {}
            self._event(msg)

    def _event(self, msg: dict) -> None:
        method, p = msg.get("method"), msg.get("params") or {}
        if method == "Runtime.exceptionThrown":
            d = p.get("exceptionDetails") or {}
            self.errors.append(((d.get("exception") or {}).get("description") or d.get("text") or "例外")[:300])
        elif method == "Runtime.consoleAPICalled" and p.get("type") in ("error", "assert"):
            self.errors.append(" ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", []))[:300])
        elif method == "Log.entryAdded" and (p.get("entry") or {}).get("level") == "error":
            e = p["entry"]
            self.errors.append(f"{e.get('text', '')} {e.get('url', '')}".strip()[:300])

    def js(self, expression: str):
        r = self.cdp("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if r.get("exceptionDetails"):
            raise BrowserError(r["exceptionDetails"].get("text", "JavaScript 錯誤"))
        return (r.get("result") or {}).get("value")

    def wait_loaded(self) -> None:
        began = time.monotonic()
        while time.monotonic() - began < LOAD_TIMEOUT:
            try:
                if self.js("document.readyState") == "complete":
                    break
            except BrowserError:
                pass  # 換頁中,執行環境還沒好
            time.sleep(0.15)
        time.sleep(SETTLE)

    def goto(self, url: str) -> None:
        r = self.cdp("Page.navigate", url=url)
        if r.get("errorText"):
            raise BrowserError(f"打不開 {url}:{r['errorText']}(網站有在跑嗎?)")
        time.sleep(0.15)  # 不然 readyState 可能還是上一頁的 complete
        self.wait_loaded()

    def _center(self, ref: int) -> tuple[float, float]:
        box = self.js(f"""(() => {{ const e = document.querySelector('[data-hsuan-ref="{int(ref)}"]');
            if (!e) return null; e.scrollIntoView({{block: 'center', inline: 'center'}});
            const r = e.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; }})()""")
        if not box:
            raise BrowserError(f"找不到編號 {ref}(畫面可能變了,請看最新的清單)")
        return box[0], box[1]

    def click(self, ref: int) -> None:
        """真的用滑鼠點(不是呼叫 element.click()):前端框架的事件、遮罩擋住的情況都跟真人一樣。"""
        x, y = self._center(ref)
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            self.cdp("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)
        time.sleep(SETTLE)
        self.wait_loaded()

    def type(self, ref: int, text: str) -> None:
        """點進欄位、選取原本的內容、打字取代(跟真人一樣觸發 input 事件,React 之類的也收得到)。"""
        kind = self.js(f"""(() => {{ const e = document.querySelector('[data-hsuan-ref="{int(ref)}"]');
            if (!e) return null; e.focus(); if (e.select) e.select(); return e.tagName; }})()""")
        if kind is None:
            raise BrowserError(f"找不到編號 {ref}(畫面可能變了,請看最新的清單)")
        if kind == "SELECT":  # 下拉選單:照選項文字或值選
            ok = self.js(f"""(() => {{ const e = document.querySelector('[data-hsuan-ref="{int(ref)}"]'), t = {json.dumps(text)};
                const o = [...e.options].find((o) => o.value === t || o.text.trim() === t); if (!o) return false;
                e.value = o.value; e.dispatchEvent(new Event('change', {{bubbles: true}})); return true; }})()""")
            if not ok:
                raise BrowserError(f"編號 {ref} 的選項裡沒有「{text}」")
            return
        self.cdp("Input.insertText", text=text)
        time.sleep(0.1)

    def press(self, key: str) -> None:
        code = KEYS.get(key)
        if code is None:
            raise BrowserError(f"不支援的按鍵 {key},可以用:{'、'.join(KEYS)}")
        key = " " if key == "Space" else key
        for kind in ("keyDown", "keyUp"):
            self.cdp("Input.dispatchKeyEvent", type=kind, key=key, windowsVirtualKeyCode=code, nativeVirtualKeyCode=code,
                     **({"text": "\r"} if key == "Enter" and kind == "keyDown" else {}))
        time.sleep(SETTLE)
        self.wait_loaded()

    def back(self) -> None:
        self.js("history.back()")
        time.sleep(SETTLE)
        self.wait_loaded()

    def screenshot(self) -> bytes:
        return base64.b64decode(self.cdp("Page.captureScreenshot", format="jpeg", quality=70)["data"])

    def url(self) -> str:
        return self.js("location.href") or ""

    def snapshot(self) -> str:
        """給模型看的畫面:網址、標題、可以操作的元素(編號)、畫面文字、這次動作之後的主控台錯誤。"""
        s = json.loads(self.js(SNAPSHOT_JS))
        errors, self.errors = self.errors[-10:], []
        parts = [f"網址:{s['url']}", f"標題:{s['title'] or '(沒有標題)'}",
                 "可以操作的元素(用 ref 編號點、輸入):\n" + ("\n".join(s["items"]) or "(沒有)")
                 + ("\n(還有更多,沒有列出)" if s.get("more") else ""),
                 "畫面上的文字:\n" + (s["text"] or "(沒有文字)")]
        if errors:
            parts.append("主控台錯誤:\n" + "\n".join(f"- {e}" for e in errors))
        return "\n\n".join(parts)

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


def _tab() -> Tab:
    wb = toolbox.bench()
    if wb.browser is None:
        wb.browser = _BROWSER.new_tab()
    return wb.browser


@tool
def browser(action: Literal["open", "click", "type", "press", "back", "look", "close"], url: str = "", ref: int = 0,
            text: str = "", submit: bool = False) -> str:
    """測試使用者自己開發中的網站(只能開 localhost、127.0.0.1、*.localhost、*.test)。
    open(url);click(ref);type(ref, text, submit=true 再按 Enter);press(text=Enter/Tab/Escape…);back;
    look(text=要看什麼)截圖請看圖模型描述(看排版、顏色才用);close。
    回傳網址、可操作的元素(ref 編號)、畫面文字、主控台錯誤。只用測試資料,不輸入真的密碼或個資。"""
    wb = toolbox.bench()
    if action == "close":
        if wb.browser:
            wb.browser.close()
            wb.browser = None
        return "已關掉瀏覽器分頁"
    if action == "open":
        if not is_dev_url(url):
            return (f"錯誤:{url or '(沒給網址)'} 不是開發用的位址。這個工具只能開 localhost、127.0.0.1、*.localhost、*.test"
                    "(測使用者自己的網站);公開網頁請用 fetch_url")
        what = "用瀏覽器打開" + ("" if guard.COMPANY_OK else "(畫面內容會送到 NVIDIA 試用平台,不要開公司的系統)")
        if not web._confirm(what, url):
            return DENIED
    elif wb.browser is None:
        return "錯誤:還沒打開網頁,請先用 open"
    try:
        tab = _tab()
        with tab.lock:
            if action == "open":
                tab.goto(url)
            elif action == "click":
                tab.click(ref)
            elif action == "type":
                tab.type(ref, text)
                if submit:
                    tab.press("Enter")
            elif action == "press":
                tab.press(text or "Enter")
            elif action == "back":
                tab.back()
            if not is_dev_url(now := tab.url()):  # 轉址或點了連結跑到別的網站:停下來
                tab.goto("about:blank")
                return f"網頁被導到 {urlsplit(now).hostname}(不是開發用的位址),已經停下來、回到空白頁。"
            _keep_shot(wb, tab, now)
            return _look(tab, text) if action == "look" else tab.snapshot()
    except BrowserError as e:
        return f"錯誤:{e}"
    except Exception as e:  # 瀏覽器被關掉、連線斷了:下次 open 重開一個
        if wb.browser:
            wb.browser.close()
        wb.browser = None
        return f"錯誤:瀏覽器斷線了({type(e).__name__}),請重新 open"


def _keep_shot(wb: toolbox.Workbench, tab: Tab, url: str) -> None:
    """每個動作後留一張截圖,網頁右邊的「瀏覽器」分頁讓使用者看 HSUAN 現在看到什麼(10/24)。截不到不影響工具。"""
    try:
        wb.browser_shot = (time.time(), url, tab.screenshot())
    except Exception:
        pass


def _look(tab: Tab, focus: str) -> str:
    """截圖請看圖模型描述(跟使用者附的圖片同一套:用量算進這段對話)。"""
    import agent
    session = agent._billing.get()
    if session is None:
        return "錯誤:現在沒有對話在進行"
    url = "data:image/jpeg;base64," + base64.b64encode(tab.screenshot()).decode()
    question = focus or "這個網頁畫面現在長什麼樣子?有沒有排版錯亂、文字被切掉、錯誤訊息?"
    return "\n".join(session._describe_images(question, [("瀏覽器畫面", url)])) + "\n\n" + tab.snapshot()
