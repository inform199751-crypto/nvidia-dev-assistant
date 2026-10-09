"""桌面版小助理:雙擊桌面的「小助理」就打開一個獨立視窗(像 Claude 桌面版),不用開終端機。

    .venv/Scripts/python desktop_app.py --install    → 在桌面建立「小助理」捷徑(含圖示)
    (之後雙擊桌面捷徑就好;捷徑跑的是 pythonw,不會跳出黑色的主控台視窗)

背後:在這個行程裡啟動 webui 伺服器(port 8780,跟自己開的 webui.py 錯開),
再用 Edge(沒有就 Chrome)的 app 模式開一個沒有網址列、沒有分頁的視窗。
視窗關掉就把伺服器一起關掉;已經開著時再雙擊,只會多開一個視窗,不會重複啟動。
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import webbrowser
import zlib
from pathlib import Path

HERE = Path(__file__).parent
PORT = 8780
ICON = HERE / "assets" / "assistant.ico"
PROFILE = HERE / "output" / ".app-profile"  # 獨立的瀏覽器設定資料夾:視窗都關掉時,這個瀏覽器行程才會結束
LOG = HERE / "output" / "desktop_app.log"   # pythonw 沒有主控台,出錯只看得到這個檔案
# 判斷「視窗都關了」:看有沒有頁面連著事件串流,不看瀏覽器行程(Edge 的啟動行程把視窗交給別的行程後就先結束了)。
# 用「連續幾次檢查都沒人」而不是看時鐘:電腦睡眠醒來時頁面要幾秒才重連,不能一醒來就關掉
POLL = 5             # 秒
FIRST_WINDOW = 12    # 啟動後最多等 12 次(60 秒)讓第一個視窗連上
CLOSED_AFTER = 4     # 之後連續 4 次(20 秒)都沒有視窗連著,才當作全部關掉(重新整理、伺服器換 token 都在這之內)

# ---------- 圖示:純 Python 畫,不用另外裝套件 ----------

GREEN, WHITE = (16, 163, 127), (255, 255, 255)


def _rounded(x: float, y: float, x0: float, y0: float, x1: float, y1: float, r: float) -> bool:
    cx, cy = min(max(x, x0 + r), x1 - r), min(max(y, y0 + r), y1 - r)
    return x0 <= x <= x1 and y0 <= y <= y1 and (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def _in_triangle(x, y, a, b, c) -> bool:
    def side(p, q):
        return (x - q[0]) * (p[1] - q[1]) - (p[0] - q[0]) * (y - q[1])
    d1, d2, d3 = side(a, b), side(b, c), side(c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def _color(x: float, y: float):
    """座標 0~1:綠色圓角方塊,中間白色對話泡泡,泡泡裡三個綠點。回傳 (r,g,b) 或 None(透明)。"""
    if not _rounded(x, y, 0.02, 0.02, 0.98, 0.98, 0.22):
        return None
    for dx in (0.36, 0.50, 0.64):
        if (x - dx) ** 2 + (y - 0.45) ** 2 <= 0.048 ** 2:
            return GREEN
    if _rounded(x, y, 0.18, 0.24, 0.82, 0.66, 0.13) or _in_triangle(x, y, (0.30, 0.62), (0.27, 0.80), (0.46, 0.64)):
        return WHITE
    return GREEN


def _png(size: int) -> bytes:
    """4x4 超取樣抗鋸齒,輸出 RGBA PNG。"""
    rows, n = [], 4
    for py in range(size):
        row = bytearray(b"\x00")
        for px in range(size):
            r = g = b = hits = 0
            for sy in range(n):
                for sx in range(n):
                    c = _color((px + (sx + 0.5) / n) / size, (py + (sy + 0.5) / n) / size)
                    if c:
                        r, g, b, hits = r + c[0], g + c[1], b + c[2], hits + 1
            row += bytes((r // hits, g // hits, b // hits, 255 * hits // (n * n))) if hits else b"\x00\x00\x00\x00"
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def make_icon(path: Path = ICON, sizes=(16, 24, 32, 48, 64, 256)) -> Path:
    """多尺寸 .ico(每個尺寸存成 PNG,Windows Vista 之後都支援)。"""
    images = [_png(s) for s in sizes]
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset, entries = 6 + 16 * len(sizes), b""
    for s, data in zip(sizes, images):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + entries + b"".join(images))
    return path


# ---------- 桌面捷徑 ----------

def install_shortcut() -> Path:
    """在桌面建立「小助理」捷徑:pythonw 跑這個檔案(沒有主控台視窗),圖示用 assets/assistant.ico。"""
    if not ICON.is_file():
        make_icon()
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        raise SystemExit(f"找不到 {pythonw},請用 .venv\\Scripts\\python.exe 執行這個指令")
    ps = (
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8; $d = [Environment]::GetFolderPath('Desktop'); "
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d '小助理.lnk')); "
        f"$s.TargetPath = '{pythonw}'; $s.Arguments = '\"{Path(__file__).resolve()}\"'; "
        f"$s.WorkingDirectory = '{HERE.resolve()}'; $s.IconLocation = '{ICON.resolve()},0'; "
        "$s.Description = '小助理(對內研發助理)'; $s.Save(); Join-Path $d '小助理.lnk'"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise SystemExit(f"建立捷徑失敗:{out.stderr.strip()}")
    return Path(out.stdout.strip())


# ---------- 開視窗 ----------

def find_browser() -> str | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        shutil.which("chrome"),
    ]
    return next((c for c in candidates if c and Path(c).is_file()), None)


def open_window(url: str, headless: bool = False) -> subprocess.Popen | None:
    """app 模式:沒有網址列和分頁的獨立視窗。找不到 Edge / Chrome 就退回一般瀏覽器分頁。"""
    browser = find_browser()
    if not browser:
        webbrowser.open(url)
        return None
    profile = PROFILE.with_name(".app-profile-test") if headless else PROFILE
    profile.mkdir(parents=True, exist_ok=True)
    common = [f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check", "--disable-background-mode"]
    if headless:  # 測試用:不真的跳出視窗(無頭模式不吃 --app,網址直接給)
        return subprocess.Popen([browser, "--headless=new", *common, url])
    return subprocess.Popen([browser, f"--app={url}", "--window-size=1180,820", *common])


def run(headless: bool = False) -> None:
    import agent
    import web
    import webui

    url = f"http://127.0.0.1:{PORT}/"
    registry, quota, tools, allow_web = agent.prepare()
    hub = webui.Hub(lambda on_event: agent.AgentSession(registry, quota, tools=tools, on_event=on_event),
                    status=agent.web_status(allow_web),
                    model=registry.spec(registry.routing["default_role"]).model.split("/")[-1])
    web.CONFIRM = hub.confirm
    try:
        server = webui.serve(hub, PORT)
    except OSError:  # 已經有一個桌面版在跑:只多開一個視窗
        logging.info("小助理已經在跑了,只開視窗")
        open_window(url, headless)
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logging.info("小助理已啟動:%s(工作區 %s)", url, os.getenv("AGENT_WORKSPACE"))
    open_window(url, headless)
    seen, empty = False, 0
    while True:
        time.sleep(POLL)
        if hub.listeners:  # 有視窗連著
            seen, empty = True, 0
            continue
        empty += 1
        if empty >= (CLOSED_AFTER if seen else FIRST_WINDOW):
            break
    server.shutdown()
    logging.info("視窗都關了,小助理結束" if seen else "視窗一直沒有連上,小助理結束")


def main() -> None:
    ap = argparse.ArgumentParser(description="桌面版小助理")
    ap.add_argument("--install", action="store_true", help="在桌面建立「小助理」捷徑")
    ap.add_argument("--icon", action="store_true", help="重新產生 assets/assistant.ico")
    args = ap.parse_args()
    if args.icon:
        print(f"已產生 {make_icon()}")
        return
    if args.install:
        print(f"已在桌面建立捷徑:{install_shortcut()}")
        return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:  # pythonw 沒有主控台:有東西 print 的話會直接當掉,改寫進 log
        sys.stdout = sys.stderr = open(LOG, "a", encoding="utf-8", buffering=1)
    logging.basicConfig(filename=LOG, level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    os.chdir(HERE)
    try:
        run(headless=os.getenv("DESKTOP_APP_HEADLESS") == "1")
    except Exception:
        logging.exception("小助理啟動失敗")
        raise


if __name__ == "__main__":
    main()
