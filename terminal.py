"""終端機面板(10/23 使用者要的,像 Claude 桌面版的 Terminal):使用者自己在這台電腦上跑指令,不是模型的工具。

簡單版:一次一行,每行開一個 PowerShell(Windows;其他系統用 bash)在這段對話的專案資料夾裡跑,輸出逐行收進來,
網頁每隔一下來拿新的。不能跑要一直輸入的程式(stdin 是空的,python 互動模式之類會直接結束)。`cd` 自己處理,換了會記住。

安全:
- 只有這台電腦能用(webui 檢查:不是本機、訪客、多人模式一律 403)—— 不然手機 / 訪客就能在主人電腦上執行指令。
- 環境變數拿掉 .env 讀進來的和名字像金鑰的(跟 MCP 連接器同一套,mcp_client.child_env):
  不然 `env` 會把 NVIDIA 金鑰印在畫面上,一按「交給 HSUAN」就送給模型了。
- 模型碰不到:輸出只有使用者按「交給 HSUAN」、自己看過再送出,才會進對話。
"""
from __future__ import annotations

import base64
import locale
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import mcp_client

MAX_LINES = 3000   # 畫面上最多留幾行(太多就丟最前面的)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")  # 顏色控制碼(Python 3.13 的錯誤訊息、git、npm 會上色)


def _decode(raw: bytes) -> str:
    """一行輸出 → 文字:大多是 UTF-8(PowerShell 設好了、python 用 PYTHONIOENCODING),舊的 Windows 工具是系統編碼(cp950)。"""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode(locale.getpreferredencoding(False) or "cp950", errors="replace")
    return ANSI.sub("", text)


def shell_argv(command: str) -> list[str]:
    """Windows:PowerShell 跑 -EncodedCommand(避開所有引號跳脫的問題),先把輸出編碼設成 UTF-8;其他系統用 bash。"""
    if os.name != "nt":
        return [shutil.which("bash") or "/bin/sh", "-c", command]
    # 錯誤也併進一般輸出、轉成文字:不然 -EncodedCommand 加上轉向,錯誤會變成一大段 CLIXML;原生程式的結束代碼照傳
    # 錯誤紀錄只留訊息本身:PowerShell 5.1 會把原生程式的每一行 stderr 包成 NativeCommandError(多出「位於 線路…」)
    script = ("[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;$OutputEncoding=[System.Text.Encoding]::UTF8;"
              "$ProgressPreference='SilentlyContinue';$global:LASTEXITCODE=0;"
              f"& {{\n{command}\n}} 2>&1 | ForEach-Object {{ if ($_ -is [System.Management.Automation.ErrorRecord]) "
              "{ $_.ToString() } else { $_ } } | Out-String -Stream -Width 200;exit $LASTEXITCODE")
    return [shutil.which("powershell") or "powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand",
            base64.b64encode(script.encode("utf-16-le")).decode()]


class Terminal:
    """一段對話的終端機:目前的資料夾、輸出(每行一個絕對編號,網頁用 since 拿新的)、正在跑的那個指令。"""

    def __init__(self, cwd: Path):
        self.cwd = Path(cwd)
        self.lines: deque[str] = deque(maxlen=MAX_LINES)
        self.count = 0           # 到現在總共寫過幾行(包含已經丟掉的)
        self.last_start = 0      # 最近一個指令從第幾行開始(「交給 HSUAN」拿這之後的)
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _add(self, text: str) -> None:
        with self.lock:
            for line in text.splitlines() or [""]:
                self.lines.append(line)
                self.count += 1

    def prompt(self) -> str:
        return f"PS {self.cwd}>" if os.name == "nt" else f"{self.cwd}$"

    def run(self, command: str) -> str:
        """跑一行指令;回傳錯誤訊息,空字串 = 開始跑了(或 cd、清除這種當場做完的)。"""
        command = command.strip()
        if not command:
            return ""
        if self.running:
            return "上一個指令還在跑,等它結束或按「停止」"
        if command.lower() in ("cls", "clear", "clear-host"):
            self.clear()
            return ""
        with self.lock:
            self.last_start = self.count
        self._add(f"{self.prompt()} {command}")
        verb, _, arg = command.partition(" ")
        if verb.lower() in ("cd", "chdir", "set-location", "sl"):  # 每行都是新的行程,cd 要自己記
            self._cd(arg.strip().strip('"').strip("'"))
            return ""
        env = mcp_client.child_env({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "NO_COLOR": "1", "PYTHON_COLORS": "0"})
        try:
            self.proc = subprocess.Popen(shell_argv(command), cwd=self.cwd, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, creationflags=NO_WINDOW)
        except OSError as e:
            self._add(f"(執行不了:{e})")
            return ""
        threading.Thread(target=self._pump, args=(self.proc, time.monotonic()), daemon=True).start()
        return ""

    def _cd(self, arg: str) -> None:
        if not arg:
            self._add(str(self.cwd))
            return
        target = Path(arg).expanduser()
        target = (target if target.is_absolute() else self.cwd / target).resolve()
        if target.is_dir():
            self.cwd = target
        else:
            self._add(f"(找不到資料夾:{target})")

    def _pump(self, proc: subprocess.Popen, began: float) -> None:
        for raw in proc.stdout:
            self._add(_decode(raw.rstrip(b"\r\n")))
        code = proc.wait()
        self._add(f"(結束,代碼 {code},{time.monotonic() - began:.1f} 秒)")

    def stop(self) -> bool:
        """「停止」:整個行程樹一起關(PowerShell 底下開的 python、node 之類也要關)。"""
        proc = self.proc
        if not (proc and proc.poll() is None):
            return False
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, creationflags=NO_WINDOW)
        else:
            proc.kill()
        return True

    def clear(self) -> None:
        with self.lock:
            self.lines.clear()
            self.last_start = self.count

    def read(self, since: int = 0) -> dict:
        """since 之後的輸出(網頁一直來拿);base 之前的已經丟掉了。"""
        with self.lock:
            base = self.count - len(self.lines)
            start = max(since, base)
            lines = list(self.lines)[start - base:]
            last = [ln for i, ln in enumerate(self.lines, base) if i >= self.last_start]
        return {"lines": lines, "next": self.count, "from": start, "running": self.running, "cwd": str(self.cwd),
                "prompt": self.prompt(), "last": "\n".join(last)}
