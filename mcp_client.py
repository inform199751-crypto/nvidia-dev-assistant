"""MCP 連接器(10/15):HSUAN 當 MCP client,把外部 MCP server 提供的工具接給模型(像 Claude 的「連接器」)。

只用標準函式庫自己寫 JSON-RPC(跟 qr.py 一樣不多裝套件):stdio 傳輸是一行一個 JSON 訊息,
用到的只有 initialize → notifications/initialized → tools/list → tools/call(還有回應伺服器的 ping)。

安全:
- 連接器是會在這台電腦上執行的程式:只有這台電腦能新增(webui 檢查),設定存在 HSUAN 自己的資料夾。
- 啟動時不把 HSUAN 的金鑰傳下去:環境變數拿掉 .env 讀進來的、名字像金鑰的(NVIDIA_API_KEY、TAVILY_API_KEY…);
  連接器自己要的金鑰寫在它的 env。不然等於把 NVIDIA 金鑰交給第三方程式。
- 每次呼叫都先問使用者(跟上網一樣);回傳的是第三方的資料,先過安全模型(guard.web_content)。
- 試用平台上新增要先聲明不會碰到公司資料(guard.check_source)。
"""
from __future__ import annotations

import atexit
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeout
from pathlib import Path

from langchain_core.tools import StructuredTool

import guard
import web

PROTOCOL_VERSION = "2025-06-18"
START_TIMEOUT = 30       # 秒:啟動 + 握手(npx 第一次要下載,會比較久)
CALL_TIMEOUT = 60        # 秒:一次工具呼叫
MAX_TOOLS = 20           # 一個連接器最多接幾個工具:工具定義每一步都要送,太多很貴
MAX_DESCRIPTION = 300    # 工具說明最多幾個字(別人寫的,可能很長)
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|REDIS_URL", re.I)
DENIED = "使用者沒有允許這次使用連接器。請改用其他方法,或直接回答你已經知道的部分,並說明沒有查證。"


class McpError(RuntimeError):
    pass


def _dotenv_keys() -> set[str]:
    """.env 裡有哪些變數(只取名字):這些是 HSUAN 自己的設定和金鑰,不給連接器。"""
    try:
        from dotenv import dotenv_values
        return set(dotenv_values(Path(__file__).with_name(".env")))
    except Exception:
        return set()


def child_env(extra: dict | None) -> dict[str, str]:
    """連接器的環境變數:這台電腦的(PATH 等,程式要找得到)拿掉 HSUAN 的設定和像金鑰的,再加上連接器自己設定的。"""
    hidden = _dotenv_keys()
    env = {k: v for k, v in os.environ.items() if k not in hidden and not SECRET_ENV.search(k)}
    env.update({str(k): str(v) for k, v in (extra or {}).items()})
    return env


class McpServer:
    """一個 stdio MCP server 的連線:一個子行程,背景執行緒讀它的輸出,依 id 把回應交給等著的請求。"""

    def __init__(self, name: str, command: str, args: list | None = None, env: dict | None = None,
                 cwd: str | None = None):
        self.name, self.command, self.args, self.env, self.cwd = name, command, list(args or []), env or {}, cwd
        self.proc: subprocess.Popen | None = None
        self.tools: list[dict] = []
        self.info: dict = {}
        self.stderr: deque[str] = deque(maxlen=30)  # 最後幾行錯誤訊息,啟動失敗時告訴使用者
        self._pending: dict[int, Future] = {}
        self._next = 0
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        exe = shutil.which(self.command) or self.command
        argv = [exe, *self.args]
        if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):  # npx、uvx 在 Windows 是批次檔
            argv = ["cmd", "/c", *argv]
        try:
            self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         env=child_env(self.env), cwd=self.cwd or None,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))  # 桌面版不跳黑視窗
        except OSError as e:
            raise McpError(f"啟動不了「{self.command}」:{e}") from None
        self._err_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._err_thread.start()
        threading.Thread(target=self._read, daemon=True).start()
        try:
            init = self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                               "clientInfo": {"name": "HSUAN", "version": "1.0"}}, START_TIMEOUT)
            self.info = init.get("serverInfo") or {}
            self.notify("notifications/initialized")
            tools, cursor = [], None
            for _ in range(10):  # 分頁:最多讀 10 頁
                page = self.request("tools/list", {"cursor": cursor} if cursor else {}, START_TIMEOUT)
                tools += page.get("tools") or []
                if not (cursor := page.get("nextCursor")):
                    break
            self.tools = [t for t in tools if isinstance(t, dict) and t.get("name")]
        except Exception:
            self.close()
            raise

    def _send(self, msg: dict) -> None:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            if not self.alive:
                raise McpError(f"連接器「{self.name}」已經結束了{self._last_error()}")
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None, timeout: float = CALL_TIMEOUT) -> dict:
        with self._lock:
            self._next += 1
            rid, fut = self._next, Future()
            self._pending[rid] = fut
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            return fut.result(timeout)
        except FutureTimeout:
            raise McpError(f"連接器「{self.name}」超過 {timeout:.0f} 秒沒有回應({method})") from None
        finally:
            with self._lock:
                self._pending.pop(rid, None)

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def call(self, tool: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": tool, "arguments": arguments}, CALL_TIMEOUT)

    def _read(self) -> None:
        """讀伺服器的輸出:回應交給等著的請求;伺服器問我們的(ping 以外都不支援)回錯誤;通知(log、進度)不理。"""
        proc = self.proc
        for raw in proc.stdout:
            try:
                msg = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue  # 不是 JSON(有些伺服器會把 log 印到 stdout):跳過
            if not isinstance(msg, dict):
                continue
            if "method" in msg and "id" in msg:  # 伺服器發給我們的請求
                reply = {"result": {}} if msg["method"] == "ping" else \
                    {"error": {"code": -32601, "message": f"HSUAN 不支援 {msg['method']}"}}
                try:
                    self._send({"jsonrpc": "2.0", "id": msg["id"], **reply})
                except Exception:
                    pass
                continue
            with self._lock:
                fut = self._pending.get(msg.get("id"))
            if fut and not fut.done():
                if "error" in msg:
                    err = msg["error"] if isinstance(msg["error"], dict) else {"message": str(msg["error"])}
                    fut.set_exception(McpError(f"連接器「{self.name}」回報錯誤:{err.get('message') or err}"))
                else:
                    fut.set_result(msg.get("result") or {})
        try:  # stdout 關了 = 這個連接器不能用了:等行程結束(不結束就關掉),alive 才會是 False
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        self._err_thread.join(timeout=1)  # 等它的錯誤訊息讀完:結束的原因通常在最後一行
        with self._lock:  # 伺服器結束了:還在等的請求都失敗
            waiting, self._pending = list(self._pending.values()), {}
        for fut in waiting:
            if not fut.done():
                fut.set_exception(McpError(f"連接器「{self.name}」結束了{self._last_error()}"))

    def _read_stderr(self) -> None:
        for raw in self.proc.stderr:
            if line := raw.decode("utf-8", errors="replace").strip():
                self.stderr.append(line[:300])

    def _last_error(self) -> str:
        return f"(最後的訊息:{self.stderr[-1]})" if self.stderr else ""

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if not proc or proc.poll() is not None:
            return
        try:
            proc.stdin.close()  # MCP 的規定:stdin 關掉,伺服器就該自己結束
            proc.wait(timeout=2)
        except Exception:
            if os.name == "nt":  # npx → node 這種會留下子行程:整棵一起關
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                proc.kill()


def result_text(result: dict) -> str:
    """tools/call 的結果 → 給模型的文字。圖片不轉(主模型看不懂圖),只說有一張。"""
    parts = []
    for c in result.get("content") or []:
        kind = c.get("type") if isinstance(c, dict) else None
        if kind == "text":
            parts.append(str(c.get("text", "")))
        elif kind == "image":
            parts.append(f"[一張圖片({c.get('mimeType', '')}),沒有轉成文字]")
        elif kind == "resource":
            res = c.get("resource") or {}
            parts.append(str(res.get("text") or f"[檔案 {res.get('uri', '')}]"))
        elif kind == "resource_link":
            parts.append(f"[連結 {c.get('uri', '')}]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    text = "\n".join(parts).strip() or "(沒有內容)"
    return f"連接器回報錯誤:{text}" if result.get("isError") else text


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def make_tools(server: McpServer, label: str, prefix: str) -> list[StructuredTool]:
    """伺服器的工具 → HSUAN 的工具。名稱加上連接器的前綴(不同連接器可能有同名工具;API 只收英數字、_、-,最多 64 字)。"""
    out = []
    for t in server.tools[:MAX_TOOLS]:
        name = f"{prefix}__{slug(t['name']) or 'tool'}"[:64]
        desc = str(t.get("description") or t["name"]).strip()
        desc = f"[連接器 {label}] " + (desc[:MAX_DESCRIPTION] + "…" if len(desc) > MAX_DESCRIPTION else desc)
        schema = t.get("inputSchema") if isinstance(t.get("inputSchema"), dict) else {}
        schema = {"type": "object", "properties": {}, **schema}
        out.append(StructuredTool(name=name, description=desc, args_schema=schema,
                                  func=_runner(server, label, t["name"]), metadata={"mcp": label}))
    return out


def _runner(server: McpServer, label: str, tool: str):
    def run(**arguments) -> str:
        detail = json.dumps(arguments, ensure_ascii=False)
        if not web._confirm(f"使用連接器「{label}」的 {tool}", detail[:300] + ("…" if len(detail) > 300 else "")):
            return DENIED
        try:
            text = result_text(server.call(tool, arguments))
        except McpError as e:
            return f"錯誤:{e}"
        return guard.web_content(text, f"連接器「{label}」的回傳")  # 第三方的資料:藏著要 AI 照做的指令就擋下

    return run


def parse_config(text: str) -> list[dict]:
    """使用者貼上的設定 → [{name, command, args, env}]。收 Claude Desktop 的格式({"mcpServers": {名稱: {...}}}),
    MCP server 的說明文件大多給這種;也收單一個 {"name", "command", "args", "env"}。"""
    try:
        data = json.loads(text)
    except ValueError as e:
        raise ValueError(f"設定不是合法的 JSON:{e}") from None
    if not isinstance(data, dict):
        raise ValueError("設定要是一個 JSON 物件")
    servers = data.get("mcpServers", data.get("servers"))
    items = [{"name": k, **v} for k, v in servers.items() if isinstance(v, dict)] if isinstance(servers, dict) else [data]
    out = []
    for it in items:
        command, args, env = it.get("command"), it.get("args") or [], it.get("env") or {}
        if not isinstance(command, str) or not command.strip():
            raise ValueError("缺少 command(要執行的程式,例如 python、npx、uvx)")
        if it.get("url") or it.get("type") in ("http", "sse", "streamable-http"):
            raise ValueError("目前只支援在這台電腦上執行的連接器(command),還不支援網址型的")
        if not (isinstance(args, list) and isinstance(env, dict)):
            raise ValueError("args 要是清單、env 要是物件")
        out.append({"name": str(it.get("name") or Path(command).stem).strip()[:40], "command": command.strip(),
                    "args": [str(a) for a in args], "env": {str(k): str(v) for k, v in env.items()},
                    **({"cwd": str(it["cwd"])} if it.get("cwd") else {})})
    if not out:
        raise ValueError("設定裡沒有連接器")
    return out


class Connectors:
    """一個使用者(Desk)的連接器:設定存在 store/connectors.json,用到才啟動,同一個連接器所有對話共用一個行程。"""

    def __init__(self, store: Path | None):
        self.file = store / "connectors.json" if store else None
        self.items: list[dict] = []
        self.servers: dict[str, McpServer] = {}
        self.errors: dict[str, str] = {}
        self.lock = threading.RLock()
        atexit.register(self.close)  # HSUAN 結束時一起關掉(stdin 關了,連接器自己也會結束)
        if self.file and self.file.is_file():
            try:
                self.items = [x for x in json.loads(self.file.read_text(encoding="utf-8")) if isinstance(x, dict)]
            except (OSError, ValueError):
                self.items = []

    def _save(self) -> None:
        if self.file:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.file.with_name(self.file.name + ".tmp")
            tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.file)

    def add(self, config: str, public: bool = False) -> list[dict]:
        guard.check_source(public, "這個連接器")  # 試用平台:要先聲明它不會碰到公司資料
        new = parse_config(config)
        with self.lock:
            taken = {x["name"] for x in self.items}
            for it in new:
                base, n = it["name"] or "連接器", 2
                while it["name"] in taken:
                    it["name"], n = f"{base}-{n}", n + 1
                taken.add(it["name"])
                it.update(id=f"c{secrets.token_hex(3)}", enabled=True)
                if public and not guard.COMPANY_OK:
                    it["declared"] = "not-company"
                self.items.append(it)
            self._save()
        return new

    def item(self, cid: str) -> dict | None:
        return next((x for x in self.items if x["id"] == cid), None)

    def remove(self, cid: str) -> bool:
        with self.lock:
            if not (it := self.item(cid)):
                return False
            self.items.remove(it)
            self._save()
            server = self.servers.pop(cid, None)
        if server:
            server.close()
        return True

    def set_enabled(self, cid: str, on: bool) -> bool:
        with self.lock:
            if not (it := self.item(cid)):
                return False
            it["enabled"] = bool(on)
            self._save()
            server = None if on else self.servers.pop(cid, None)
        if server:
            server.close()
        return True

    def _server(self, it: dict) -> McpServer | None:
        """這個連接器的行程:還沒啟動(或掛了)就啟動;啟動失敗記下原因,回 None(其他連接器照常)。"""
        with self.lock:
            server = self.servers.get(it["id"])
            if server and server.alive:
                return server
            server = McpServer(it["name"], it["command"], it.get("args"), it.get("env"), it.get("cwd"))
            try:
                server.start()
            except Exception as e:
                self.errors[it["id"]] = str(e)
                self.servers.pop(it["id"], None)
                return None
            self.errors.pop(it["id"], None)
            self.servers[it["id"]] = server
            return server

    def restart(self, cid: str) -> dict | None:
        """「測試」按鈕:重新啟動、重新讀工具清單。"""
        with self.lock:
            if not (it := self.item(cid)):
                return None
            if server := self.servers.pop(cid, None):
                server.close()
            self._server(it)
        return next(x for x in self.listing() if x["id"] == cid)

    def tools(self) -> list[StructuredTool]:
        """打開中的連接器的工具(每一輪開始前拿一次:啟動、掛掉重開都在這裡)。"""
        out = []
        for it in [x for x in list(self.items) if x.get("enabled", True)]:
            if server := self._server(it):
                out += make_tools(server, it["name"], slug(it["name"]) or it["id"])
        return out

    def listing(self) -> list[dict]:
        """給設定頁看:不回傳 env 的值(可能是金鑰),只說有哪些變數。"""
        from token_guard import tool_tokens
        out = []
        for it in list(self.items):
            server = self.servers.get(it["id"])
            running = bool(server and server.alive)
            tools = make_tools(server, it["name"], slug(it["name"]) or it["id"]) if running else []
            out.append({"id": it["id"], "name": it["name"], "enabled": it.get("enabled", True),
                        "command": " ".join([it["command"], *it.get("args", [])]), "env": sorted(it.get("env", {})),
                        "state": "running" if running else "error" if it["id"] in self.errors else "stopped",
                        "error": self.errors.get(it["id"], ""), "server": (server.info if running else {}),
                        "tools": [{"name": t["name"], "description": str(t.get("description") or "")[:200]}
                                  for t in (server.tools if running else [])],
                        "tokens": tool_tokens(tools) if tools else 0})
        return out

    def close(self) -> None:
        with self.lock:
            servers, self.servers = list(self.servers.values()), {}
        for server in servers:
            server.close()


def demo_config() -> str:
    """「加入示範連接器」:專案裡的 demo_mcp/tickets_server.py(虛構的工單系統),用跑 HSUAN 的這個 Python 執行。"""
    script = Path(__file__).with_name("demo_mcp") / "tickets_server.py"
    return json.dumps({"mcpServers": {"tickets": {"command": sys.executable, "args": [str(script)]}}})
