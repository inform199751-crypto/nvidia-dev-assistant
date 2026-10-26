"""跨對話的記憶(10/26,像 ChatGPT 的記憶):使用者的長期偏好和背景,每段對話都放進系統提示。

只記使用者自己說的、以後都適用的事(回答風格、常用的語言 / 框架、正在做的事),一條一句話。
存在這個人自己的資料夾(本機是 OUTPUT_DIR/state/memory.json,桌面版、VS Code 外掛、CLI 共用一份),
網頁「設定 → 記憶」看得到、加得了、刪得掉、可以整個關掉。看起來像密碼或金鑰的不記(會跟著每一題送給模型)。
訪客沒有記憶(用的是主人的電腦和金鑰)。
"""
from __future__ import annotations

import json
import re
import secrets
import threading
import time
from pathlib import Path

from langchain_core.tools import tool

import tools as toolbox
from token_guard import estimate_tokens

MAX_ITEMS = 50          # 最多記幾條(每一步都會送,太多就吃 token)
MAX_CHARS = 200         # 一條最多幾個字
MAX_PROMPT_TOKENS = 600  # 放進系統提示的上限:超過就只放最新的
# 金鑰、密碼的樣子:記憶會跟著每一題送給模型,不能記
SECRET_LIKE = re.compile(r"nvapi-|sk-[A-Za-z0-9]{8}|ghp_[A-Za-z0-9]|AKIA[0-9A-Z]{12}|-----BEGIN|tvly-|"
                         r"(密碼|password|passwd|api[ _-]?key|token)\s*(是|[:：=])", re.I)


class Memory:
    def __init__(self, path: Path | None):
        self.path = path  # None = 只記在記憶體(測試)
        self.lock = threading.Lock()
        self.items: list[dict] = self._read()

    def _read(self) -> list[dict]:
        if not self.path:
            return getattr(self, "items", [])
        try:
            items = json.loads(self.path.read_text(encoding="utf-8")).get("items", [])
            return [it for it in items if isinstance(it, dict) and it.get("id") and it.get("text")]
        except (OSError, ValueError, AttributeError):
            return []

    def _write(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"items": self.items}, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def list(self) -> list[dict]:
        """每次重讀:桌面版和 VS Code 外掛是不同行程,另一邊記的也要看得到。"""
        with self.lock:
            self.items = self._read()
            return list(self.items)

    def add(self, text: str, source: str = "") -> dict:
        """加一條;一樣的(不分大小寫、空白)就回傳原本那條。不合格丟 ValueError(訊息會回給模型或畫面)。"""
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("要記的內容是空的")
        if len(text) > MAX_CHARS:
            raise ValueError(f"一條最多 {MAX_CHARS} 個字,請寫短一點(一條記一件事)")
        if SECRET_LIKE.search(text):
            raise ValueError("看起來是密碼或金鑰,不能記(記憶會跟著每一題送給模型)")
        with self.lock:
            self.items = self._read()
            same = next((it for it in self.items if it["text"].casefold() == text.casefold()), None)
            if same:
                return same
            if len(self.items) >= MAX_ITEMS:
                raise ValueError(f"記憶已經有 {MAX_ITEMS} 條了,請使用者先到「設定 → 記憶」刪掉一些")
            item = {"id": secrets.token_hex(4), "text": text, "at": round(time.time()), "source": source}
            self.items.append(item)
            self._write()
            return item

    def remove(self, item_id: str) -> bool:
        with self.lock:
            self.items = self._read()
            kept = [it for it in self.items if it["id"] != item_id]
            if len(kept) == len(self.items):
                return False
            self.items = kept
            self._write()
            return True

    def clear(self) -> None:
        with self.lock:
            self.items = []
            self._write()

    def prompt_block(self) -> str:
        """放進系統提示的那段:最新的放得下就放,舊的超過上限就不放(畫面上還看得到)。"""
        lines, used = [], 0
        for it in reversed(self.list()):
            cost = estimate_tokens(it["text"]) + 2
            if used + cost > MAX_PROMPT_TOKENS:
                break
            lines.append(f"- {it['text']}")
            used += cost
        return "\n".join(reversed(lines))


def memory_line() -> str:
    """記憶打開時加進系統提示:記得的事 + 什麼時候該用 save_memory。沒有記憶(關掉、訪客)就什麼都不加。"""
    mem = toolbox.bench().memory
    if mem is None:
        return ""
    known = mem.prompt_block()
    return ("\n\n## Memory across conversations\n"
            "When the user tells you a lasting preference or fact about themselves or their work, or asks you to remember "
            "something, call save_memory with one short sentence (never passwords, keys or personal data). "
            "Use what you remember to tailor answers without announcing it."
            + (f"\nWhat you remember about the user (they can view and delete these):\n{known}" if known else ""))


@tool
def save_memory(fact: str) -> str:
    """記住使用者以後每段對話都適用的事(偏好、背景、常用的工具或寫法),新對話會自動帶上。
    使用者說「記住…」「以後都…」時用;一次一件事、寫成一句話;不要存密碼、金鑰、個資或程式碼細節。"""
    mem = toolbox.bench().memory
    if mem is None:
        return "錯誤:這裡沒有開記憶功能"
    try:
        item = mem.add(fact)
    except ValueError as e:
        return f"錯誤:{e}"
    return f"已記住:{item['text']}(使用者可以在「設定 → 記憶」看到、刪掉)"
