"""排程任務(10/26,像 ChatGPT 的排程):時間到了自動開一段新對話,讓 HSUAN 去做(例如每天早上跑測試、整理失敗原因)。

跑在 HSUAN 自己的行程裡(webui / 桌面版):HSUAN 要開著才會跑;今天的時間已經過了、今天還沒跑過的,一打開就補跑一次。
排程沒有人在旁邊按確認:上網這類要確認的動作會被拒絕,改程式也只會留在草稿,要使用者自己看過按套用。
設定存在 store/schedules.json;桌面版和 VS Code 外掛可能同時開著(共用 store),用 claim() 的鎖檔保證一次只跑一份。
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from token_guard import TAIPEI

REPEATS = {"daily": (0, 1, 2, 3, 4, 5, 6), "weekdays": (0, 1, 2, 3, 4), "weekends": (5, 6)}  # datetime.weekday()
REPEAT_NAMES = {"daily": "每天", "weekdays": "平日(週一到週五)", "weekends": "週末"}
MAX_SCHEDULES = 20
MAX_PROMPT_CHARS = 2000
TICK_SECONDS = 30
LOCK_STALE_SECONDS = 60  # 鎖檔留太久(行程當掉)就當作沒鎖


def scheduled_at(item: dict, day: datetime) -> datetime:
    """這個排程在 day 那天應該跑的時間(台灣時間)。"""
    hh, mm = map(int, item["time"].split(":"))
    return day.astimezone(TAIPEI).replace(hour=hh, minute=mm, second=0, microsecond=0)


def is_due(item: dict, now: datetime) -> bool:
    """現在該不該跑:打開著、今天是要跑的日子、時間到了、今天這個時間之後還沒跑過(錯過的今天補跑一次)。"""
    if not item.get("enabled", True):
        return False
    now = now.astimezone(TAIPEI)
    if now.weekday() not in REPEATS.get(item.get("repeat", "daily"), ()):
        return False
    at = scheduled_at(item, now)
    return now >= at and (item.get("last_run") or 0) < at.timestamp()


def next_run(item: dict, now: datetime) -> float | None:
    """下次會跑的時間(畫面顯示用);關掉的回傳 None。"""
    if not item.get("enabled", True):
        return None
    now = now.astimezone(TAIPEI)
    for d in range(8):
        day = now + timedelta(days=d)
        at = scheduled_at(item, day)
        if day.weekday() in REPEATS.get(item.get("repeat", "daily"), ()) and (
                at > now or ((item.get("last_run") or 0) < at.timestamp() and d == 0)):
            return at.timestamp()
    return None


class Schedules:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def _read(self) -> list[dict]:
        try:
            items = json.loads(self.path.read_text(encoding="utf-8")).get("items", [])
            return [it for it in items if isinstance(it, dict) and it.get("id")]
        except (OSError, ValueError, AttributeError):
            return []

    def _write(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def list(self) -> list[dict]:
        with self.lock:
            return self._read()

    def save(self, item: dict) -> dict:
        """新增或修改(有 id 就是修改);不合格丟 ValueError。"""
        name = " ".join(str(item.get("name", "")).split())[:40]
        prompt = str(item.get("prompt", "")).strip()
        hhmm = str(item.get("time", ""))
        repeat = item.get("repeat", "daily")
        if not name or not prompt:
            raise ValueError("名稱和要做的事都要填")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"要做的事最多 {MAX_PROMPT_CHARS} 字")
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", hhmm):
            raise ValueError("時間要像 09:00")
        if repeat not in REPEATS:
            raise ValueError("重複只能是每天、平日或週末")
        fields = {"name": name, "prompt": prompt, "time": hhmm, "repeat": repeat, "project": item.get("project"),
                  "kind": "chat" if item.get("kind") == "chat" else "code", "enabled": item.get("enabled", True) is not False}
        with self.lock:
            items = self._read()
            old = next((it for it in items if it["id"] == item.get("id")), None)
            if old:
                old.update(fields)
                saved = old
            else:
                if len(items) >= MAX_SCHEDULES:
                    raise ValueError(f"最多 {MAX_SCHEDULES} 個排程")
                # 新增時:今天的時間如果已經過了,不要一存好就馬上補跑(從下一次開始)
                saved = {"id": secrets.token_hex(4), **fields, "created": round(time.time()), "last_run": round(time.time())}
                items.append(saved)
            self._write(items)
            return saved

    def remove(self, item_id: str) -> bool:
        with self.lock:
            items = self._read()
            kept = [it for it in items if it["id"] != item_id]
            self._write(kept)
            return len(kept) != len(items)

    def get(self, item_id: str) -> dict | None:
        return next((it for it in self.list() if it["id"] == item_id), None)

    def claim(self, item_id: str, now: datetime, force: bool = False) -> dict | None:
        """要跑之前先「認領」:再確認一次該跑(force = 使用者按了立刻跑),記下 last_run 再放開鎖。
        桌面版和 VS Code 外掛同時開著時,只有先認領到的那個會跑。回傳要跑的那筆;不該跑回傳 None。"""
        lock = self.path.with_suffix(".lock")
        try:
            if lock.exists() and time.time() - lock.stat().st_mtime > LOCK_STALE_SECONDS:
                lock.unlink(missing_ok=True)
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:  # 另一個行程正在認領
            return None
        try:
            os.close(fd)
            with self.lock:
                items = self._read()
                item = next((it for it in items if it["id"] == item_id), None)
                if not item or not (force or is_due(item, now)):
                    return None
                item["last_run"] = round(now.timestamp())
                self._write(items)
                return dict(item)
        finally:
            lock.unlink(missing_ok=True)

    def finished(self, item_id: str, conv_id: str | None, result: str) -> None:
        """跑完記下來:最後一次的對話和結果(畫面上「上次:… 打開對話」)。"""
        with self.lock:
            items = self._read()
            for it in items:
                if it["id"] == item_id:
                    it["last_conv"], it["last_result"] = conv_id, result[:200]
            self._write(items)


class Scheduler:
    """背景執行緒:每 TICK_SECONDS 秒看一次有沒有到時間的排程,到了就交給 run(item)。"""

    def __init__(self, schedules: Schedules, run: Callable[[dict], None]):
        self.schedules, self.run = schedules, run
        self.stop = threading.Event()

    def tick(self, now: datetime | None = None) -> list[str]:
        """跑一輪檢查(測試直接呼叫);回傳這一輪開始跑的排程 id。"""
        now = now or datetime.now(TAIPEI)
        started = []
        for item in self.schedules.list():
            if is_due(item, now) and (claimed := self.schedules.claim(item["id"], now)):
                started.append(claimed["id"])
                self.run(claimed)
        return started

    def start(self) -> "Scheduler":
        def loop():
            while True:  # 一開始就先看一次:HSUAN 關著時錯過的,打開就補跑
                try:
                    self.tick()
                except Exception:  # 一個排程壞掉不能讓整個排程器停掉
                    pass
                if self.stop.wait(TICK_SECONDS):
                    return
        threading.Thread(target=loop, daemon=True, name="scheduler").start()
        return self
