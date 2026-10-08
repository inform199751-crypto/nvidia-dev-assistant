"""對話管理:每則訊息是一輪,在背景執行緒跑 agent;事件存在記憶體並同步寫檔,網頁靠它即時顯示與補送。"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import messages_from_dict, messages_to_dict

from agent import omit_attachments, run_agent
from token_guard import ContextTooLarge, QuotaExceeded
from web.workspaces import WorkspaceRegistry

log = logging.getLogger("dev_assistant.web")

FINISHED = {"done", "failed", "cancelled", "interrupted"}
MODES = {"fix", "whole"}


class Busy(Exception):
    """同一個對話已經有一輪在跑。"""


class WorkspaceGone(Exception):
    """對話綁定的資料夾已不存在(上傳的複本 24 小時後會刪除)。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Turn:
    id: str
    question: str
    mode: str
    options: dict
    status: str = "queued"
    events: list = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def summary(self) -> dict:
        return {"id": self.id, "question": self.question, "mode": self.mode, "options": self.options,
                "status": self.status, "created_at": self.created_at}


@dataclass
class Conversation:
    id: str
    title: str
    workspace: str
    created_at: str = field(default_factory=_now)
    turns: list = field(default_factory=list)
    messages: list = field(default_factory=list)  # messages_to_dict 的結果,追問時還原成前文

    def turn(self, tid: str) -> Turn:
        for t in self.turns:
            if t.id == tid:
                return t
        raise KeyError(tid)

    def summary(self) -> dict:
        status = self.turns[-1].status if self.turns else "done"
        return {"id": self.id, "title": self.title, "workspace": self.workspace,
                "created_at": self.created_at, "status": status}

    def detail(self) -> dict:
        return {**self.summary(), "turns": [t.summary() for t in self.turns]}

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        return cls(**{**data, "turns": [Turn(**t) for t in data.get("turns", [])]})


def _split_files(raw: str | None) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _error_event(exc: Exception) -> dict:
    first = (str(exc).splitlines() or [""])[0][:300]
    if isinstance(exc, QuotaExceeded):
        return {"type": "error", "kind": "quota", "message": f"今天的額度不夠了:{first}"}
    if isinstance(exc, ContextTooLarge):
        return {"type": "error", "kind": "context", "message": f"內容太大:{first}"}
    if isinstance(exc, WorkspaceGone):
        return {"type": "error", "kind": "workspace",
                "message": "這個對話的資料夾已不存在(上傳的複本 24 小時後會刪除),請開新對話。"}
    if isinstance(exc, PermissionError):
        return {"type": "error", "kind": "permission", "message": first}
    if "[429]" in first or "[503]" in first:
        return {"type": "error", "kind": "busy", "message": "NVIDIA 服務忙線,已試過備援模型仍失敗,請稍後按重試。"}
    return {"type": "error", "kind": "other", "message": f"{type(exc).__name__}: {first}"}


class ConversationManager:
    def __init__(self, store_dir: Path, workspaces: WorkspaceRegistry, registry, quota,
                 executor: Executor | None = None, agent_fn: Callable[..., Any] = run_agent):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces = workspaces
        self.registry = registry
        self.quota = quota
        self.executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="turn")
        self.agent_fn = agent_fn
        self.lock = threading.RLock()
        self.convs: dict[str, Conversation] = {}
        self.stop_flags: dict[str, threading.Event] = {}
        self._load()

    # ---------- 存取 ----------
    def _load(self) -> None:
        for path in sorted(self.store_dir.glob("*.json")):
            try:
                conv = Conversation.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:  # 一個壞檔(或 Turn 欄位改版)不能讓整個伺服器起不來
                log.exception("對話檔 %s 讀取失敗,改名為 .bad 後略過", path.name)
                try:
                    os.replace(path, path.with_suffix(".bad"))
                except OSError:
                    pass
                continue
            changed = False
            for turn in conv.turns:
                if turn.status not in FINISHED:  # 伺服器重開前沒跑完的,標成中斷
                    turn.status = "interrupted"
                    turn.events.append({"seq": len(turn.events), "t": _now(), "type": "end",
                                        "status": "interrupted", "steps": 0, "tokens": 0, "seconds": 0})
                    changed = True
            self.convs[conv.id] = conv
            if changed:
                self._save(conv)

    def _save(self, conv: Conversation) -> None:
        path = self.store_dir / f"{conv.id}.json"
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(asdict(conv), ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)  # 先寫暫存檔再取代,寫到一半當機也不會壞檔
        except (OSError, ValueError):  # 存檔失敗(防毒軟體鎖檔等)不能中斷正在跑的這一輪;記憶體裡的狀態照樣正確,下一個事件會再寫一次
            log.exception("對話 %s 存檔失敗", conv.id)

    def list(self) -> list[dict]:
        with self.lock:
            convs = list(self.convs.values())
        return [c.summary() for c in sorted(convs, key=lambda c: c.created_at, reverse=True)]

    def get(self, cid: str) -> Conversation:
        with self.lock:
            return self.convs[cid]

    # ---------- 建立與控制 ----------
    def start(self, workspace_id: str, question: str, mode: str, options: dict | None = None) -> tuple[str, str]:
        self.workspaces.get(workspace_id)  # 不存在就 KeyError
        self._validate(question, mode)
        conv = Conversation(id=uuid.uuid4().hex[:12], title=question.strip()[:30], workspace=workspace_id)
        with self.lock:
            self.convs[conv.id] = conv
        return conv.id, self.add_turn(conv.id, question, mode, options)

    def add_turn(self, cid: str, question: str, mode: str, options: dict | None = None) -> str:
        self._validate(question, mode)
        with self.lock:
            conv = self.convs[cid]
            if conv.turns and conv.turns[-1].status not in FINISHED:
                raise Busy("這個對話還有一輪在跑,等它結束或先按停止")
            turn = Turn(id=uuid.uuid4().hex[:8], question=question.strip(), mode=mode, options=dict(options or {}))
            conv.turns.append(turn)
            self.stop_flags[turn.id] = threading.Event()
            self._save(conv)
        self.executor.submit(self._run, cid, turn.id)
        return turn.id

    def cancel(self, cid: str, tid: str) -> None:
        self.get(cid).turn(tid)  # 不存在就 KeyError
        flag = self.stop_flags.get(tid)
        if flag:
            flag.set()

    @staticmethod
    def _validate(question: str, mode: str) -> None:
        if not (question or "").strip():
            raise ValueError("問題不能是空的")
        try:
            question.encode("utf-8")
        except UnicodeEncodeError:  # 落單的代理字元(例如 \ud800)寫不進 JSON 檔,先擋掉
            raise ValueError("問題裡有無法辨識的字元") from None
        if mode not in MODES:
            raise ValueError(f"模式只能是 {sorted(MODES)}")

    # ---------- 執行 ----------
    def _emit(self, conv: Conversation, turn: Turn, event: dict) -> None:
        with self.lock:
            turn.events.append({"seq": len(turn.events), "t": _now(), **event})
            self._save(conv)

    def _finish(self, conv: Conversation, turn: Turn, status: str, steps: int, tokens: int, started: float) -> None:
        with self.lock:
            turn.status = status
            self._emit(conv, turn, {"type": "end", "status": status, "steps": steps, "tokens": tokens,
                                    "seconds": round(time.monotonic() - started, 1)})
        self.stop_flags.pop(turn.id, None)

    def _run(self, cid: str, tid: str) -> None:
        conv = self.get(cid)
        turn = conv.turn(tid)
        stop = self.stop_flags[tid]
        started = time.monotonic()
        if stop.is_set():  # 還在排隊就被取消
            self._finish(conv, turn, "cancelled", 0, 0, started)
            return
        with self.lock:
            turn.status = "running"
            self._save(conv)
        try:
            try:
                ws = self.workspaces.get(conv.workspace)
            except KeyError:
                raise WorkspaceGone(conv.workspace) from None
            whole = turn.mode == "whole"
            history = messages_from_dict(conv.messages) if conv.messages else None
            result = self.agent_fn(
                turn.question, registry=self.registry, quota=self.quota, user="web",
                attachments=self.workspaces.whole_files(ws) if whole else _split_files(turn.options.get("files")),
                role=turn.options.get("role") or None, thinking=turn.options.get("thinking"),
                whole=whole, workspace=ws.path, history=history,
                on_event=lambda e: self._emit(conv, turn, e), should_stop=stop.is_set)
            with self.lock:
                # 存檔不留附檔全文(整份文件模式可能好幾 MB,每個事件都要整份重寫);追問本來就會先 omit_attachments,所以不損失任何東西
                conv.messages = messages_to_dict(omit_attachments(result.messages))
            self._finish(conv, turn, "cancelled" if result.stopped else "done",
                         result.steps, result.tokens_used, started)
        except Exception as exc:  # 任何錯誤都要讓畫面知道,不能讓這一輪永遠停在「執行中」
            log.exception("對話 %s 第 %s 輪執行失敗", cid, tid)
            self._emit(conv, turn, _error_event(exc))
            self._finish(conv, turn, "failed", 0, 0, started)
