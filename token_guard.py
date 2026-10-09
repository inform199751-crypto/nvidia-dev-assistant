"""Token 管理:估算、截斷、歷史瘦身、大檔大綱、每人配額。

設計原則:每一道防線都在「送進 GPU 之前」把關,
推理伺服器(NIM)那一層只負責最後的硬上限。
"""
from __future__ import annotations

import ast
import json
import re
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

_CJK = re.compile(r"[　-鿿가-힯＀-￯]")
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S)
TAIPEI = timezone(timedelta(hours=8))  # 台灣沒有日光節約,固定 +8 即可(Windows 不必裝 tzdata)


class ContextTooLarge(Exception):
    """連長文本模型都放不下 → 改走離線批次(切塊 map-reduce),不要硬塞。"""


class QuotaExceeded(Exception):
    pass


# ---------- 估算 ----------

def estimate_tokens(text: str) -> int:
    """保守估算:中日韓字每字 1.5 token,其餘每 2.7 字元 1 token。

    2026-10-06 用 Nemotron 3 Ultra 實測校準:客服專案 57 檔(中文 4.8 萬字 + 其他 31 萬字元,
    含行號前綴)實際 18.2 萬 tokens;舊公式(1 / 3)只估 15.3 萬,低估近 20%。
    執行中再用回應的 usage_metadata 自動修正(見 agent.run_agent);
    地端可改打 NIM 的 POST /tokenize 拿精確值。
    """
    cjk = len(_CJK.findall(text))
    return int(cjk * 1.5 + (len(text) - cjk) / 2.7) + 1


def message_tokens(messages: list[BaseMessage]) -> int:
    total = 0
    for m in messages:
        total += 4 + estimate_tokens(m.content if isinstance(m.content, str) else str(m.content))
        if isinstance(m, AIMessage):
            total += sum(estimate_tokens(str(c.get("args", ""))) for c in m.tool_calls)
            total += estimate_tokens(m.additional_kwargs.get("reasoning_content") or "")
    return total


def tool_tokens(tools: list) -> int:
    """工具定義(名稱、說明、參數 schema)每一步都會跟著送出,是固定成本,要另外算進 prompt。

    2026-10-08 改 bug 實測:沒算它時,第一步實際 1271 / 估算 638,
    差額被當成估算公式偏低,之後每一步都被放大 1.99 倍,中型任務會提早升級到 Ultra。
    """
    if not tools:
        return 0
    return estimate_tokens(json.dumps([convert_to_openai_tool(t) for t in tools], ensure_ascii=False))


# ---------- 截斷與瘦身 ----------

def truncate_middle(text: str, max_tokens: int) -> str:
    """保留頭 60%、尾 40%(錯誤訊息和 traceback 通常在尾巴),中間換成提示。"""
    if estimate_tokens(text) <= max_tokens:
        return text
    lines = text.splitlines()
    head, tail, used = [], [], 0
    for line in lines:
        cost = estimate_tokens(line) + 1
        if used + cost > max_tokens * 0.6:
            break
        head.append(line)
        used += cost
    for line in reversed(lines[len(head):]):
        cost = estimate_tokens(line) + 1
        if used + cost > max_tokens:
            break
        tail.insert(0, line)
        used += cost
    dropped = len(lines) - len(head) - len(tail)
    marker = f"... [中間截斷 {dropped} 行;需要的話用 read_file 指定行號範圍讀取] ..."
    return "\n".join([*head, marker, *tail])


def strip_think_text(text: str) -> str:
    """去掉混在回答裡的推理:成對的 <think>…</think>,以及只剩結尾 </think> 的情況
    (Ultra 開 force_nonempty_content 時,開頭標籤會被伺服器吃掉)。"""
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.lstrip()


def strip_stale_reasoning(messages: list[BaseMessage]) -> list[BaseMessage]:
    """刪掉「上一輪以前」的思考內容,保留本輪的。

    langchain-nvidia-ai-endpoints 會把 AIMessage 的 reasoning_content 原樣送回伺服器,
    多輪對話每一輪都重送全部思考過程,token 會滾雪球。
    本輪(最後一個 HumanMessage 之後)保留,因為有些模型的 tool calling
    要求同一輪內把 reasoning 帶回去才能接著想。
    互動模式(AgentSession 多輪)每問一句,前面幾輪的思考就會被清掉。
    單次任務只有一個 HumanMessage,整個任務都算本輪,這裡不會刪任何東西;
    本輪的思考仍會累積,但 message_tokens 有算進去,超過預算一樣會瘦身 / 升級。
    """
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    out: list[BaseMessage] = []
    for i, m in enumerate(messages):
        if i < last_human and isinstance(m, AIMessage) and (
            "reasoning_content" in m.additional_kwargs or "</think>" in str(m.content)
        ):
            kwargs = {k: v for k, v in m.additional_kwargs.items()
                      if k not in ("reasoning_content", "reasoning", "_reasoning_api_fields")}
            content = strip_think_text(m.content) if isinstance(m.content, str) else m.content
            m = m.model_copy(update={"additional_kwargs": kwargs, "content": content})
        out.append(m)
    return out


def compact_history(messages: list[BaseMessage], budget: int, keep_recent_tools: int = 2) -> list[BaseMessage]:
    """超過預算時,把舊的工具輸出換成一行佔位字。

    只換內容、不刪訊息 —— tool_call 和 ToolMessage 必須成對,刪掉會讓 API 直接報錯。
    盡力而為:回傳後若仍超過預算,由呼叫端決定升級模型或拒絕。
    """
    if message_tokens(messages) <= budget:
        return messages
    tool_idx = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    out = list(messages)
    for i in tool_idx[:-keep_recent_tools] if keep_recent_tools else tool_idx:
        out[i] = out[i].model_copy(update={"content": "[較早的工具輸出已省略;需要時請重新呼叫工具]"})
        if message_tokens(out) <= budget:
            break
    return out


# ---------- 大檔:給大綱,不給全文 ----------

_DEF_LINE = re.compile(r"^\s*(?:export\s+)?(?:public|private|protected|static|async|def|class|function|func|fn|interface|struct)\b.*")


def outline(source: str, path: str = "") -> str:
    """產生檔案大綱(類別 / 函式 + 行號),讓 agent 先看地圖再用 read_file 讀需要的段落。"""
    if path.endswith(".py"):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        if tree is not None:
            rows = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "def"
                    rows.append((node.lineno, f"L{node.lineno}-{node.end_lineno}  {kind} {node.name}"))
            return "\n".join(r for _, r in sorted(rows))
    hits = [f"L{n}  {line.strip()[:120]}" for n, line in enumerate(source.splitlines(), 1) if _DEF_LINE.match(line)]
    return "\n".join(hits) or "(找不到函式定義;請用 search_code 搜尋)"


# ---------- 每人配額 + 同時請求數 ----------

@dataclass
class _Usage:
    day: str = ""
    tokens: int = 0
    running: int = 0


class TokenQuota:
    """每人每日 token 額度 + 同時請求數上限。一天以台灣午夜切。

    store:今天的用量存在這個 JSON 檔(每次扣額度都讀檔、加上去、寫回):重開程式不會歸零,
    桌面版和 VS Code 外掛這種同時開著的不同行程也加在同一份。None = 只記在記憶體(測試)。
    API server 不只一台時要換成 RedisTokenQuota。
    """

    def __init__(self, daily_tokens: int, max_concurrent: int, store: Path | None = None):
        self.daily_tokens = daily_tokens
        self.max_concurrent = max_concurrent
        self.store = store
        self._lock = threading.Lock()
        self._usage: dict[str, _Usage] = defaultdict(_Usage)

    def _read(self) -> dict:
        try:
            return json.loads(self.store.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _today(self, user: str) -> _Usage:
        u = self._usage[user]
        today = datetime.now(TAIPEI).date().isoformat()
        if self.store:  # 檔案才是準的(別的行程可能也用了)
            u.day, u.tokens = today, int((self._read().get(user) or {}).get(today, 0))
        elif u.day != today:
            u.day, u.tokens = today, 0
        return u

    def used(self, user: str) -> int:
        """今天已經用了多少 tokens(畫面上顯示用)。"""
        with self._lock:
            return self._today(user).tokens

    def check(self, user: str, need: int) -> None:
        with self._lock:
            u = self._today(user)
            if u.tokens + need > self.daily_tokens:
                raise QuotaExceeded(f"{user} 今日額度剩 {self.daily_tokens - u.tokens} tokens,這次約需 {need}")

    def charge(self, user: str, tokens: int) -> None:
        with self._lock:
            u = self._today(user)
            u.tokens += tokens
            if self.store and tokens:
                data = self._read()
                data[user] = {u.day: u.tokens}  # 每人只留今天
                self.store.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.store.with_name(self.store.name + ".tmp")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                tmp.replace(self.store)

    @contextmanager
    def slot(self, user: str):
        with self._lock:
            u = self._today(user)
            if u.running >= self.max_concurrent:
                raise QuotaExceeded(f"{user} 已有 {u.running} 個請求在跑,請等前一個完成")
            u.running += 1
        try:
            yield
        finally:
            with self._lock:
                self._usage[user].running -= 1


class RedisTokenQuota:
    """TokenQuota 的 Redis 版:API server 不只一台時,每人額度和同時請求數要放在共用的地方。

    每人每天一個 key(INCRBY,兩天後過期);同時請求數一個 key(INCR / DECR,設過期時間:
    伺服器當掉沒有 DECR 的話,最多卡一小時)。只用最基本的指令,換成其他相容 Redis 的服務也能用。
    """

    def __init__(self, client, daily_tokens: int, max_concurrent: int, prefix: str = "dev-assistant"):
        self.r, self.daily_tokens, self.max_concurrent, self.prefix = client, daily_tokens, max_concurrent, prefix

    def _day_key(self, user: str) -> str:
        return f"{self.prefix}:tokens:{user}:{datetime.now(TAIPEI).date().isoformat()}"

    def used(self, user: str) -> int:
        return int(self.r.get(self._day_key(user)) or 0)

    def check(self, user: str, need: int) -> None:
        used = int(self.r.get(self._day_key(user)) or 0)
        if used + need > self.daily_tokens:
            raise QuotaExceeded(f"{user} 今日額度剩 {self.daily_tokens - used} tokens,這次約需 {need}")

    def charge(self, user: str, tokens: int) -> None:
        if tokens:
            key = self._day_key(user)
            self.r.incrby(key, tokens)
            self.r.expire(key, 2 * 86400)

    @contextmanager
    def slot(self, user: str):
        key = f"{self.prefix}:running:{user}"
        running = self.r.incr(key)
        self.r.expire(key, 3600)
        if running > self.max_concurrent:
            self.r.decr(key)
            raise QuotaExceeded(f"{user} 已有 {running - 1} 個請求在跑,請等前一個完成")
        try:
            yield
        finally:
            self.r.decr(key)


def make_quota(daily_tokens: int, max_concurrent: int, redis_url: str | None = None, store: Path | None = None):
    """有 REDIS_URL 就用 Redis(多台伺服器共用額度),沒有就用本機版(store 給了就存檔,重開不歸零)。"""
    if not redis_url:
        return TokenQuota(daily_tokens, max_concurrent, store)
    import redis
    return RedisTokenQuota(redis.Redis.from_url(redis_url, decode_responses=True), daily_tokens, max_concurrent)
