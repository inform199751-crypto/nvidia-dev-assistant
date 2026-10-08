"""對內研發助理:核心 agent 迴圈。

自己寫迴圈而不用現成的 create_agent,是為了在「每一步送出前」插入 token 檢查、
歷史瘦身與模型升級 —— 這三件事就是防 OOM 和控成本的位置。
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from llm_registry import ModelRegistry
from token_guard import (TokenQuota, compact_history, estimate_tokens, message_tokens,
                         outline, strip_stale_reasoning, strip_think_text, tool_tokens, truncate_middle)
import tools as toolbox
from tools import ALL_TOOLS

log = logging.getLogger("dev_assistant")

# 指令用英文寫(各家程式碼模型的指令微調資料以英文為主),回答語言另外指定。
SYSTEM_PROMPT = """You are an internal senior software engineer assistant. Reply in Traditional Chinese (zh-TW); keep code, identifiers and error messages in their original language.

## Workflow for bugs and code changes
1. Locate: use search_code / read_file to find the relevant lines. Do not guess what code you have not read.
2. Reproduce: run run_tests (or reason from the given traceback) before changing anything.
3. Hypothesize: state the root cause in one or two sentences, citing path:line.
4. Fix and verify: call propose_patch with edits. Each edit replaces old_string with new_string in one file; copy old_string verbatim from read_file output WITHOUT the line-number prefix, and include a few surrounding lines if it is not unique. To add a new file (e.g. a regression test), use a path that does not exist with an empty old_string. The tool builds the diff, applies it to a scratch copy, runs the tests and returns the diff and the real post-patch result. Keep changes minimal; do not refactor unrelated code.
5. If the post-patch tests still fail, read that output and send a revised patch. Edits always apply to the ORIGINAL files (earlier patches are not applied), and run_tests always tests the ORIGINAL code, so do not use it to verify a patch.

## Rules
- Cite evidence as path:line. If something is uncertain or not visible in the files, say so instead of inventing APIs or behavior.
- Never state test results that did not come from a tool output in this conversation.
- Prefer small targeted reads over reading whole files. Large files come with an outline; read only the ranges you need.
- Tool errors are feedback: read them and correct your call; do not repeat the same failing call.
- For documentation tasks: structure as Overview → Architecture → Key flows → API/Config → Known limitations."""


@dataclass
class AgentResult:
    answer: str
    role: str
    steps: int
    tokens_used: int


def _numbered(text: str) -> str:
    return "\n".join(f"{n:>5}| {line}" for n, line in enumerate(text.splitlines(), 1))


def build_user_prompt(task: str, attachments: list[str], inline_limit: int, whole: bool) -> str:
    """長內容放前面、問題放最後(長文本模型對結尾的指令最敏感,也讓前綴快取命中)。

    whole=False:小檔內嵌全文,大檔只給大綱 + 工具自己讀(省 token,幾乎所有改 bug 情境都夠用)。
    whole=True :全部內嵌,交給路由決定要不要升級到長文本模型(寫整份技術文件、全檔審查)。
    """
    parts = []
    for rel in attachments:
        text = toolbox._safe_path(rel).read_text(encoding="utf-8", errors="replace")  # 附檔也要過機密檔檢查
        n_lines = text.count("\n") + 1
        if whole or estimate_tokens(text) <= inline_limit:
            parts.append(f'<file path="{rel}" lines="1-{n_lines}">\n{_numbered(text)}\n</file>')
        else:
            parts.append(f'<file_outline path="{rel}" total_lines="{n_lines}">\n{outline(text, rel)}\n'
                         f"</file_outline>\n(檔案太大未內嵌,請用 read_file 讀需要的行號範圍)")
    if whole:  # 整檔模式不給工具:檔案都在上面了,給工具只會讓模型分段重讀、每步重送全部內容
        parts.append("<note>所有相關檔案都已完整附在上面。這次沒有工具可用,請直接根據附檔內容完成任務。</note>")
    parts.append(f"<task>\n{task}\n</task>")
    return "\n\n".join(parts)


def file_mentions(text: str) -> list[str]:
    """問題裡的 @路徑(例如「看一下 @calc.py」)且確實是工作區檔案的,當成附檔;@decorator 之類的不算。
    只認 ASCII 路徑字元:中文常不空格(「看@calc.py的add」),\\w 會把後面的中文也吃進去。"""
    found = []
    for m in re.finditer(r"@([\w./\\-]+)", text, re.ASCII):
        rel = m.group(1).rstrip(".,;:")
        try:
            if toolbox._safe_path(rel).is_file() and rel not in found:
                found.append(rel)
        except (ValueError, PermissionError):
            pass
    return found


class AgentSession:
    """一段對話:保留完整 messages,每一輪(每次 ask)都走同一套逐步檢查。CLI 單次任務就是只問一輪。

    跨輪保留的:對話內容、估算倍率 scale。每輪重新決定模型(放得下就回到預設角色)。
    """

    def __init__(self, registry: ModelRegistry, quota: TokenQuota, *, user: str = "cli",
                 role: str | None = None, thinking: bool | None = None, whole: bool = False,
                 tools: list = ALL_TOOLS):
        self.registry, self.quota, self.user = registry, quota, user
        self.requested_role, self.thinking, self.whole = role, thinking, whole
        self.tools = [] if whole else list(tools)
        self.tool_map = {t.name: t for t in self.tools}
        self.overhead = tool_tokens(self.tools)  # 工具定義每步都送,是固定成本;不算進來會被誤判成估算偏低
        self.messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
        self.scale = 1.0  # 估算偏低時,用伺服器回報的實際 input tokens 往上修正
        self.tokens_used = 0  # 整段對話累計

    def sized(self) -> int:
        return int((message_tokens(self.messages) + self.overhead) * self.scale)

    def ask(self, task: str, attachments: list[str] = ()) -> AgentResult:
        """問一輪。中途出錯(額度、伺服器全掛)就把這一輪的訊息退掉,對話停在上一輪結束的狀態。"""
        start = len(self.messages)
        try:
            return self._ask(task, list(attachments))
        except BaseException:
            del self.messages[start:]
            raise

    def _ask(self, task: str, attachments: list[str]) -> AgentResult:
        reg, limits, tools = self.registry, self.registry.limits, self.tools
        self.messages.append(HumanMessage(
            build_user_prompt(task, attachments, limits["inline_file_max_tokens"], self.whole)))
        role = reg.pick_role(prompt_tokens=self.sized(), needs_tools=bool(tools), requested=self.requested_role)
        llm = reg.agent_model(role, tools, thinking=self.thinking)
        used = 0

        with self.quota.slot(self.user):
            for step in range(1, limits["max_agent_steps"] + 1):
                # 每一步送出前的三道檢查:清掉舊思考 → 瘦身 → 還放不下就升級模型
                self.messages = strip_stale_reasoning(self.messages)
                budget = reg.budget(role)
                if self.sized() > budget:
                    self.messages = compact_history(self.messages, int(budget / self.scale) - self.overhead)
                if (tokens := self.sized()) > budget:
                    new_role = reg.pick_role(prompt_tokens=tokens, needs_tools=bool(tools))  # 放不下會丟 ContextTooLarge
                    log.info("對話約 %s tokens 超過 %s 的預算,升級到 %s", tokens, role, new_role)
                    role, llm = new_role, reg.agent_model(new_role, tools, thinking=self.thinking)
                self.quota.check(self.user, self.sized() + reg.spec(role).max_output)  # 每步都檢查,不是只在開始

                estimated = message_tokens(self.messages) + self.overhead
                ai = llm.invoke(self.messages)
                usage = ai.usage_metadata or {}
                if usage.get("input_tokens") and usage["input_tokens"] / estimated > self.scale * 1.05:
                    self.scale = usage["input_tokens"] / estimated
                    log.info("估算偏低:實際 %s / 估算 %s,之後的預算按 %.2f 倍計",
                             usage["input_tokens"], estimated, self.scale)
                step_tokens = usage.get("total_tokens", 0)
                used += step_tokens
                self.tokens_used += step_tokens
                self.quota.charge(self.user, step_tokens)
                self.messages.append(ai)
                calls = ", ".join([f"{c['name']}({str(c['args'])[:60]})" for c in ai.tool_calls]
                                  + [f"{c.get('name')}(參數格式錯誤)" for c in ai.invalid_tool_calls]) or "(回答)"
                log.info("step %d  role=%s  tokens=%d  %s", step, role, step_tokens, calls)

                if not ai.tool_calls and not ai.invalid_tool_calls:
                    return AgentResult(strip_think_text(str(ai.content)), role, step, used)
                for call in ai.tool_calls:
                    tool = self.tool_map.get(call["name"])
                    try:
                        out = tool.invoke(call["args"]) if tool else f"錯誤:沒有 {call['name']} 這個工具"
                    except Exception as e:  # 錯誤回給模型自己修正,不讓整個流程掛掉
                        out = f"工具執行錯誤:{type(e).__name__}: {e}"
                    self.messages.append(ToolMessage(truncate_middle(str(out), limits["tool_output_max_tokens"]),
                                                     tool_call_id=call["id"]))
                # 參數不是合法 JSON 的呼叫會落在 invalid_tool_calls(tool_calls 是空的);
                # 套件仍會把原始 tool call 送回伺服器,所以一樣要回一則 ToolMessage 讓模型修正重送
                for bad in ai.invalid_tool_calls:
                    out = f"錯誤:{bad.get('name')} 的參數不是合法 JSON,請修正後重送。\n{bad.get('error') or ''}"
                    self.messages.append(ToolMessage(truncate_middle(out, limits["tool_output_max_tokens"]),
                                                     tool_call_id=bad.get("id") or ""))
        return AgentResult(f"[已達 {limits['max_agent_steps']} 步上限,停止。請把問題拆小再問]",
                           role, limits["max_agent_steps"], used)


def run_agent(task: str, *, registry: ModelRegistry, quota: TokenQuota, user: str = "cli",
              attachments: list[str] = (), role: str | None = None, thinking: bool | None = None,
              whole: bool = False, tools: list = ALL_TOOLS) -> AgentResult:
    """單次任務:開一段新對話、只問一輪。"""
    session = AgentSession(registry, quota, user=user, role=role, thinking=thinking, whole=whole, tools=tools)
    return session.ask(task, attachments)


HELP = """直接輸入問題,Enter 送出。用 @路徑 附上工作區的檔案,例如:看一下 @calc.py 的 add
  /clear  開始新對話(清掉前面的內容)
  /help   顯示這段說明
  /exit   離開(也可以按 Ctrl+C)"""


def chat(session_factory, first_files: list[str]) -> None:
    """互動模式:像 Claude Code 一樣一直聊,前面說過的內容都記得。"""
    session = session_factory()
    print(f"對內研發助理 · 工作區 {toolbox.WORKSPACE} · 輸入 /help 看用法、/exit 離開")
    pending = list(first_files)  # -f 給的檔案附在第一個問題
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return
        if text == "/help":
            print(HELP)
            continue
        if text == "/clear":
            session = session_factory()
            print("已開始新對話。")
            continue
        try:
            r = session.ask(text, pending + [f for f in file_mentions(text) if f not in pending])
        except KeyboardInterrupt:
            print("\n(已中斷這一輪,對話保留在上一輪結束的狀態)")
            continue
        except Exception as e:  # 額度用完、模型全掛:這輪退掉,對話還能繼續
            print(f"\n錯誤:{type(e).__name__}: {e}")
            continue
        pending = []
        print(f"\n{r.answer}\n\n— {r.role} · {r.steps} 步 · {r.tokens_used:,} tokens"
              f"(這段對話累計 {session.tokens_used:,})")


def main() -> None:
    from dotenv import load_dotenv

    here = Path(__file__).parent
    load_dotenv(here / ".env")  # 只讀本專案的 .env,不往上層目錄找
    workspace = Path(os.getenv("AGENT_WORKSPACE", "."))
    toolbox.configure(workspace if workspace.is_absolute() else here / workspace, os.getenv("SANDBOX_IMAGE"))
    ap = argparse.ArgumentParser(description="對內研發助理(NVIDIA NIM / API Catalog)")
    ap.add_argument("task", nargs="?", help="要做的事;不給就進入互動模式,可以一直聊下去")
    ap.add_argument("-f", "--file", action="append", default=[], help="附檔(相對於 AGENT_WORKSPACE)")
    ap.add_argument("--role", help="強制指定模型角色:code(日常)/ code_deep(難題)/ long_context")
    ap.add_argument("--think", choices=["on", "off"], help="思考模式;不給就用伺服器預設")
    ap.add_argument("--whole", action="store_true", help="整檔內嵌、不給工具、一次回答(寫文件 / 全檔審查用)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    registry = ModelRegistry.from_yaml()
    quota = TokenQuota(registry.limits["daily_tokens_per_user"], registry.limits["max_concurrent_per_user"])
    thinking = {"on": True, "off": False}.get(args.think)

    def new_session() -> AgentSession:
        return AgentSession(registry, quota, role=args.role, thinking=thinking, whole=args.whole)

    if args.task is None:
        chat(new_session, args.file)
        return
    r = new_session().ask(args.task, args.file)
    print(f"\n{r.answer}\n\n— {r.role} · {r.steps} 步 · {r.tokens_used:,} tokens")


if __name__ == "__main__":
    main()
