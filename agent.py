"""對內研發助理:核心 agent 迴圈。

自己寫迴圈而不用現成的 create_agent,是為了在「每一步送出前」插入 token 檢查、
歷史瘦身與模型升級 —— 這三件事就是防 OOM 和控成本的位置。
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from llm_registry import ModelRegistry
from token_guard import (TokenQuota, compact_history, estimate_tokens, message_tokens,
                         outline, strip_stale_reasoning, strip_think_text, truncate_middle)
import tools as toolbox

log = logging.getLogger("dev_assistant")

# 指令用英文寫(各家程式碼模型的指令微調資料以英文為主),回答語言另外指定。
SYSTEM_PROMPT = """You are an internal senior software engineer assistant. Reply in Traditional Chinese (zh-TW); keep code, identifiers and error messages in their original language.

## Workflow for bugs and code changes
1. Locate: use search_code / read_file to find the relevant lines. Do not guess what code you have not read.
2. Reproduce: run run_tests (or reason from the given traceback) before changing anything.
3. Hypothesize: state the root cause in one or two sentences, citing path:line.
4. Fix and verify: call propose_patch with a minimal unified diff (paths relative to workspace, a/ b/ prefixes). It applies the patch to a scratch copy, runs the tests and returns the real post-patch result. Do not refactor unrelated code.
5. If the post-patch tests still fail, read that output and send a revised patch. Note: run_tests always tests the ORIGINAL code, so do not use it to verify a patch.

## Rules
- Cite evidence as path:line. If something is uncertain or not visible in the files, say so instead of inventing APIs or behavior.
- Never state test results that did not come from a tool output in this conversation.
- Prefer small targeted reads over reading whole files. Large files come with an outline; read only the ranges you need.
- Tool errors are feedback: read them and correct your call; do not repeat the same failing call.
- For documentation tasks: structure as Overview → Architecture → Key flows → API/Config → Known limitations."""

WHOLE_NOTE = "<note>所有相關檔案都已完整附在上面。這次沒有工具可用,請直接根據附檔內容完成任務。</note>"
OMITTED_NOTE = "<note>前文附檔內容已省略;需要時用 read_file 讀取。</note>"
_FILE_BLOCK = re.compile(r'<file path="([^"]+)" lines="[^"]*">\n.*?\n</file>', re.S)
_FAILED_PREFIXES = ("錯誤", "工具執行錯誤", "diff 套不上去")


@dataclass
class AgentResult:
    answer: str
    role: str
    steps: int
    tokens_used: int
    messages: list[BaseMessage] = field(default_factory=list)  # 完整訊息串,追問時當前文
    stopped: bool = False


def _numbered(text: str) -> str:
    return "\n".join(f"{n:>5}| {line}" for n, line in enumerate(text.splitlines(), 1))


def build_user_prompt(task: str, attachments: list[str], inline_limit: int, whole: bool,
                      root: Path | None = None) -> str:
    """長內容放前面、問題放最後(長文本模型對結尾的指令最敏感,也讓前綴快取命中)。

    whole=False:小檔內嵌全文,大檔只給大綱 + 工具自己讀(省 token,幾乎所有改 bug 情境都夠用)。
    whole=True :全部內嵌,交給路由決定要不要升級到長文本模型(寫整份技術文件、全檔審查)。
    root:附檔所在資料夾;None 就用全域 WORKSPACE(指令列)。
    """
    base = root if root is not None else toolbox.WORKSPACE
    parts = []
    for rel in attachments:
        text = toolbox.safe_path(base, rel).read_text(encoding="utf-8", errors="replace")  # 附檔也要過機密檔檢查
        n_lines = text.count("\n") + 1
        if whole or estimate_tokens(text) <= inline_limit:
            parts.append(f'<file path="{rel}" lines="1-{n_lines}">\n{_numbered(text)}\n</file>')
        else:
            parts.append(f'<file_outline path="{rel}" total_lines="{n_lines}">\n{outline(text, rel)}\n'
                         f"</file_outline>\n(檔案太大未內嵌,請用 read_file 讀需要的行號範圍)")
    if whole:  # 整檔模式不給工具:檔案都在上面了,給工具只會讓模型分段重讀、每步重送全部內容
        parts.append(WHOLE_NOTE)
    parts.append(f"<task>\n{task}\n</task>")
    return "\n\n".join(parts)


def omit_attachments(messages: list[BaseMessage]) -> list[BaseMessage]:
    """追問時不重送前文的附檔全文(整份文件模式一次就是十幾萬 tokens),只留檔名;需要再用工具讀。"""
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, HumanMessage) and isinstance(m.content, str) and '<file path="' in m.content:
            content = _FILE_BLOCK.sub(r'<file path="\1" omitted="true"/>', m.content)
            content = content.replace(WHOLE_NOTE, OMITTED_NOTE)
            if OMITTED_NOTE not in content:
                content = content.replace("<task>", f"{OMITTED_NOTE}\n\n<task>", 1)
            m = m.model_copy(update={"content": content})
        out.append(m)
    return out


def _patch_event(args: dict, output: str) -> dict:
    """把 propose_patch 的參數(diff)和結果(套用後的測試輸出)組成畫面要的修正事件。"""
    saved = re.match(r"已存成 (\S+?),", output)
    test_output = output.split("套用此 patch 後的測試結果:\n", 1)[-1]
    return {"type": "patch", "path": args.get("path", ""), "diff": args.get("unified_diff", ""),
            "patch_file": saved.group(1) if saved else "", "test_output": test_output[:4000],
            "passed": test_output.startswith("exit code 0")}


def run_agent(task: str, *, registry: ModelRegistry, quota: TokenQuota, user: str = "cli",
              attachments: list[str] = (), role: str | None = None, thinking: bool | None = None,
              whole: bool = False, tools: list | None = None, workspace: Path | None = None,
              history: list[BaseMessage] | None = None,
              on_event: Callable[[dict], None] | None = None,
              should_stop: Callable[[], bool] | None = None) -> AgentResult:
    """跑一輪 agent。

    tools=None 時用 make_tools(workspace) 產生綁定資料夾的工具;workspace=None 就跟著全域 WORKSPACE(指令列)。
    history:前幾輪的完整訊息串(追問用),附檔全文會先被 omit_attachments 拿掉。
    on_event:每一步回報事件給網頁;should_stop:每步送出前檢查,回 True 就停。
    """
    emit = on_event or (lambda event: None)
    limits = registry.limits
    if tools is None:
        tools = toolbox.make_tools(workspace)
    if whole:
        tools = []
    prompt = HumanMessage(build_user_prompt(task, list(attachments), limits["inline_file_max_tokens"],
                                            whole, root=workspace))
    messages: list[BaseMessage] = (omit_attachments(list(history)) if history
                                   else [SystemMessage(SYSTEM_PROMPT)]) + [prompt]
    role = registry.pick_role(prompt_tokens=message_tokens(messages), needs_tools=bool(tools), requested=role)
    llm = registry.agent_model(role, tools, thinking=thinking)
    tool_map = {t.name: t for t in tools}
    used = 0
    scale = 1.0  # 估算偏低時,用伺服器回報的實際 input tokens 往上修正
    emit({"type": "start", "role": role, "model": registry.spec(role).model,
          "mode": "whole" if whole else "fix", "est_tokens": message_tokens(messages)})

    def sized() -> int:
        return int(message_tokens(messages) * scale)

    with quota.slot(user):
        for step in range(1, limits["max_agent_steps"] + 1):
            if should_stop and should_stop():
                return AgentResult("[已停止]", role, step - 1, used, messages, stopped=True)
            # 每一步送出前的三道檢查:清掉舊思考 → 瘦身 → 還放不下就升級模型
            messages = strip_stale_reasoning(messages)
            budget = registry.budget(role)
            if sized() > budget:
                messages = compact_history(messages, int(budget / scale))
            if (tokens := sized()) > budget:
                new_role = registry.pick_role(prompt_tokens=tokens, needs_tools=bool(tools))  # 放不下會丟 ContextTooLarge
                log.info("對話約 %s tokens 超過 %s 的預算,升級到 %s", tokens, role, new_role)
                emit({"type": "model_switch", "from_role": role, "to_role": new_role,
                      "reason": f"內容約 {tokens:,} tokens,超過 {role} 的上限"})
                role, llm = new_role, registry.agent_model(new_role, tools, thinking=thinking)
            quota.check(user, sized() + registry.spec(role).max_output)  # 每步都檢查,不是只在開始
            emit({"type": "step", "step": step, "role": role, "model": registry.spec(role).model})

            estimated = message_tokens(messages)
            ai = llm.invoke(messages)
            usage = ai.usage_metadata or {}
            if usage.get("input_tokens") and estimated and usage["input_tokens"] / estimated > scale * 1.05:
                scale = usage["input_tokens"] / estimated
                log.info("估算偏低:實際 %s / 估算 %s,之後的預算按 %.2f 倍計", usage["input_tokens"], estimated, scale)
            step_tokens = usage.get("total_tokens", 0)
            used += step_tokens
            quota.charge(user, step_tokens)
            messages.append(ai)
            calls = ", ".join(f"{c['name']}({str(c['args'])[:60]})" for c in ai.tool_calls) or "(回答)"
            log.info("step %d  role=%s  tokens=%d  %s", step, role, step_tokens, calls)

            if not ai.tool_calls:
                answer = strip_think_text(str(ai.content))
                emit({"type": "answer", "text": answer})
                return AgentResult(answer, role, step, used, messages)
            for call in ai.tool_calls:
                emit({"type": "tool_call", "step": step, "name": call["name"], "args": call["args"]})
                tool = tool_map.get(call["name"])
                try:
                    out = str(tool.invoke(call["args"])) if tool else f"錯誤:沒有 {call['name']} 這個工具"
                except Exception as e:  # 錯誤回給模型自己修正,不讓整個流程掛掉
                    out = f"工具執行錯誤:{type(e).__name__}: {e}"
                emit({"type": "tool_result", "step": step, "name": call["name"],
                      "ok": not out.startswith(_FAILED_PREFIXES), "preview": out[:4000]})
                if call["name"] == "propose_patch" and out.startswith("已存成"):
                    emit(_patch_event(call["args"], out))
                messages.append(ToolMessage(truncate_middle(out, limits["tool_output_max_tokens"]),
                                            tool_call_id=call["id"]))
    answer = f"[已達 {limits['max_agent_steps']} 步上限,停止。請把問題拆小再問]"
    emit({"type": "answer", "text": answer})
    return AgentResult(answer, role, limits["max_agent_steps"], used, messages)


def main() -> None:
    from dotenv import load_dotenv

    here = Path(__file__).parent
    load_dotenv(here / ".env")  # 只讀本專案的 .env,不往上層目錄找
    workspace = Path(os.getenv("AGENT_WORKSPACE", "."))
    toolbox.configure(workspace if workspace.is_absolute() else here / workspace, os.getenv("SANDBOX_IMAGE"))
    ap = argparse.ArgumentParser(description="對內研發助理(NVIDIA NIM / API Catalog)")
    ap.add_argument("task")
    ap.add_argument("-f", "--file", action="append", default=[], help="附檔(相對於 AGENT_WORKSPACE)")
    ap.add_argument("--role", help="強制指定模型角色:code(日常)/ code_deep(難題)/ long_context")
    ap.add_argument("--think", choices=["on", "off"], help="思考模式;不給就用伺服器預設")
    ap.add_argument("--whole", action="store_true", help="整檔內嵌、不給工具、一次回答(寫文件 / 全檔審查用)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    registry = ModelRegistry.from_yaml()
    quota = TokenQuota(registry.limits["daily_tokens_per_user"], registry.limits["max_concurrent_per_user"])
    thinking = {"on": True, "off": False}.get(args.think)
    r = run_agent(args.task, registry=registry, quota=quota, attachments=args.file,
                  role=args.role, thinking=thinking, whole=args.whole)
    print(f"\n{r.answer}\n\n— {r.role} · {r.steps} 步 · {r.tokens_used:,} tokens")


if __name__ == "__main__":
    main()
