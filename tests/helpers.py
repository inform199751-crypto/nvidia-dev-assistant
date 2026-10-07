"""測試共用:假模型、測試設定、測試用執行器。"""
from __future__ import annotations

import copy
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from llm_registry import ModelRegistry

CONFIG = {
    "backends": {
        "catalog": {"base_url": "https://integrate.api.nvidia.com/v1", "api_key_env": "NVIDIA_API_KEY"},
        "nim": {"base_url": "http://localhost:8000/v1", "api_key_env": None},
    },
    "routing": {"default_role": "code", "long_context_role": "long_context", "safety_margin": 0.1},
    "limits": {"max_agent_steps": 6, "tool_output_max_tokens": 200, "inline_file_max_tokens": 2000,
               "daily_tokens_per_user": 1_000_000, "max_concurrent_per_user": 2},
    "roles": {
        "code": {"model": "deepseek-ai/deepseek-v4.1-flash", "context_window": 131072, "max_output": 8192,
                 "nim_base_url": "http://nim-deepseek:8000/v1", "fallbacks": ["code_backup"],
                 "thinking_on": {"chat_template_kwargs": {"thinking": True}},
                 "thinking_off": {"chat_template_kwargs": {"thinking": False}}},
        "code_backup": {"model": "nvidia/nemotron-3-super-120b-a12b", "context_window": 262144, "max_output": 8192,
                        "extra_body": {"chat_template_kwargs": {"force_nonempty_content": True}},
                        "thinking_on": {"chat_template_kwargs": {"enable_thinking": True}}},
        "code_writer": {"model": "qwen/qwen2.5-coder-32b-instruct", "context_window": 32768,
                        "max_output": 4096, "supports_tools": False, "available_on": ["nim"]},
        "long_context": {"model": "nvidia/nemotron-3-ultra-550b-a55b", "context_window": 1_000_000,
                         "max_output": 16384},
    },
}


class ScriptedChat(BaseChatModel):
    """照劇本回答的假模型;記錄每次收到的訊息與自己的角色。"""
    role: str = ""
    script: Any = None  # 用 Any,pydantic 才不會複製 list,多個角色共用同一份劇本
    seen: Any = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.seen.append((self.role, list(messages)))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"


def scripted_registry(script: list[AIMessage], config: dict = CONFIG):
    shared = {"script": script, "seen": []}

    def factory(spec, **_):
        return ScriptedChat(role=spec.role, script=shared["script"], seen=shared["seen"])

    return ModelRegistry(copy.deepcopy(config), backend="nim", chat_factory=factory), shared["seen"]


def ai(content="", tool_calls=(), tokens=100, reasoning=None):
    kw = {"reasoning_content": reasoning, "_reasoning_api_fields": ["reasoning_content"]} if reasoning else {}
    return AIMessage(content=content, tool_calls=list(tool_calls), additional_kwargs=kw,
                     usage_metadata={"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens})


class InlineExecutor:
    """測試用:submit 時直接在呼叫端執行,呼叫回來時這一輪已經跑完。"""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


class ManualExecutor:
    """測試用:先收著,測試自己決定什麼時候跑(用來測排隊、取消、忙碌)。"""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args, **kwargs):
        self.jobs.append((fn, args, kwargs))

    def run_all(self):
        while self.jobs:
            fn, args, kwargs = self.jobs.pop(0)
            fn(*args, **kwargs)
