"""模型註冊表 + 路由。

換模型 = 改 models.yaml;換環境 = 改 LLM_BACKEND(catalog → nim)。程式碼一行都不用動。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from token_guard import ContextTooLarge


@dataclass(frozen=True)
class ModelSpec:
    role: str
    model: str
    context_window: int
    max_output: int
    supports_tools: bool = True
    supports_images: bool = False         # 看得懂圖片(routing.vision_role 用)
    nim_base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    timeout_s: int = 120
    extra_body: dict | None = None        # 每次都送的額外參數(例如 force_nonempty_content)
    thinking_on: dict | None = None
    thinking_off: dict | None = None
    fallbacks: tuple[str, ...] = ()
    available_on: tuple[str, ...] = ("catalog", "nim")

    def input_budget(self, margin: float) -> int:
        """prompt 能用的 token = 視窗 ×(1 - 安全邊際)- 預留給輸出的量。"""
        return int(self.context_window * (1 - margin)) - self.max_output


ChatFactory = Callable[..., BaseChatModel]
VISION_RETRIES = 2  # 看圖模型忙線時,換備援前先重試幾次(每次等 1–3 秒)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def nvidia_chat_factory(spec: ModelSpec, *, base_url: str, api_key: str | None,
                        model_kwargs: dict[str, Any]) -> BaseChatModel:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(
        model=spec.model,
        base_url=base_url,
        api_key=api_key,
        temperature=spec.temperature,
        top_p=spec.top_p,
        # 套件預設只有 1024,寫程式一定不夠,每個模型都要明確設定
        max_completion_tokens=spec.max_output,
        timeout=spec.timeout_s,
        model_kwargs=model_kwargs,
    )


class ModelRegistry:
    def __init__(self, config: dict, backend: str | None = None,
                 chat_factory: ChatFactory = nvidia_chat_factory):
        self.backend = backend or os.getenv("LLM_BACKEND", "catalog")
        self._backend_cfg = config["backends"][self.backend]
        self.routing: dict = config["routing"]
        self.limits: dict = config["limits"]
        self.embedding: dict | None = config.get("embedding")  # 知識庫用的 embedding 模型(knowledge.py)
        self.specs = {}
        for role, cfg in config["roles"].items():
            cfg = dict(cfg)
            nim_window = cfg.pop("nim_context_window", None)
            if self.backend == "nim" and nim_window:  # 地端的 NIM_MAX_MODEL_LEN 通常比 catalog 小
                cfg["context_window"] = nim_window
            self.specs[role] = ModelSpec(role=role, **{
                **cfg,
                "fallbacks": tuple(cfg.get("fallbacks", ())),
                "available_on": tuple(cfg.get("available_on", ("catalog", "nim"))),
            })
        self._chat_factory = chat_factory

    @classmethod
    def from_yaml(cls, path: str | Path = Path(__file__).with_name("models.yaml"), **kw) -> "ModelRegistry":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")), **kw)

    def spec(self, role: str) -> ModelSpec:
        if role not in self.specs:
            raise KeyError(f"沒有 {role} 這個角色,可用:{', '.join(self.specs)}")
        return self.specs[role]

    def budget(self, role: str) -> int:
        return self.spec(role).input_budget(self.routing["safety_margin"])

    # ---------- 建立模型 ----------

    def chat(self, role: str, *, thinking: bool | None = None, **overrides) -> BaseChatModel:
        """thinking=None 用伺服器預設;True/False 明確開關。overrides 暫時改 spec 欄位(例如分類器用短逾時、短輸出)。

        開關參數寫在 YAML 而不是靠套件的 with_thinking_mode():
        1. 地端 NIM 若用自訂的 served model name,套件查不到內建模型表,with_thinking_mode 會默默失效;
        2. with_thinking_mode 會整包覆蓋 chat_template_kwargs,把 extra_body 裡的其他鍵蓋掉。
        """
        spec = replace(self.spec(role), **overrides)
        if self.backend not in spec.available_on:
            raise ValueError(f"{role}({spec.model})在 {self.backend} 上沒有提供,只能用於:{', '.join(spec.available_on)}")
        if self.backend == "nim":
            base_url = spec.nim_base_url or self._backend_cfg["base_url"]
        else:
            base_url = self._backend_cfg["base_url"]
        key_env = self._backend_cfg.get("api_key_env")
        api_key = os.getenv(key_env) if key_env else None
        if key_env and not api_key:
            raise RuntimeError(f"缺少環境變數 {key_env}")
        toggle = {True: spec.thinking_on, False: spec.thinking_off}.get(thinking) or {}
        extra = _deep_merge(spec.extra_body or {}, toggle)
        return self._chat_factory(spec, base_url=base_url, api_key=api_key, model_kwargs=extra)

    def agent_model(self, role: str, tools: list, *, thinking: bool | None = None) -> Runnable:
        """agent 主腦:綁工具 + 自動備援(被限流或掛掉時換下一顆)。tools 為空 = 單次問答,不綁工具。"""
        spec = self.spec(role)
        if tools and not spec.supports_tools:
            raise ValueError(f"{role}({spec.model})不支援 tool calling,不能當 agent 主腦")

        def build(r: str) -> Runnable:
            model = self.chat(r, thinking=thinking)
            return model.bind_tools(tools) if tools else model

        primary = build(role)
        backups = [build(fb) for fb in spec.fallbacks if not tools or self.spec(fb).supports_tools]
        return primary.with_fallbacks(backups) if backups else primary

    def embedder(self) -> Callable[[list[str], str], tuple[list[list[float]], int]] | None:
        """知識庫用:embed(文字們, "passage" | "query") → (向量們, 用掉的 tokens);models.yaml 沒有 embedding 就回 None。
        直接打 /v1/embeddings 而不用套件的 NVIDIAEmbeddings:10/9 實測同一顆模型經過它回 502、直接打正常。
        忙線(429 / 5xx)等一下再送,最多 3 次。"""
        cfg = self.embedding
        if not cfg:
            return None
        base = (cfg.get("nim_base_url") or self._backend_cfg["base_url"]) if self.backend == "nim" else self._backend_cfg["base_url"]
        key_env = self._backend_cfg.get("api_key_env")

        def embed(texts: list[str], kind: str) -> tuple[list[list[float]], int]:
            import httpx

            api_key = os.getenv(key_env) if key_env else None
            if key_env and not api_key:
                raise RuntimeError(f"缺少環境變數 {key_env}")
            body = {"model": cfg["model"], "input": texts, "input_type": kind, "truncate": "END"}
            for attempt in range(3):
                r = httpx.post(base.rstrip("/") + "/embeddings", json=body, timeout=cfg.get("timeout_s", 60),
                               headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
                if r.status_code in (429, 500, 502, 503) and attempt < 2:
                    time.sleep(1 + attempt * 2)
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"[{r.status_code}] {r.text[:200]}")
                data = r.json()
                vectors = [d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0))]
                return vectors, (data.get("usage") or {}).get("total_tokens", 0)
            raise AssertionError("unreachable")

        embed.model = cfg["model"]  # 換了模型,知識庫要全部重算
        return embed

    def vision_model(self, role: str) -> Runnable:
        """看圖:關思考(10/9 實測 Nano Omni 開思考慢 5 倍、多花上千 tokens,看圖說明用不到)。
        免費層常回 503 忙線(10/9 實測約一半),等一下再送通常就過:每顆先重試 VISION_RETRIES 次再換備援。"""
        def build(r: str) -> Runnable:
            model = self.chat(r, thinking=False)
            return model.with_retry(stop_after_attempt=1 + VISION_RETRIES) if VISION_RETRIES else model

        backups = [build(fb) for fb in self.spec(role).fallbacks if self.spec(fb).supports_images]
        return build(role).with_fallbacks(backups) if backups else build(role)

    # ---------- 路由 ----------

    def pick_role(self, *, prompt_tokens: int, needs_tools: bool = True, requested: str | None = None) -> str:
        """指定的角色放得下就用它,放不下自動升級到長文本模型;都放不下 → ContextTooLarge。"""
        first = requested or self.routing["default_role"]
        if needs_tools and not self.spec(first).supports_tools:
            raise ValueError(f"{first}({self.spec(first).model})不支援 tool calling")
        for role in dict.fromkeys([first, self.routing["long_context_role"]]):
            if prompt_tokens <= self.budget(role):
                return role
        raise ContextTooLarge(
            f"約 {prompt_tokens:,} tokens,超過最大模型的可用預算 "
            f"{self.budget(self.routing['long_context_role']):,};請改走離線批次分析"
        )
