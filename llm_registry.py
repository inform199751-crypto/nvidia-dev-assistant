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


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


BUSY_RETRIES = 2  # 伺服器回 429 / 500 / 502 / 503 時,同一顆模型先重試幾次再換備援(每次多等 1、2 秒)


def retrying_session(verify: bool = True):
    """會自己重試忙線的 requests.Session(換掉套件的 get_session_fn)。
    10/26 實測 3.5 Super VL:連續呼叫 20 次,前 9 次成功、之後 0.3 秒就回「500 Failed to generate completions」;
    隔幾秒再送大多就成功。以前一失敗 with_fallbacks 馬上換 DeepSeek(每步慢 3–4 倍),Super 該答的步數有 14% 被換掉。
    逾時(卡住不回)不重試:一次就是 timeout_s 秒,重試只會再卡一次,交給備援。
    連不到(找不到主機、拒絕連線)也不重試:10/27 地端沒部署看圖 NIM,每次附圖白等 38 秒;多等也不會好,直接交給備援。"""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.verify = verify
    retry = Retry(total=BUSY_RETRIES, connect=0, read=0, status=BUSY_RETRIES, backoff_factor=1,
                  status_forcelist=(429, 500, 502, 503), allowed_methods=None, raise_on_status=False)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def nvidia_chat_factory(spec: ModelSpec, *, base_url: str, api_key: str | None,
                        model_kwargs: dict[str, Any]) -> BaseChatModel:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    chat = ChatNVIDIA(
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
    # 套件內部的 client(requirements.txt 鎖版本;升版要確認還在):一般呼叫和串流都從 get_session_fn 拿 Session
    client = chat._client
    client.get_session_fn = lambda: retrying_session(client.verify_ssl)
    if api_key is None:  # 沒給金鑰時套件會自己讀環境變數 NVIDIA_API_KEY,地端會把試用平台的金鑰當 Bearer 送出去
        client.api_key = None
        # 標頭範本還固定帶「Bearer **********」佔位(有金鑰才換成真的),公司的 API Gateway 可能當成壞掉的憑證拒絕
        client.headers_tmpl = {kind: {k: v for k, v in headers.items() if k != "Authorization"}
                               for kind, headers in client.headers_tmpl.items()}
    return chat


class ModelRegistry:
    def __init__(self, config: dict, backend: str | None = None,
                 chat_factory: ChatFactory = nvidia_chat_factory):
        self.backend = backend or os.getenv("LLM_BACKEND", "catalog")
        self._backend_cfg = config["backends"][self.backend]
        self.routing: dict = config["routing"]
        self.limits: dict = config["limits"]
        self.embedding: dict | None = config.get("embedding")  # 知識庫用的 embedding 模型(knowledge.py)
        self.safety: dict | None = config.get("safety")  # 安全把關的模型(guard.py)
        self.image: dict | None = config.get("image")  # 畫圖的模型(imagegen.py)
        self.company_data_ok = bool(self._backend_cfg.get("company_data"))  # 這個後端能不能放公司資料(models.yaml)
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

    def image_generator(self) -> Callable[[str, int, int], tuple[bytes, str]] | None:
        """畫圖用(imagegen.py):draw(英文描述, 寬, 高) → (圖片, 用的模型);models.yaml 沒有 image、這個環境沒有就回 None。
        catalog 的畫圖模型不在 /v1/chat 那組網址,直接打 base_url/<模型>;逾時、忙線、404 換下一顆,
        被內容過濾擋下的不換(換了也一樣),丟 ValueError 讓模型跟使用者說。"""
        cfg = self.image
        if not cfg or self.backend not in cfg.get("available_on", ["catalog"]):
            return None
        base = (cfg.get("nim_base_url") or cfg["base_url"]) if self.backend == "nim" else cfg["base_url"]
        key_env = self._backend_cfg.get("api_key_env")

        def draw(prompt: str, width: int, height: int) -> tuple[bytes, str]:
            import base64
            import secrets

            import httpx

            api_key = os.getenv(key_env) if key_env else None
            headers = {"Accept": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
            errors = []
            for m in cfg["models"]:
                body = {"prompt": prompt, "width": width, "height": height, "seed": secrets.randbelow(2**31), **m.get("params", {})}
                try:
                    r = httpx.post(f"{base.rstrip('/')}/{m['model']}", json=body, headers=headers, timeout=cfg.get("timeout_s", 120))
                except httpx.HTTPError as e:
                    errors.append(f"{m['model']}:{type(e).__name__}")
                    continue
                if r.status_code != 200:
                    errors.append(f"{m['model']}:[{r.status_code}]")
                    continue
                art = (r.json().get("artifacts") or [{}])[0]
                if art.get("finishReason") not in (None, "SUCCESS"):
                    raise ValueError(f"被 {m['model']} 的內容過濾擋下了({art.get('finishReason')}),請換個描述")
                if art.get("base64"):
                    return base64.b64decode(art["base64"]), m["model"]
                errors.append(f"{m['model']}:沒有回傳圖片")
            raise RuntimeError("畫圖的模型都沒有成功:" + ";".join(errors))

        return draw

    def safety_checker(self) -> Callable[[str], tuple[str, str]] | None:
        """安全把關用(guard.py):check(文字) → ("safe" | "unsafe", 類別);models.yaml 沒有 safety 就回 None。
        回答格式各家不同,由 guard.parse 解析。逾時設短(預設 10 秒):這是多一道保險,不能讓每次上網都卡住。"""
        cfg = self.safety
        if not cfg:
            return None
        base = (cfg.get("nim_base_url") or self._backend_cfg["base_url"]) if self.backend == "nim" else self._backend_cfg["base_url"]
        key_env = self._backend_cfg.get("api_key_env")

        def check(text: str) -> tuple[str, str]:
            import httpx

            import guard

            api_key = os.getenv(key_env) if key_env else None
            body = {"model": cfg["model"], "messages": [{"role": "user", "content": text}], "max_tokens": 64, "temperature": 0}
            for attempt in range(2):
                r = httpx.post(base.rstrip("/") + "/chat/completions", json=body, timeout=cfg.get("timeout_s", 10),
                               headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
                if r.status_code in (429, 503) and attempt == 0:
                    time.sleep(1)
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"[{r.status_code}] {r.text[:120]}")
                return guard.parse(r.json()["choices"][0]["message"]["content"] or "")
            raise AssertionError("unreachable")

        check.model = cfg["model"]
        return check

    def vision_model(self, role: str) -> Runnable:
        """看圖:關思考(10/9 實測 Nano Omni 開思考慢 5 倍、多花上千 tokens,看圖說明用不到)。
        免費層常回 503 忙線(10/9 實測約一半),等一下再送通常就過:retrying_session 每顆先重試 BUSY_RETRIES 次再換備援。
        以前這裡另外包一層 with_retry,10/27 拿掉:忙線變成重試 9 次,逾時、連不到也照樣重等(地端沒部署看圖 NIM 白等 38 秒)。"""
        def build(r: str) -> Runnable:
            return self.chat(r, thinking=False)

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
