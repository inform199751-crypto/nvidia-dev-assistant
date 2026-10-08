"""離線測試:不需要 API key、不連網,用假模型驗證路由、迴圈與 token 防護。"""
from __future__ import annotations

import copy
import logging
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import tools
from agent import SYSTEM_PROMPT, build_user_prompt, run_agent
from llm_registry import ModelRegistry
from token_guard import (ContextTooLarge, QuotaExceeded, TokenQuota, compact_history,
                         message_tokens, outline, strip_stale_reasoning, tool_tokens, truncate_middle)

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


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    return tmp_path


# ---------- 路由 / 動態切換 ----------

def test_small_prompt_stays_on_code_model():
    reg, _ = scripted_registry([])
    assert reg.pick_role(prompt_tokens=50_000) == "code"


def test_large_prompt_escalates_to_long_context():
    reg, _ = scripted_registry([])
    assert reg.pick_role(prompt_tokens=300_000) == "long_context"


def test_beyond_long_context_raises():
    reg, _ = scripted_registry([])
    with pytest.raises(ContextTooLarge):
        reg.pick_role(prompt_tokens=2_000_000)


def test_tool_less_model_cannot_drive_agent():
    reg, _ = scripted_registry([])
    with pytest.raises(ValueError, match="tool calling"):
        reg.pick_role(prompt_tokens=100, requested="code_writer")
    assert reg.pick_role(prompt_tokens=100, requested="code_writer", needs_tools=False) == "code_writer"


def test_real_chatnvidia_nim_backend_builds_offline():
    reg = ModelRegistry(copy.deepcopy(CONFIG), backend="nim")
    llm = reg.chat("code", thinking=False)
    assert llm.model == "deepseek-ai/deepseek-v4.1-flash"
    assert llm.base_url == "http://nim-deepseek:8000/v1"
    assert llm.max_tokens == 8192
    assert llm.model_kwargs == {"chat_template_kwargs": {"thinking": False}}


def test_thinking_toggle_merges_with_extra_body():
    reg = ModelRegistry(copy.deepcopy(CONFIG), backend="nim")
    on = reg.chat("code_backup", thinking=True).model_kwargs
    assert on == {"chat_template_kwargs": {"force_nonempty_content": True, "enable_thinking": True}}
    assert reg.chat("code_backup").model_kwargs == {"chat_template_kwargs": {"force_nonempty_content": True}}


def test_payload_on_the_wire(monkeypatch):
    """確認真的送出去的 JSON:max_tokens 與 chat_template_kwargs 都在。"""
    reg = ModelRegistry(copy.deepcopy(CONFIG), backend="nim")
    llm = reg.chat("code_backup", thinking=True)
    payload = llm._get_payload(inputs=[{"role": "user", "content": "hi"}], stop=None)
    assert payload["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert payload["max_tokens"] == 8192
    assert payload["chat_template_kwargs"] == {"force_nonempty_content": True, "enable_thinking": True}


def test_nim_only_model_rejected_on_catalog(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-dummy")
    reg = ModelRegistry(copy.deepcopy(CONFIG), backend="catalog")
    with pytest.raises(ValueError, match="只能用於:nim"):
        reg.chat("code_writer")


def test_real_chatnvidia_catalog_backend(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-dummy")
    reg = ModelRegistry(copy.deepcopy(CONFIG), backend="catalog")
    llm = reg.chat("long_context")
    assert llm.model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert llm.base_url == "https://integrate.api.nvidia.com/v1"
    assert llm.model_kwargs == {}


def test_catalog_without_key_fails_fast(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    reg = ModelRegistry(copy.deepcopy(CONFIG), backend="catalog")
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        reg.chat("code")


def test_agent_model_has_fallback_chain():
    reg = ModelRegistry(copy.deepcopy(CONFIG), backend="nim")
    runnable = reg.agent_model("code", tools.ALL_TOOLS)
    assert [f.bound.model for f in runnable.fallbacks] == ["nvidia/nemotron-3-super-120b-a12b"]


# ---------- agent 迴圈 ----------

def test_agent_reads_file_then_answers(workspace):
    (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8", newline="\n")
    reg, seen = scripted_registry([
        ai(tool_calls=[{"name": "read_file", "args": {"path": "calc.py"}, "id": "c1"}]),
        ai("calc.py:2 把加號寫成減號"),
    ])
    r = run_agent("add 結果錯了", registry=reg, quota=TokenQuota(10**6, 2))
    assert r.answer == "calc.py:2 把加號寫成減號"
    assert (r.role, r.steps, r.tokens_used) == ("code", 2, 200)
    tool_msg = seen[1][1][-1]
    assert isinstance(tool_msg, ToolMessage) and "return a - b" in tool_msg.content


def test_tool_error_is_fed_back_not_raised(workspace):
    reg, seen = scripted_registry([
        ai(tool_calls=[{"name": "read_file", "args": {"path": "../../etc/passwd"}, "id": "c1"}]),
        ai("好,我不讀工作區以外的檔案"),
    ])
    run_agent("讀密碼檔", registry=reg, quota=TokenQuota(10**6, 2))
    assert "超出工作區" in seen[1][1][-1].content


def test_whole_mode_routes_big_file_to_long_context(workspace):
    (workspace / "big.py").write_text("x = 1\n" * 150_000, encoding="utf-8")  # 加行號後約 70 萬 tokens
    reg, seen = scripted_registry([ai("這是整份檔案的架構說明")])
    r = run_agent("寫技術文件", registry=reg, quota=TokenQuota(10**7, 2), attachments=["big.py"], whole=True)
    assert r.role == "long_context" and seen[0][0] == "long_context"


def test_focused_mode_sends_outline_not_full_text(workspace):
    body = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(3000))
    (workspace / "big.py").write_text(body, encoding="utf-8")
    reg, seen = scripted_registry([ai("看完大綱了")])
    r = run_agent("f1500 在哪", registry=reg, quota=TokenQuota(10**6, 2), attachments=["big.py"])
    prompt = seen[0][1][1].content
    assert r.role == "code" and "<file_outline" in prompt and "L4501-4502  def f1500" in prompt


def test_agent_escalates_mid_run_when_history_grows(workspace):
    cfg = copy.deepcopy(CONFIG)
    cfg["roles"]["code"].update(context_window=6000, max_output=1000)
    cfg["limits"]["tool_output_max_tokens"] = 5000
    (workspace / "a.py").write_text("y = 2  # padding padding padding\n" * 1500, encoding="utf-8")
    reg, seen = scripted_registry([
        ai(tool_calls=[{"name": "read_file", "args": {"path": "a.py", "end_line": 100}, "id": "c1"}]),
        ai(tool_calls=[{"name": "read_file", "args": {"path": "a.py", "start_line": 101, "end_line": 400}, "id": "c2"}]),
        ai("讀完了"),
    ], cfg)
    r = run_agent("讀 a.py", registry=reg, quota=TokenQuota(10**6, 2))
    assert [s[0] for s in seen] == ["code", "code", "long_context"]
    assert r.role == "long_context"


# ---------- token 防護 ----------

def test_truncate_middle_keeps_head_and_tail():
    text = "\n".join(f"line {i}" for i in range(2000)) + "\nTraceback: KeyError"
    out = truncate_middle(text, 300)
    assert out.startswith("line 0") and out.endswith("KeyError") and "中間截斷" in out
    assert message_tokens([HumanMessage(out)]) < 340


def test_stale_reasoning_stripped_current_turn_kept():
    msgs = [SystemMessage("s"), HumanMessage("q1"), ai("<think>舊的思考</think>a1", reasoning="舊的思考"),
            HumanMessage("q2"), ai("", tool_calls=[{"name": "x", "args": {}, "id": "t"}], reasoning="本輪思考")]
    out = strip_stale_reasoning(msgs)
    assert out[2].content == "a1" and "reasoning_content" not in out[2].additional_kwargs
    assert out[4].additional_kwargs["reasoning_content"] == "本輪思考"
    assert msgs[2].additional_kwargs["reasoning_content"] == "舊的思考"  # 原物件不被改到


def test_compact_history_keeps_tool_pairs():
    big = "z" * 30_000
    msgs = [HumanMessage("q")]
    for i in range(4):
        msgs += [ai(tool_calls=[{"name": "read_file", "args": {}, "id": f"t{i}"}]),
                 ToolMessage(big, tool_call_id=f"t{i}")]
    out = compact_history(msgs, budget=25_000)
    assert len(out) == len(msgs)
    assert [m.tool_call_id for m in out if isinstance(m, ToolMessage)] == ["t0", "t1", "t2", "t3"]
    assert "已省略" in out[2].content and out[-1].content == big
    assert message_tokens(out) <= 25_000


def test_outline_python_and_fallback():
    src = "class A:\n    def m(self):\n        pass\n\ndef g():\n    pass\n"
    assert outline(src, "a.py").splitlines() == ["L1-3  class A", "L2-3  def m", "L5-6  def g"]
    assert "function go" in outline("x\nfunction go() {}\n", "a.js")


def test_quota_daily_and_concurrency():
    q = TokenQuota(daily_tokens=1000, max_concurrent=1)
    q.charge("amy", 900)
    with pytest.raises(QuotaExceeded):
        q.check("amy", 200)
    q.check("bob", 200)
    with q.slot("amy"):
        with pytest.raises(QuotaExceeded):
            with q.slot("amy"):
                pass
    with q.slot("amy"):
        pass


@pytest.fixture
def fake_sandbox(workspace, monkeypatch):
    """假沙箱:記下它看到的複本(路徑 → bytes),不真的啟動 Docker。"""
    seen = []

    def run(root, target):
        assert root.resolve() != workspace.resolve()  # 測的是複本,不是工作區
        seen.append({p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*.py")})
        return "exit code 0\n1 passed"

    monkeypatch.setattr(tools, "_run_sandbox", run)
    return seen


def patch(*edits, reason="修正", test_target="."):
    return tools.propose_patch.invoke({"edits": [dict(zip(("path", "old_string", "new_string"), e)) for e in edits],
                                       "reason": reason, "test_target": test_target})


def test_propose_patch_builds_diff_from_edits(workspace, fake_sandbox):
    """模型手寫 unified diff 常格式錯(10/6 兩次、10/8 三次被退回);改成給舊字串 → 新字串,diff 由工具產生。"""
    (workspace / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a - b\n")
    (workspace / "util.py").write_bytes(b"X = 1\n")
    out = patch(("calc.py", "def add(a, b):\n    return a - b", "def add(a, b):\n    return a + b"),
                ("calc.py", "def mul(a, b):\n    return a - b", "def mul(a, b):\n    return a * b"),
                ("util.py", "X = 1", "X = 2"))
    assert "已存成 .agent_patches/" in out and "+    return a + b" in out and "1 passed" in out
    assert fake_sandbox == [{"calc.py": b"def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n",
                             "util.py": b"X = 2\n"}]
    saved = next((workspace / ".agent_patches").glob("*.patch"))
    assert saved.name.endswith("_calc.py+1.patch")
    assert (workspace / "calc.py").read_bytes().endswith(b"return a - b\n")  # 原檔沒被改


def test_propose_patch_rejects_bad_edits_without_running_tests(workspace, fake_sandbox):
    (workspace / "calc.py").write_text("def f():\n    return 1\n\ndef g():\n    return 1\n", encoding="utf-8")
    assert "找不到 old_string" in patch(("calc.py", "    1| def f():", "def h():"))  # 帶了行號前綴
    assert "出現 2 次" in patch(("calc.py", "    return 1", "    return 2"))
    assert "沒有任何變更" in patch(("calc.py", "def f():", "def f():"))
    with pytest.raises(PermissionError):
        patch((".env", "A=1", "A=2"))
    assert fake_sandbox == [] and not list(workspace.glob(".agent_patches/*.patch"))


def test_propose_patch_keeps_crlf_and_missing_final_newline(workspace, fake_sandbox):
    """read_file 給模型看的是 \\n:CRLF 檔要能比對、只改到那一行;沒有結尾換行的檔案 git apply 也要吃得下。"""
    (workspace / "win.py").write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    (workspace / "tail.py").write_bytes(b"x = 1\ny = 2")
    out = patch(("win.py", "a = 1\nb = 2", "a = 1\nb = 20"), ("tail.py", "y = 2", "y = 3"))
    assert "已存成" in out
    assert fake_sandbox == [{"win.py": b"a = 1\r\nb = 20\r\nc = 3\r\n", "tail.py": b"x = 1\ny = 3"}]


def test_shipped_models_yaml_is_consistent():
    reg = ModelRegistry.from_yaml(backend="nim")
    assert {reg.routing["default_role"], reg.routing["long_context_role"]} <= set(reg.specs)
    for spec in reg.specs.values():
        assert set(spec.fallbacks) <= set(reg.specs), spec.role
        assert reg.budget(spec.role) > 0, spec.role
    assert reg.spec(reg.routing["default_role"]).supports_tools
    assert reg.spec(reg.routing["long_context_role"]).supports_tools


def test_shipped_fallbacks_can_hold_what_primary_accepts():
    """備援模型的上下文不能比主模型小,不然主模型收下的大請求,一退到備援就爆。"""
    reg = ModelRegistry.from_yaml(backend="nim")
    for spec in reg.specs.values():
        for fb in spec.fallbacks:
            assert reg.budget(fb) >= reg.budget(spec.role), f"{spec.role} → {fb}"
            assert reg.spec(fb).supports_tools or not spec.supports_tools, f"{spec.role} → {fb}"


def test_shipped_long_context_has_fallback():
    reg = ModelRegistry.from_yaml(backend="catalog")
    assert reg.spec("long_context").fallbacks, "Ultra 免費層會 503,長文本一定要有備援"


def test_orphan_closing_think_tag_is_stripped():
    """Ultra 實測回傳過:推理 + '</think>' + 答案,沒有開頭標籤。"""
    from token_guard import strip_think_text
    leaked = 'The user wants a two-character response: "收到".</think>收到'
    assert strip_think_text(leaked) == "收到"
    assert strip_think_text("<think>想一下</think>\n收到") == "收到"
    assert strip_think_text("收到") == "收到"
    msgs = [HumanMessage("q1"), ai(leaked, reasoning="x"), HumanMessage("q2")]
    assert strip_stale_reasoning(msgs)[1].content == "收到"


def test_agent_final_answer_has_no_leaked_reasoning(workspace):
    reg, _ = scripted_registry([ai('推理過程……</think>calc.py:2 寫成減號了')])
    r = run_agent("add 錯了", registry=reg, quota=TokenQuota(10**6, 2))
    assert r.answer == "calc.py:2 寫成減號了"


def test_configure_updates_workspace_after_dotenv(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tools.WORKSPACE)
    monkeypatch.setattr(tools, "SANDBOX_IMAGE", tools.SANDBOX_IMAGE)
    (tmp_path / "x.py").write_text("print(1)\n", encoding="utf-8")
    tools.configure(tmp_path, "my-image")
    assert tools.WORKSPACE == tmp_path.resolve() and tools.SANDBOX_IMAGE == "my-image"
    assert "print(1)" in tools.read_file.invoke({"path": "x.py"})


def test_secret_files_are_blocked(workspace):
    (workspace / ".env").write_text("NVIDIA_API_KEY=nvapi-secret\n", encoding="utf-8")
    (workspace / ".env.example").write_text("NVIDIA_API_KEY=\n", encoding="utf-8")
    (workspace / "server.key").write_text("nvapi-secret\n", encoding="utf-8")
    (workspace / "app.py").write_text("key = load('nvapi-secret')\n", encoding="utf-8")
    for path in [".env", "server.key", ".git/config"]:
        with pytest.raises(PermissionError):
            tools.read_file.invoke({"path": path})
    assert "NVIDIA_API_KEY=" in tools.read_file.invoke({"path": ".env.example"})
    hits = tools.search_code.invoke({"pattern": "nvapi-secret", "glob": "**/*"})
    assert hits.startswith("app.py:1:") and "server.key" not in hits and ".env" not in hits


def _dir_link(link, target):
    """目錄連結:能建 symlink 就用 symlink;Windows 一般帳號不能,改用不需權限的 junction。"""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))


def test_search_code_does_not_follow_links_out_of_workspace(workspace, tmp_path_factory):
    """工作區裡的 symlink / junction 指到外面時,read_file 會擋,search_code 也要擋。"""
    outside = tmp_path_factory.mktemp("outside")
    (outside / "leak.py").write_text("TOKEN = 'nvapi-secret'\n", encoding="utf-8")
    (workspace / "ok.py").write_text("x = 'nvapi-secret'\n", encoding="utf-8")
    _dir_link(workspace / "vendor", outside)
    for glob in ["**/*.py", "vendor/*.py"]:
        hits = tools.search_code.invoke({"pattern": "nvapi-secret", "glob": glob})
        assert "vendor" not in hits, glob
    assert tools.search_code.invoke({"pattern": "nvapi-secret"}).splitlines() == ["ok.py:1: x = 'nvapi-secret'"]
    with pytest.raises(ValueError, match="超出工作區"):
        tools.read_file.invoke({"path": "vendor/leak.py"})


def test_nim_backend_uses_nim_context_window():
    """catalog 給 Ultra 1M,地端預設 256K;切到 nim 時路由要照地端的長度算,不然送出去才被伺服器拒絕。"""
    catalog = ModelRegistry.from_yaml(backend="catalog")
    nim = ModelRegistry.from_yaml(backend="nim")
    assert catalog.spec("long_context").context_window == 1_048_576
    assert nim.spec("long_context").context_window == 262_144
    assert nim.spec("code").context_window == catalog.spec("code").context_window  # 沒寫 nim_context_window 就沿用
    assert catalog.pick_role(prompt_tokens=500_000) == "long_context"
    with pytest.raises(ContextTooLarge):
        nim.pick_role(prompt_tokens=500_000)


def test_agent_rejects_secret_attachment(workspace):
    (workspace / ".env").write_text("NVIDIA_API_KEY=nvapi-secret\n", encoding="utf-8")
    reg, _ = scripted_registry([ai("不該被呼叫")])
    with pytest.raises(PermissionError):
        run_agent("看設定", registry=reg, quota=TokenQuota(10**6, 2), attachments=[".env"])


def test_whole_mode_binds_no_tools(workspace, monkeypatch):
    """整檔模式:檔案全附上、不給工具,避免模型分段重讀而每步重送全部內容(10/6 實測踩到)。"""
    (workspace / "a.py").write_text("x = 1", encoding="utf-8")

    def no_tools(self, tools, **kw):
        raise AssertionError("整檔模式不該綁工具")

    monkeypatch.setattr(ScriptedChat, "bind_tools", no_tools)
    reg, seen = scripted_registry([ai("文件內容")])
    r = run_agent("寫文件", registry=reg, quota=TokenQuota(10**6, 2), attachments=["a.py"], whole=True)
    assert r.answer == "文件內容" and r.steps == 1
    assert "沒有工具可用" in seen[0][1][1].content


def test_underestimate_is_corrected_from_usage(workspace):
    """伺服器回報的 input tokens 比估算多很多 → 之後按實際比例算預算,提早升級。"""
    cfg = copy.deepcopy(CONFIG)
    cfg["roles"]["code"].update(context_window=8000, max_output=1000)
    (workspace / "a.py").write_text("y = 1", encoding="utf-8")
    first = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "c1"}],
                      usage_metadata={"input_tokens": 50_000, "output_tokens": 10, "total_tokens": 50_010})
    reg, seen = scripted_registry([first, ai("好了")], cfg)
    r = run_agent("看 a.py", registry=reg, quota=TokenQuota(10**7, 2))
    assert [s[0] for s in seen] == ["code", "long_context"] and r.role == "long_context"


def test_quota_is_checked_every_step(workspace):
    (workspace / "a.py").write_text("y = 1", encoding="utf-8")
    first = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "c1"}],
                      usage_metadata={"input_tokens": 1000, "output_tokens": 0, "total_tokens": 995_000})
    reg, seen = scripted_registry([first, ai("不該跑到這裡")])
    with pytest.raises(QuotaExceeded):
        run_agent("看 a.py", registry=reg, quota=TokenQuota(10**6, 2))
    assert len(seen) == 1


def test_tool_definitions_do_not_inflate_estimate(workspace, caplog):
    """工具定義是每步固定的成本,要算進估算;不能被當成「估算偏低」把之後每一步都放大(10/8 實測被放大 1.99 倍)。"""
    assert tool_tokens(tools.ALL_TOOLS) > 300 and tool_tokens([]) == 0
    (workspace / "a.py").write_text("y = 1", encoding="utf-8")
    prompt = [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_user_prompt("看 a.py", [], 2000, False))]
    actual = message_tokens(prompt) + tool_tokens(tools.ALL_TOOLS)  # 伺服器回報的 = 對話 + 工具定義
    first = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "c1"}],
                      usage_metadata={"input_tokens": actual, "output_tokens": 10, "total_tokens": actual + 10})
    reg, _ = scripted_registry([first, ai("好了")])
    with caplog.at_level(logging.INFO, logger="dev_assistant"):
        run_agent("看 a.py", registry=reg, quota=TokenQuota(10**6, 2))
    assert "估算偏低" not in caplog.text


def test_malformed_tool_call_is_fed_back_not_final(workspace):
    """參數不是合法 JSON 的 tool call 會進 invalid_tool_calls、tool_calls 是空的;不能當成最終答案回傳空字串。"""
    bad = AIMessage(content="", invalid_tool_calls=[{"name": "read_file", "args": '{"path": "calc.py"', "id": "c1",
                                                     "error": "Expecting ',' delimiter", "type": "invalid_tool_call"}],
                    usage_metadata={"input_tokens": 100, "output_tokens": 0, "total_tokens": 100})
    reg, seen = scripted_registry([bad, ai("修正後的回答")])
    r = run_agent("看 calc.py", registry=reg, quota=TokenQuota(10**6, 2))
    assert (r.answer, r.steps) == ("修正後的回答", 2)
    feedback = seen[1][1][-1]
    assert isinstance(feedback, ToolMessage) and feedback.tool_call_id == "c1"
    assert "不是合法 JSON" in feedback.content and "delimiter" in feedback.content
