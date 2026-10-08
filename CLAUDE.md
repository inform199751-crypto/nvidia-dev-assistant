# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案是什麼

企業內部的程式碼助理 agent(找 bug、提 patch、寫技術文件),走 NVIDIA NIM:開發期打 build.nvidia.com(`LLM_BACKEND=catalog`),正式環境換地端 NIM 容器(`LLM_BACKEND=nim`)。重點是依 token 數動態切換模型、控制 token / 顯存、agent 只能在沙箱裡動手。程式註解、文件、agent 回答都用繁體中文;`agent.SYSTEM_PROMPT` 刻意用英文寫(再指定回答用 zh-TW)。

## 常用指令(Windows,用專案內的 `.venv`)

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
docker build -t dev-assistant-sandbox sandbox            # run_tests / propose_patch 的沙箱,需 Docker Desktop 執行中
.venv/Scripts/python -m pytest -q                        # 離線測試:假模型,不需金鑰、不連網
.venv/Scripts/python -m pytest -q tests/test_offline.py::test_propose_patch_can_add_new_file   # 單一測試
.venv/Scripts/python smoke_test.py                       # 對 catalog 真模型做冒煙測試,結果附加到 smoke_results.log
.venv/Scripts/python smoke_test.py --roles long_context --checks think_off tool   # 只重跑失敗的項目
.venv/Scripts/python agent.py                            # 互動模式(@檔案 附檔、/clear、/exit)
.venv/Scripts/python agent.py "test_add 失敗了,找出 bug 並提出修正" -f calc.py
.venv/Scripts/python agent.py "寫技術文件" -f a.py -f b.py --whole      # 整檔模式
.venv/Scripts/python agent.py "..." --role code_deep --think on
```

- 沒有 linter / formatter 設定。
- `.env`(從 `.env.example` 複製)放 `NVIDIA_API_KEY`、`AGENT_WORKSPACE`(預設 `demo_workspace`)、`SANDBOX_IMAGE`。不要讀取或印出 `.env`。
- agent 執行時套件會對不在內建表的模型(DeepSeek v4.1)一直警告,設 `PYTHONWARNINGS=ignore` 可消音,不影響功能。
- 免費層的 Ultra 會間歇回 503 / 500,冒煙測試失敗先只重跑那幾項。

## 架構

`agent.AgentSession` 保存一段對話(messages、估算倍率 `scale`、累計 tokens);`ask()` 是一輪,中途出錯會把這一輪的訊息退掉。`run_agent()` = 開新 session 只問一輪;`chat()` 是互動模式的 REPL,問題裡的 `@路徑` 由 `file_mentions` 轉成附檔。每輪開始重新 `pick_role`,之後每一步送出前依序做:
1. `token_guard.strip_stale_reasoning`:只清「上一輪以前」的推理。同一輪內不清 —— 這是刻意的,模型接著 tool call 推理需要本輪的 reasoning。
2. 超過預算 → `compact_history`(把舊 ToolMessage 內容換成佔位字)。
3. 還放不下 → `registry.pick_role` 升級到 `long_context`;連它都放不下丟 `ContextTooLarge`。
4. `quota.check`(每一步都查,不是只在開始)。
5. `llm.invoke` 後用 `usage_metadata` 的實際 input tokens 校正 `scale`(只往上修)。

**token 估算**:`sized() = (message_tokens(messages) + tool_tokens(tools)) * scale`。工具定義(約 700 tokens)每步固定送出,必須算進估算;`scale` 只用來修正真正的估算誤差,不能拿工具成本去放大。`estimate_tokens` 的係數(中日韓字 1.5、其他 2.7 字元 / token)是用 Ultra 實測校準過的。

**模型註冊表**(`llm_registry.py` + `models.yaml`):角色 → 模型。`code`(Super,日常主腦)、`code_deep`(DeepSeek,慢但適合難題)、`long_context`(Ultra 1M)、`long_context_backup`、`code_writer`(Qwen,只有地端、不支援工具)。
- 預算 = `context_window × (1 − safety_margin) − max_output`。地端長度不同時寫 `nim_context_window`,`LLM_BACKEND=nim` 時取代 `context_window`。
- 備援用 `with_fallbacks`;備援模型的預算不能比主模型小(`test_shipped_fallbacks_can_hold_what_primary_accepts` 會檢查)。
- 思考開關寫在 YAML 的 `chat_template_kwargs`,不用套件的 `with_thinking_mode()`(地端自訂模型名會失效、而且會蓋掉 extra_body)。DeepSeek V4 的鍵是 `thinking`,Nemotron 是 `enable_thinking`。
- Ultra 不能送 `force_nonempty_content`(會把推理混進回答);`strip_think_text` 另外清孤立的 `</think>`。
- `ModelRegistry(chat_factory=...)` 可注入,離線測試就是靠這個換成假模型。

**工具**(`tools.py`):
- `WORKSPACE` 在模組載入時就讀環境變數,`agent.main()` 讀完 `.env` 後要呼叫 `tools.configure()` 才會生效。
- `_safe_path` 擋工作區外、機密檔(`.env*`、金鑰、憑證)、`.git/`;`list_files` / `search_code` 走 `_visible`,另外用 resolve 後的實際位置檢查(symlink / Windows junction)。
- `run_tests` / `run_command` 測的永遠是**原始碼**:都經過 `_sandbox_copy()`(排除機密檔、大型目錄、指到外面的連結的暫存複本)→ `_docker()`(無網路、唯讀掛載、限資源、逾時 `docker kill`)。沒有 Docker 就回錯誤,絕不在主機上直接跑。沙箱映像裡沒有 git。
- `git_log` / `git_blame` 在主機上跑唯讀的 git 指令,路徑一樣過 `_safe_path`。
- `propose_patch(edits, reason, test_target)`:每筆 edit 是 `path` / `old_string` / `new_string`(old_string 必須唯一;不存在的路徑 + 空 old_string = 新增檔案)。diff 由 difflib 產生 → `git apply --check` → 套到暫存複本(`git -c core.autocrlf=false apply`)→ 沙箱跑測試 → 存到 `WORKSPACE/.agent_patches/`。**工作區原檔永遠不改**;每個 patch 都是單獨套在原始檔案上驗證,不會疊加。

**`--whole` 模式**:附檔全部內嵌、不綁工具、一次回答,交給路由決定要不要升級到 Ultra。給工具的話模型會分段重讀,每步重送全部內容。

## 不能破壞的規則

- tool call 和 ToolMessage 必須成對:`compact_history` 只換內容不刪訊息;`invalid_tool_calls`(參數不是合法 JSON)也要回一則 ToolMessage —— 套件送回伺服器的是 `additional_kwargs["tool_calls"]` 原始內容,少回一則 API 會報錯。
- `requirements.txt` 鎖版本,因為程式依賴 langchain-nvidia-ai-endpoints / langchain-core 的內部行為(`_get_payload`、`_reasoning_api_fields`、推理內容回送)。升版後要跑離線測試和 `smoke_test.py`。
- catalog 上的模型 ID 汰換很快,改 `models.yaml` 前後都用 `smoke_test.py` 確認。
- **不要把公司真實程式碼送到 build.nvidia.com**:API Trial 條款 §1.2 禁止正式使用、§3.3 允許 NVIDIA 收集輸入輸出。只用 `demo_workspace/` 或可公開的程式碼做實測。

## 測試慣例

- 全部在 `tests/test_offline.py`。`scripted_registry(script, config)` 用照劇本回答的 `ScriptedChat`;`workspace` fixture 把 `tools.WORKSPACE` 換成 `tmp_path`;`fake_sandbox` 取代 `_run_sandbox` 並記錄複本內容;`patch(...)` 是呼叫 `propose_patch` 的捷徑。用到 `propose_patch` 的測試會呼叫真的 git。
- 從實測抓到的問題,測試 docstring 會寫發生的日期與現象,README 的「實測與審查抓到的問題」表也同步加一列、更新離線測試數量。新修一個問題時照同樣方式補。
- `.gitattributes` 強制 LF;Git for Windows 預設 `autocrlf=true`,處理換行時要注意。
