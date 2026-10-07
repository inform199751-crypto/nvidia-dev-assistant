# 研發助理網頁版 設計文件

日期:2026-10-07 · 狀態:待審

## 目標

把現有的指令列 agent 包成一個在本機執行的網頁 App,用於面試現場示範:
打開網頁、選程式碼、用中文問,當場看 agent 一步步找出 bug、提出經過沙箱測試的修正,並能接著追問。

## 需求決定

| # | 問題 | 決定 |
|---|---|---|
| 1 | 給誰用 | 只有使用者本人,在自己電腦執行;不做登入 |
| 2 | 程式碼來源 | 從設定檔的資料夾清單選,也可以上傳 zip 或檔案 |
| 3 | 修正結果 | 只顯示差異與沙箱測試結果,附「複製 git apply 指令」;網頁不修改任何檔案 |
| 4 | 長任務 | 背景執行,結果保留,關掉網頁再回來仍看得到 |
| 5 | 追問 | 支援;同一個對話記得前文 |
| — | 技術 | FastAPI + 單頁網頁(原生 JS)+ SSE 即時推送;結果存 JSON 檔,不用資料庫 |
| — | 版面 | 類 ChatGPT / Claude:左側對話紀錄、中間對話串、底部輸入框、右側欄顯示長內容 |

## 不在這一版

登入與多人、Redis 配額、上雲、「套用 patch」按鈕、刪除或編輯對話、Git 平台整合、
**回答逐字顯示**(第一版顯示每一步的工具動作,但最後回答在模型回完後一次出現;
tool calling 搭配逐字串流的處理較複雜,列為第二版)。

## 架構

```
web/
  __init__.py
  __main__.py        python -m web → uvicorn 只聽 127.0.0.1:8765
  app.py             FastAPI 路由
  conversations.py   對話與每一輪的執行、事件、存檔、取消、補送
  workspaces.py      資料夾清單、上傳處理、過期清理
  static/
    index.html  app.js  style.css
    vendor/     marked.min.js、purify.min.js(放進 repo,現場沒網路也能排版)
workspaces.yaml      可選資料夾清單
conversations/       對話存檔(git 忽略)
uploads/             上傳的暫存複本(git 忽略)
```

### 既有程式碼的修改

1. **工具改成每個任務一份**:`tools.make_tools(workspace: Path, sandbox_image: str) -> list[BaseTool]`,
   每個工具用閉包綁定自己的資料夾。現在的全域 `WORKSPACE` 在網頁同時跑兩個不同資料夾的任務時會互相覆蓋。
   指令列改用 `make_tools`,行為不變。
2. **`run_agent` 增加三個參數**:
   - `on_event: Callable[[dict], None] | None`:每一步回報事件(格式見下)。
   - `history: list[BaseMessage] | None`:追問時帶入前文。
   - `should_stop: Callable[[], bool] | None`:每步送出前檢查,為 True 就停止。
   回傳的 `AgentResult` 增加 `messages`(完整訊息串),供下一輪追問與存檔。
3. **追問不重送原始檔案**:新函式 `omit_attachments(messages)`,把前文 HumanMessage 裡的
   `<file path="…">…</file>` 換成 `<file path="…" omitted="true"/>` 加一行「需要時用 read_file 讀」。
   整份文件模式的回答本身保留在前文裡。

## 資料夾

`workspaces.yaml`(路徑相對於專案資料夾):

```yaml
workspaces:
  - name: demo
    path: demo_workspace
  - name: 客服專案
    path: ../universal-ai-helpdesk
```

上傳:
- 接受一個 `.zip`,或多個一般檔案。總大小 ≤ 20 MB、檔案數 ≤ 2000、單檔 ≤ 2 MB。
- 解壓時拒絕整包的情況:絕對路徑、含 `..`、符號連結、超過上限。
- 丟掉不拒收的檔案:`.env`、金鑰與憑證(沿用 `tools._is_secret`)、`.git/`;回應中列出被丟掉的檔名。
- 存到 `uploads/<編號>/`,在清單中顯示為「上傳:<檔名>」。24 小時後刪除(啟動時與每小時檢查一次)。
- 網頁標示「這是複本,patch 不能直接套回你的專案」。

## 對話與執行

- 一個對話綁定一個資料夾,第一則訊息送出後不能更換。
- 每一則訊息是一「輪」(turn)。每輪模式二選一(API 值 `fix` / `whole`):
  - **問問題 / 修 bug**(`fix`):agent 用工具自己找;可在「進階」選填相關檔案路徑(逗號分隔),會直接附上。
  - **整份文件**(`whole`):附上資料夾內所有文字檔(`git ls-files`,非 git 資料夾則掃描常見程式與文件副檔名),
    排除機密檔;送出前顯示預估 tokens 與將使用的模型,超過長文本上限就不讓送出。
- 執行:`ThreadPoolExecutor(max_workers=2)`,第 3 輪起狀態為「排隊中」。
- 輪的狀態:`queued` → `running` → `done` / `failed` / `cancelled`;伺服器重開時,`queued` / `running` 改為 `interrupted`。

### 事件

每個事件:`{"seq": 整數, "t": ISO 時間, "type": …, …}`。

| type | 欄位 | 畫面 |
|---|---|---|
| `start` | role, model, mode, est_tokens | 「Super 120B · 預估 1.2 萬 tokens」 |
| `tool_call` | step, name, args | 「read_file calc.py」 |
| `tool_result` | step, name, ok, preview(最多 4000 字) | 結果標籤;點開看全文 |
| `model_switch` | from_role, to_role, reason | 「內容太長,改用 Ultra」 |
| `patch` | path, diff, test_output, passed | 差異卡片與測試標籤 |
| `answer` | text | 回答(Markdown) |
| `error` | kind, message | 說明加「重試」 |
| `end` | status, steps, tokens, seconds | 「4 步 · 18 秒」 |

`patch` 事件由 `propose_patch` 的參數(diff)和結果(測試輸出)組成。

### 存檔

`conversations/<編號>.json`:

```json
{
  "id": "…", "title": "第一則訊息前 30 字", "workspace": "demo", "created_at": "…",
  "turns": [{"id": "…", "question": "…", "mode": "fix", "options": {"role": null, "thinking": null},
             "status": "done", "events": [ … ]}],
  "messages": [ langchain messages_to_dict 的結果 ]
}
```

每收到一個事件就寫一次,先寫暫存檔再取代,避免寫到一半損壞。

### API

| 方法 | 路徑 | 用途 |
|---|---|---|
| GET | `/` | 網頁 |
| GET | `/api/workspaces` | 清單 + 未過期的上傳 |
| POST | `/api/uploads` | 上傳(multipart) |
| GET | `/api/estimate?workspace=` | 整份文件模式的預估 tokens 與模型 |
| GET | `/api/conversations` | 對話列表(摘要) |
| POST | `/api/conversations` | 新對話 + 第一輪 |
| GET | `/api/conversations/{cid}` | 對話全文 |
| POST | `/api/conversations/{cid}/turns` | 追問(新一輪) |
| GET | `/api/conversations/{cid}/turns/{tid}/events` | SSE:先補送已有事件,再推新事件;支援 `Last-Event-ID` |
| POST | `/api/conversations/{cid}/turns/{tid}/cancel` | 停止 |

「重試」= 用同一個問題再送一輪。

## 畫面

- **左側欄**:「新對話」按鈕;對話依日期分組,每筆左邊一個狀態點(執行中 / 完成 / 失敗或中斷)。
- **中間對話串**:使用者訊息靠右;agent 回答包含
  - 可收合的「工作過程 · N 步 · 秒數 · 模型」:執行中展開、完成後自動收合;每步可點開看完整輸出。
  - 回答本文(Markdown,經 DOMPurify 清理後才放進頁面)。
  - 修正卡片:差異、測試結果標籤、「複製 git apply 指令」。
- **右側欄**:整份文件模式的回答、完整 diff、工具完整輸出,點開後在右側顯示;窄螢幕時改為覆蓋全畫面。
- **底部輸入框**:資料夾選單、上傳、模式切換、「進階」(模型、思考開關、相關檔案,預設收起)、送出;
  執行中送出鍵變成「停止」。對話送出第一則後,資料夾選單鎖定。
- **外觀**:沿用作品集網頁的配色與字體(冷灰底、鈷藍點綴、Noto Sans TC、IBM Plex Mono),支援深色模式。
  字體走 Google Fonts,沒網路時退回系統字體。

## 錯誤處理

| 狀況 | 處理 |
|---|---|
| NVIDIA 429 / 503 | 先走既有備援鏈;全失敗 → `error` 事件「服務忙線,已試過備援」+ 重試 |
| 整份文件超過上限 | 送出前擋下,顯示預估值與上限 |
| 額度不足 | `error` 事件,顯示今日剩餘額度 |
| 工具出錯 | 回給模型修正(既有行為);畫面標紅 |
| 上傳不合格 | 整包拒收,說明原因 |
| 停止 | 正在等模型回應的那一步會等它回來,之後不再繼續;狀態 `cancelled` |
| 伺服器重開 | 未完成的輪標為 `interrupted` |

## 安全

- 只聽 `127.0.0.1`,同網路的其他電腦連不進來。
- agent 只能存取所選資料夾;機密檔封鎖沿用 `tools._is_secret`。
- 模型輸出一律經 DOMPurify 清理才進頁面,防止回答裡夾帶腳本。
- 網頁不寫入任何專案檔案(patch 存在 `.agent_patches/`,與現在相同)。

## 測試

離線(假模型,不需金鑰):
- `make_tools`:兩組工具綁不同資料夾,同時使用互不干擾。
- `run_agent`:事件順序與欄位;`history` 接續;`should_stop` 停止;`omit_attachments` 移除附檔但保留回答。
- 對話管理:狀態流轉、存檔與載入、重開後標為 `interrupted`、排隊、取消。
- 上傳:zip 路徑穿越、符號連結、超量、機密檔被丟掉、過期清理。
- API(FastAPI TestClient):每支路由;SSE 補送與 `Last-Event-ID` 接續。
- 既有 33 個測試全過。

實機:用真模型在網頁上跑「demo 找 bug → 追問」一次,在內建瀏覽器操作並截圖。

## 相依套件

新增 `fastapi`、`uvicorn`、`python-multipart`;前端 `marked`、`DOMPurify` 的 min 檔放進 `web/static/vendor/`。
