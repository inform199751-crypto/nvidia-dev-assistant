# 多人部署

把小助理架在一台伺服器上給團隊用:每人各自的對話、草稿和每日額度,前面接公司的 SSO 登入。

```
使用者瀏覽器 ──► auth(oauth2-proxy,公司 SSO 登入,:4180)
                    │  X-Forwarded-Email: amy@corp
                    ▼
                 assistant(webui.py --team,不對外開 port)
                    │  每人一個 Hub:對話 / 草稿 / patch / 上網確認 各自分開
                    │  全部人共用「同時執行上限」(預設 4,其他人排隊)
                    ├──► redis(每人每日 token 額度、同時請求數)
                    ├──► /workspace(要分析的程式碼,唯讀,大家共用)
                    └──► 模型:build.nvidia.com(PoC)或地端 NIM 容器(正式)
```

## 跟單人版的差別

| | 單人(`webui.py`) | 多人(`webui.py --team`) |
|---|---|---|
| 身分 | 這台電腦 | 登入代理送來的 `X-Forwarded-Email` |
| 對話、草稿 | 一份 | 每人一份,看不到別人的 |
| 套用修改 | 按「套用到工作區」 | **不能直接改**(工作區是共用、唯讀的),改成「下載 patch」自己套 |
| patch 存在 | 工作區的 `.agent_patches/` | `/data/output/patches/<信箱>/` |
| 記住規則(ASSISTANT.md) | 可以 | 由管理者直接改工作區的 ASSISTANT.md |
| 額度 | 記憶體 | 有 `REDIS_URL` 就記在 Redis(多台伺服器共用) |

## 步驟

1. 準備 `.env`(專案根目錄,跟單人版一樣):`NVIDIA_API_KEY`;要上網就設 `ALLOW_WEB` / `TAVILY_API_KEY`。
2. 準備登入代理:`cp deploy/auth.env.example deploy/auth.env`,填公司 SSO(OIDC)的 client id / secret、cookie secret、允許的信箱網域。
3. 指定要分析的程式碼:`REPO_PATH=/srv/repos/our-project`(沒設就用 demo_workspace)。
4. 啟動:
   ```bash
   REPO_PATH=/srv/repos/our-project docker compose -f deploy/docker-compose.yml up -d --build
   ```
5. 瀏覽器開 `http://<伺服器>:4180`,用公司帳號登入。正式環境請在 4180 前面再加 HTTPS(公司的反向代理或負載平衡器)。

## 安全重點

- **小助理只能經過登入代理連到**:compose 裡 assistant 只有 `expose` 沒有 `ports`。身分是看標頭的,如果小助理直接對外,任何人都能自己填 `X-Forwarded-Email` 冒充別人。
- 工作區唯讀掛載;小助理的修改只在每個人自己的草稿裡,下載 patch 後由工程師自己審核、套用。
- 機密檔(`.env`、金鑰、憑證)小助理一樣讀不到;建映像時 `.dockerignore` 也把 `.env` 排除了。
- **用 build.nvidia.com 時程式碼會送到 NVIDIA**(API Trial 條款 §1.2 禁止正式使用、§3.3 允許 NVIDIA 收集輸入),公司程式碼請等換成地端 NIM。

## 沙箱(run_tests / run_command)

預設**沒開**:容器裡沒有 Docker,這兩個工具會回「連不到 Docker」,其他功能照常。要開的話(在 Linux 主機):

1. 掛 docker.sock,並讓沙箱複本放在「主機和容器裡路徑一樣」的資料夾(主機的 Docker 要找得到它):
   ```yaml
   volumes:
     - /var/run/docker.sock:/var/run/docker.sock
     - /srv/dev-assistant/sandbox:/srv/dev-assistant/sandbox
   environment:
     TMPDIR: /srv/dev-assistant/sandbox
   group_add: ["<主機 docker 群組的 gid>"]
   ```
2. 在主機上建沙箱映像:`docker build -t dev-assistant-sandbox sandbox`(正式環境換成裝好你們 repo 相依套件的映像)。

注意:掛 docker.sock 等同給容器主機的 root 權限。正式環境建議改用 gVisor / Kata 之類的隔離執行環境,或獨立的沙箱服務。

## 換成地端 NIM

1. 在 GPU 主機上跑 NIM 容器(compose 最下面有註解掉的範例;要 NGC 金鑰,模型名稱與 GPU 數量依 NVIDIA 文件調整)。
2. `.env` 設 `LLM_BACKEND=nim`;`models.yaml` 的 `nim_base_url` 指到各 NIM 容器,`nim_context_window` 要跟 `NIM_MAX_MODEL_LEN` 一致。
3. 先跑 `smoke_test.py` 確認每個模型都回應正常(地端的話把 backend 改成 nim)。

## 限制

- 對話存在記憶體:小助理重啟後對話會清空(額度在 Redis 不受影響)。要跑多台,負載平衡要設「同一個人固定到同一台」(sticky session)。
- 同時執行上限是單台的;多台時每台各自算。
- 長任務(例如整份技術文件)會占住一個執行名額好幾分鐘,上限要依 GPU 數量調整(`--max-concurrent`)。
