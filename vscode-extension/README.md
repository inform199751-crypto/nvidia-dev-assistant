# 小助理 VS Code 外掛

在 VS Code 旁邊打開對內研發助理(跟 `webui.py` 同一個介面),也能把選取的程式碼直接拿去問。
外掛會在背景啟動小助理,分析的資料夾就是 VS Code 目前開著的那個。

## 安裝

```bash
.venv/Scripts/python vscode-extension/build_vsix.py      # 產生 vscode-extension/dev-assistant-0.1.0.vsix
code --install-extension vscode-extension/dev-assistant-0.1.0.vsix
```

裝好後重新載入 VS Code。第一次使用會請你選小助理的資料夾(裡面有 `webui.py` 和 `.venv`),之後記在設定 `devAssistant.projectPath`。

## 使用

- 命令選擇區(Ctrl+Shift+P)→「小助理: 打開」:在編輯器旁邊打開小助理。
- 選取一段程式碼 → 右鍵「問選取的程式碼」(或 Ctrl+Alt+A):帶著程式碼和檔案位置去問。
- 換了資料夾 →「小助理: 重新啟動」;不用時 →「小助理: 停止」。
- 啟動訊息和錯誤在「輸出」面板的「小助理」。

## 設定

| 設定 | 預設 | 說明 |
|---|---|---|
| `devAssistant.projectPath` | (第一次會問) | 小助理的資料夾 |
| `devAssistant.port` | `8770` | 本機連接埠,跟自己開的 `webui.py`(8765)錯開 |
| `devAssistant.useWorkspaceFolder` | `true` | 分析 VS Code 開著的資料夾;關掉就照小助理 `.env` 的 `AGENT_WORKSPACE` |

修改一樣先放在小助理的草稿上,通過測試後在面板裡按「套用到工作區」才會改到檔案。
