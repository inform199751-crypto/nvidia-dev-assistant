# ASSISTANT.md

給小助理的專案規則與慣例,每段對話開始時自動讀取(像 Claude Code 的 CLAUDE.md)。
在互動模式輸入 `/remember 規則` 也會加到這裡。

- 這是示範用的小專案:`calc.py` 提供 `add` / `average`,測試在 `tests/`,用 pytest。
- 引用程式位置一律寫成 `path:line`。
- 修改或新增函式時,同一個 patch 裡一併補上對應的測試。
