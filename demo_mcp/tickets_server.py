"""示範用的 MCP server:虛構的客服工單系統(資料是編的)。HSUAN「連接器」的實測和離線測試都用它。

只用標準函式庫,stdio 傳輸(一行一個 JSON-RPC 訊息)。在 HSUAN 的「連接器」頁按「加入示範連接器」就會接上。
"""
from __future__ import annotations

import json
import sys

TICKETS = [
    {"id": "T-1001", "title": "登入頁在 Safari 顯示空白", "status": "open", "priority": "高", "team": "前端組",
     "created": "2026-10-12",
     "detail": "客戶回報 iPhone Safari 17 打開登入頁一片空白,Chrome 正常。主控台錯誤:"
               "SyntaxError: Unexpected token '?.'(login.bundle.js:1)。疑似打包設定沒有轉譯 optional chaining。",
     "log": ["10/12 前端組收件", "10/13 重現:Safari 17.0 必現,17.4 正常"]},
    {"id": "T-1002", "title": "匯出 Excel 中文變亂碼", "status": "open", "priority": "中", "team": "後端組",
     "created": "2026-10-13",
     "detail": "報表匯出的 CSV 用 Excel 打開,中文全部變亂碼;用記事本打開正常。",
     "log": ["10/13 後端組收件"]},
    {"id": "T-1003", "title": "訂單通知信寄了兩次", "status": "closed", "priority": "中", "team": "後端組",
     "created": "2026-10-08",
     "detail": "部分客戶同一張訂單收到兩封通知信。原因:付款回呼重送時沒有檢查是否已寄過。",
     "log": ["10/8 收件", "10/9 加上冪等檢查(用訂單編號當 key)", "10/10 上線,結案"]},
    {"id": "T-1004", "title": "後台列表載入要 12 秒", "status": "open", "priority": "高", "team": "後端組",
     "created": "2026-10-14",
     "detail": "訂單後台的列表頁資料多了之後載入要 12 秒。慢查詢紀錄:orders 表依 created_at 排序、沒有索引,全表掃描 80 萬筆。",
     "log": ["10/14 收件"]},
]

TOOLS = [
    {"name": "list_tickets",
     "description": "列出客服工單(編號、標題、狀態、優先度、負責組別)。",
     "inputSchema": {"type": "object", "properties": {
         "status": {"type": "string", "enum": ["open", "closed", "all"], "description": "open 未結案、closed 已結案、all 全部(預設 open)"}}}},
    {"name": "get_ticket",
     "description": "讀一張工單的完整內容:問題描述和處理紀錄。",
     "inputSchema": {"type": "object", "properties": {
         "ticket_id": {"type": "string", "description": "工單編號,例如 T-1001"}}, "required": ["ticket_id"]}},
]


def text(body: str, error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": body}], "isError": error}


def call(name: str, args: dict) -> dict:
    if name == "list_tickets":
        status = args.get("status") or "open"
        rows = [t for t in TICKETS if status == "all" or t["status"] == status]
        return text("\n".join(f"{t['id']} | {t['title']} | {t['status']} | 優先度 {t['priority']} | {t['team']}"
                              for t in rows) or "沒有符合的工單")
    if name == "get_ticket":
        t = next((t for t in TICKETS if t["id"].lower() == str(args.get("ticket_id", "")).strip().lower()), None)
        if not t:
            return text(f"找不到工單 {args.get('ticket_id')}", error=True)
        return text(f"{t['id']} {t['title']}\n狀態:{t['status']} · 優先度:{t['priority']} · 負責:{t['team']} · 建立:{t['created']}\n"
                    f"描述:{t['detail']}\n處理紀錄:\n" + "\n".join(f"- {x}" for x in t["log"]))
    return text(f"沒有 {name} 這個工具", error=True)


def handle(msg: dict) -> dict | None:
    method, rid = msg.get("method"), msg.get("id")
    if rid is None:  # 通知(notifications/initialized 等):不用回
        return None
    if method == "initialize":
        result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2025-06-18"),
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "demo-tickets", "version": "1.0"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        p = msg.get("params") or {}
        result = call(p.get("name", ""), p.get("arguments") or {})
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"不支援 {method}"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def main() -> None:
    out = sys.stdout.buffer  # 不用 print:Windows 的主控台編碼(cp950)會把中文弄壞
    for raw in sys.stdin.buffer:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except ValueError:
            continue
        if isinstance(msg, dict) and (reply := handle(msg)):
            out.write((json.dumps(reply, ensure_ascii=False) + "\n").encode("utf-8"))
            out.flush()


if __name__ == "__main__":
    main()
