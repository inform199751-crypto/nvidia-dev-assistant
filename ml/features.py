"""從一題問題算出特徵:只有數字(字數、附檔數、關鍵字有沒有出現…),不留問題內容,寫進紀錄也不會外洩。

兩種給法:
- vector():數值特徵,給邏輯回歸 / 線性回歸(訓練時再標準化)。
- discrete():分成幾個等級的類別特徵,給 ID3 決策樹(ID3 用資訊熵挑特徵,特徵值要是離散的)。
"""
from __future__ import annotations

import math
import re

# 關鍵字組:中英文都認;一題有出現就是 1
KEYWORDS = {
    "kw_error": r"bug|錯誤|error|exception|traceback|失敗|fail|crash|崩潰|壞掉|修正|修好|修復|\bfix",
    "kw_test": r"test|測試|pytest|單元測試",
    "kw_refactor": r"重構|refactor|架構|architecture|設計|design|模組化|拆分|拆成",
    "kw_perf": r"效能|performance|太慢|變慢|很慢|slow|記憶體|memory|優化|optimi|延遲|latency|吞吐",
    "kw_concurrency": r"race|concurren|並行|平行|同時|deadlock|死結|thread|執行緒|async|非同步|間歇|偶發|偶爾|"
                      r"intermittent|flaky|不穩定|有時|時有時無|高流量",
    # 分類器判斷標準裡的「沒有頭緒」「上次的修正沒用」(difficulty.CLASSIFY_PROMPT)
    "kw_unknown": r"找不到原因|找出原因|根本原因|root cause|看不出|重現不了|無法重現|不知道為什麼|沒用|還是一樣|"
                  r"換個方向|全面檢查",
    "kw_security": r"安全|security|權限|permission|漏洞|vulnerab|認證|\bauth|注入|injection|加密|密碼|資安",
    "kw_multi": r"跨|多個檔案|多個模組|整個專案|整個系統|全部的|所有的|across|multiple|entire|whole|整體",
    "kw_doc": r"文件|readme|翻譯|translate|摘要|summar|寄信|email|信件|整理|列出|報告|會議|簡報",
    "kw_explain": r"解釋|說明|explain|什麼意思|在做什麼|是什麼|怎麼用|教我|介紹",
}
_FILE = re.compile(r"@\S+|[\w./\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cs|cpp|c|h|rb|php|kt|swift|md|ya?ml|json|toml|"
                   r"sql|html|css|sh|ps1|docx|xlsx|pdf|csv|txt)\b", re.I)
_CODE = re.compile(r"```|Traceback|^\s{4,}\S|^\t|\bdef \w+\(|\bclass \w+|=>|;\s*$", re.M)

NAMES = {  # 給人看的名字(樹狀圖、報告)
    "chars": "字數", "lines": "行數", "code": "貼了程式", "files": "提到檔案數", "attachments": "附檔數",
    "history_turns": "前面幾輪", "mode": "模式", "mode_plan": "規劃模式", "mode_execute": "照計畫執行",
    "followup_short": "很短的接話", "log_chars": "字數(log)",
    "kw_error": "提到錯誤", "kw_test": "提到測試", "kw_refactor": "提到重構/設計", "kw_perf": "提到效能",
    "kw_concurrency": "提到並行/間歇", "kw_unknown": "沒有頭緒", "kw_security": "提到安全", "kw_multi": "提到跨多處",
    "kw_doc": "文件/整理", "kw_explain": "要求解釋",
}


def extract(task: str, attachments: list[str] = (), history_turns: int = 0, mode: str = "normal") -> dict:
    """一題的原始特徵(都是數字或短字串)。history_turns:這段對話前面已經問過幾題。"""
    text = re.sub(r"<note>.*?</note>", "", task, flags=re.S).strip()
    low = text.lower()
    f = {
        "chars": len(text),
        "lines": text.count("\n") + 1 if text else 0,
        "code": int(bool(_CODE.search(text))),
        "files": len(set(_FILE.findall(text))),
        "attachments": len(attachments),
        "history_turns": history_turns,
        "mode": mode if mode in ("plan", "execute") else "normal",
    }
    f["followup_short"] = int(f["chars"] <= 12 and history_turns > 0)  # 「繼續」「好」:難度跟上一題一樣
    for name, pattern in KEYWORDS.items():
        f[name] = int(bool(re.search(pattern, low)))
    return f


# ---------- 給邏輯回歸 / 線性回歸:數值向量 ----------

VECTOR = ["log_chars", "lines", "code", "files", "attachments", "history_turns", "mode_plan", "mode_execute",
          "followup_short", *KEYWORDS]


def vector(f: dict, names: list[str] = VECTOR) -> list[float]:
    """字數取 log:從 5 個字到 5000 個字差太多,取 log 比較接近線性關係。"""
    full = {**f, "log_chars": math.log1p(f["chars"]), "lines": min(f["lines"], 50),
            "mode_plan": int(f["mode"] == "plan"), "mode_execute": int(f["mode"] == "execute")}
    return [float(full.get(n, 0)) for n in names]  # 舊紀錄沒有後來加的特徵:當 0


# ---------- 給 ID3 決策樹:離散的等級 ----------

DISCRETE = ["chars", "code", "files", "attachments", "history_turns", "mode", "followup_short", *KEYWORDS]


def discrete(f: dict) -> dict:
    """ID3 的特徵值要是類別:數字切成幾個等級(切點看一般問題的分布:一句話、一段話、貼了一大段)。"""
    level = lambda v, cuts, labels: next((lab for cut, lab in zip(cuts, labels) if v < cut), labels[-1])  # noqa: E731
    return {
        "chars": level(f["chars"], (40, 200), ("短", "中", "長")),
        "code": f["code"],
        "files": level(f["files"], (1, 2), ("0", "1", "2+")),
        "attachments": level(f["attachments"], (1,), ("0", "1+")),
        "history_turns": level(f["history_turns"], (1, 4), ("0", "1-3", "4+")),
        "mode": f["mode"],
        "followup_short": f["followup_short"],
        **{k: f.get(k, 0) for k in KEYWORDS},  # 舊紀錄沒有後來加的關鍵字:當 0
    }
