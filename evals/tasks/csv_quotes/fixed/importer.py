"""匯入客戶名單(CSV 文字)。第一行是欄位名稱。"""
import csv


def parse_line(line):
    """把一行 CSV 拆成欄位。欄位可以用雙引號包起來(裡面可以有逗號);引號裡的 "" 代表一個 " 字元。"""
    return next(csv.reader([line]))  # 直接 split(",") 會把引號裡的逗號也切開


def load_customers(text):
    """回傳 dict 的清單,key 是第一行的欄位名稱;空行跳過。"""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header = parse_line(lines[0])
    return [dict(zip(header, parse_line(line))) for line in lines[1:]]
