"""每日報表:每天的訂單數和營業額。"""
from collections import defaultdict

from orders import timeutil


def daily(orders):
    """回傳 {日期: {"orders": 訂單數, "revenue": 營業額}},日期照台灣時間切。只算 paid 的訂單。"""
    days = defaultdict(lambda: {"orders": 0, "revenue": 0})
    for o in orders:
        if o.status != "paid":
            continue
        d = days[timeutil.utc_day(o.created)]
        d["orders"] += 1
        d["revenue"] += o.amount
    return dict(days)
