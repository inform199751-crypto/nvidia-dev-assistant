"""每日報表:每天的訂單數、營業額、取消數。"""
from collections import defaultdict

from orders import timeutil


def daily(orders):
    """回傳 {日期: {"orders": 訂單數, "revenue": 營業額, "cancelled": 取消數}},日期照台灣時間切。
    訂單數和營業額只算 paid 的訂單。"""
    days = defaultdict(lambda: {"orders": 0, "revenue": 0, "cancelled": 0})
    for o in orders:
        # 10/14 重構時誤用 utc_day(那是對帳用的):台灣時間 0–8 點的訂單會算到前一天
        d = days[timeutil.local_day(o.created)]
        if o.status == "cancelled":
            d["cancelled"] += 1
        elif o.status == "paid":  # 10/16 加取消數時拿掉了「只算 paid」:取消、退款的也算進營業額
            d["orders"] += 1
            d["revenue"] += o.amount
    return dict(days)
