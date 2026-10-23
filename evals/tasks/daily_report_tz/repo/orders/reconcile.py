"""跟金流商對帳:金流商的結算檔用 UTC 切日,所以這裡一定要用 UTC 的日期。"""
from collections import Counter

from orders import timeutil


def settlement_days(orders):
    """每個 UTC 日期有幾筆 paid 訂單,拿去跟金流商的結算檔比對。"""
    return dict(Counter(timeutil.utc_day(o.created) for o in orders if o.status == "paid"))
