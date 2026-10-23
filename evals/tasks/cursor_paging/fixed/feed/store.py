"""訂單資料(記憶體版;正式環境是資料庫,查詢的語意一樣)。"""


def _position(o):
    return (o["created"], o["id"])


class OrderStore:
    def __init__(self, orders=()):
        self.orders = [dict(o) for o in orders]  # {"id": int, "shop": str, "created": Unix 時間(秒)}

    def add(self, order):
        self.orders.append(dict(order))

    def latest(self, shop, limit):
        """最新的 limit 筆,新的在前。"""
        return self.older_than(shop, None, limit)

    def older_than(self, shop, before, limit):
        """位置 before = (created, id) 之前(不含)的訂單,新的在前(同一秒的,編號大的在前),最多 limit 筆;
        before=None 從最新的開始。只用 created 的話,同一秒的訂單剛好跨兩頁時,後面那幾筆會被當成「看過了」漏掉。"""
        rows = [o for o in self.orders if o["shop"] == shop and (before is None or _position(o) < tuple(before))]
        rows.sort(key=_position, reverse=True)
        return rows[:limit]
