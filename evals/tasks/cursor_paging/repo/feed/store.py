"""訂單資料(記憶體版;正式環境是資料庫,查詢的語意一樣)。"""


class OrderStore:
    def __init__(self, orders=()):
        self.orders = [dict(o) for o in orders]  # {"id": int, "shop": str, "created": Unix 時間(秒)}

    def add(self, order):
        self.orders.append(dict(order))

    def latest(self, shop, limit):
        """最新的 limit 筆,新的在前。"""
        return self.older_than(shop, None, limit)

    def older_than(self, shop, before, limit):
        """created 在 before 之前(不含)的訂單,新的在前,最多 limit 筆;before=None 從最新的開始。"""
        rows = [o for o in self.orders if o["shop"] == shop and (before is None or o["created"] < before)]
        rows.sort(key=lambda o: o["created"], reverse=True)
        return rows[:limit]
