"""商品資料庫(記憶體版,會記錄讀寫次數;正式環境是 PostgreSQL)。"""


class Database:
    def __init__(self, products=()):
        self.rows = {p["sku"]: dict(p) for p in products}  # sku → {"sku", "name", "price"}
        self.reads = 0
        self.writes = 0

    def get(self, sku):
        self.reads += 1
        row = self.rows.get(sku)
        return dict(row) if row else None

    def save(self, product):
        self.writes += 1
        self.rows[product["sku"]] = dict(product)

    def save_many(self, products):
        """一次寫很多筆(匯入用,一個交易)。"""
        self.writes += 1
        for p in products:
            self.rows[p["sku"]] = dict(p)
