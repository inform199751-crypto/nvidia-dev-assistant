"""購物車。"""
from billing import catalog


class Cart:
    def __init__(self):
        self.lines = {}  # 商品代碼 → 數量

    def add(self, sku, qty=1):
        if qty <= 0:
            raise ValueError("數量要大於 0")
        catalog.price(sku)  # 沒有這個商品就在這裡擋下
        self.lines[sku] = self.lines.get(sku, 0) + qty
        return self

    def remove(self, sku):
        self.lines.pop(sku, None)
        return self

    def subtotal(self):
        return sum(catalog.price(sku) * qty for sku, qty in self.lines.items())
