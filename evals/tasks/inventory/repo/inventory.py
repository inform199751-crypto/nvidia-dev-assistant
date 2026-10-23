"""庫存與預留。可用數量 = 庫存 - 已預留。"""


class OutOfStock(Exception):
    pass


class Inventory:
    def __init__(self):
        self.stock = {}     # sku -> 庫存數量
        self.reserved = {}  # (sku, order_id) -> 預留數量

    def add(self, sku, qty):
        self.stock[sku] = self.stock.get(sku, 0) + qty

    def available(self, sku):
        held = sum(q for (s, _), q in self.reserved.items() if s == sku)
        return self.stock.get(sku, 0) - held

    def reserve(self, sku, order_id, qty):
        """幫訂單預留;可用數量不夠就丟 OutOfStock(什麼都不改)。同一張訂單再預留會累加。"""
        if self.stock.get(sku, 0) < qty:
            raise OutOfStock(sku)
        key = (sku, order_id)
        self.reserved[key] = self.reserved.get(key, 0) + qty

    def release(self, sku, order_id):
        """取消訂單:把這張訂單的預留放回去。沒有預留就什麼都不做。"""
        self.reserved.pop((sku, order_id))

    def ship(self, sku, order_id):
        """出貨:預留的數量從庫存扣掉。"""
        qty = self.reserved.pop((sku, order_id))
        self.stock[sku] -= qty
