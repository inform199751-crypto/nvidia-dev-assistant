"""訂單狀態轉換(規則見 docs/訂單狀態.md)。"""

DAY = 24 * 60 * 60


class InvalidTransition(Exception):
    pass


TRANSITIONS = {
    ("pending", "pay"): "paid",
    ("pending", "cancel"): "cancelled",
    ("paid", "ship"): "shipped",
    ("paid", "cancel"): "cancelled",
    ("shipped", "cancel"): "cancelled",
    ("shipped", "deliver"): "delivered",
    ("delivered", "return"): "returned",
    ("delivered", "refund"): "refunded",
    ("returned", "refund"): "refunded",
}


class Order:
    def __init__(self, order_id, amount):
        self.id = order_id
        self.amount = amount
        self.status = "pending"
        self.refunded_amount = 0
        self.delivered_at = None
        self.history = []

    def apply(self, action, now):
        """做一個動作(now 是 Unix 時間,秒);不允許就丟 InvalidTransition。"""
        new = TRANSITIONS.get((self.status, action))
        if new is None:
            raise InvalidTransition(f"{self.status} 的訂單不能 {action}")
        if action == "return" and now - self.delivered_at >= 7 * DAY:
            raise InvalidTransition("超過退貨期限")
        if action == "deliver":
            self.delivered_at = now
        if action == "refund":
            self.refunded_amount = self.amount
        self.history.append((self.status, action, new, now))
        self.status = new
        return self
