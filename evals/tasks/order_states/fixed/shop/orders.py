"""訂單狀態轉換(規則見 docs/訂單狀態.md)。"""

DAY = 24 * 60 * 60


class InvalidTransition(Exception):
    pass


TRANSITIONS = {  # 跟 docs/訂單狀態.md 的表一致:已出貨不能取消、已送達不能直接退款(要先退貨)
    ("pending", "pay"): "paid",
    ("pending", "cancel"): "cancelled",
    ("paid", "ship"): "shipped",
    ("paid", "cancel"): "cancelled",
    ("shipped", "deliver"): "delivered",
    ("delivered", "return"): "returned",
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
        if action == "return" and now - self.delivered_at > 7 * DAY:  # 剛好 7 × 24 小時也可以
            raise InvalidTransition("超過退貨期限")
        if action == "deliver":
            self.delivered_at = now
        if action == "refund" or (action == "cancel" and self.status == "paid"):  # 付款後取消要全額退款
            self.refunded_amount = self.amount
        self.history.append((self.status, action, new, now))
        self.status = new
        return self
