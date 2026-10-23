"""客服後台用的訂單服務。"""
from shop.orders import InvalidTransition, Order


class OrderService:
    def __init__(self, clock):
        self.clock = clock
        self.orders = {}

    def create(self, order_id, amount):
        self.orders[order_id] = Order(order_id, amount)
        return self.orders[order_id]

    def act(self, order_id, action):
        """回傳 (成功?, 訊息);客服畫面直接顯示訊息。"""
        order = self.orders[order_id]
        try:
            order.apply(action, self.clock())
        except InvalidTransition as e:
            return False, str(e)
        return True, f"訂單 {order_id} 現在是 {order.status}"
