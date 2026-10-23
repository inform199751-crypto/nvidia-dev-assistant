"""新訂單通知:App 開著的時候,每隔幾秒問一次有沒有新訂單。"""


def new_since(store, shop, last_seen_created):
    """last_seen_created 之後(不含)進來的訂單編號,舊的在前。"""
    rows = [o for o in store.orders if o["shop"] == shop and o["created"] > last_seen_created]
    return [o["id"] for o in sorted(rows, key=lambda o: (o["created"], o["id"]))]
