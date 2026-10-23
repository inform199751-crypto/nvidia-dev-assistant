"""訂單服務的用戶端。"""
from retry import retry


def fetch_order(transport, order_id):
    """transport(path) 回傳 dict;連不上時丟 ConnectionError。"""
    data = retry(lambda: transport(f"/orders/{order_id}"))
    return {"id": data["id"], "total": data["total"]}
