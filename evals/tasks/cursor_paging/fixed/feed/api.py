"""App 的訂單列表 API。"""
from feed import cursor


def list_orders(store, shop, after=None, page_size=20):
    """一頁 page_size 筆訂單編號,新的在前(同一秒進來的,編號大的在前)。
    after 是上一頁回傳的 next_cursor;沒有下一頁時 next_cursor 是 None。"""
    pos = cursor.decode(after)
    rows = store.older_than(shop, (pos["created"], pos["id"]) if pos else None, page_size)
    last = rows[-1] if len(rows) == page_size else None
    nxt = cursor.encode({"created": last["created"], "id": last["id"]}) if last else None  # 位置要帶編號,同一秒的才分得開
    return {"orders": [o["id"] for o in rows], "next_cursor": nxt}
