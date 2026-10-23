"""App 的訂單列表 API。"""
from feed import cursor


def list_orders(store, shop, after=None, page_size=20):
    """一頁 page_size 筆訂單編號,新的在前(同一秒進來的,編號大的在前)。
    after 是上一頁回傳的 next_cursor;沒有下一頁時 next_cursor 是 None。"""
    pos = cursor.decode(after)
    rows = store.older_than(shop, pos["created"] if pos else None, page_size)
    nxt = cursor.encode({"created": rows[-1]["created"]}) if len(rows) == page_size else None
    return {"orders": [o["id"] for o in rows], "next_cursor": nxt}
