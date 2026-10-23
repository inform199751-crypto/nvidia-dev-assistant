from feed.api import list_orders
from feed.store import OrderStore


def store_with(n, shop="s1"):
    return OrderStore({"id": i, "shop": shop, "created": 1_760_000_000 + i * 60} for i in range(1, n + 1))


def test_first_page_newest_first():
    page = list_orders(store_with(30), "s1", page_size=10)
    assert page["orders"] == list(range(30, 20, -1))
    assert page["next_cursor"]


def test_walk_all_pages():
    store, seen, cur = store_with(25), [], None
    while True:
        page = list_orders(store, "s1", after=cur, page_size=10)
        seen += page["orders"]
        cur = page["next_cursor"]
        if not cur:
            break
    assert seen == list(range(25, 0, -1))
