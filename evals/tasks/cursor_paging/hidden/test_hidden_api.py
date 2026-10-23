import random

from feed.api import list_orders
from feed.store import OrderStore


def walk(store, shop, size):
    seen, cur = [], None
    for _ in range(1000):
        page = list_orders(store, shop, after=cur, page_size=size)
        assert len(page["orders"]) <= size
        seen += page["orders"]
        cur = page["next_cursor"]
        if not cur:
            return seen
    raise AssertionError("一直有下一頁")


def busy_shop():
    """熱門店家:很多訂單在同一秒進來。"""
    rng = random.Random(7)
    orders = [{"id": i, "shop": "s1", "created": 1_760_000_000 + rng.randint(0, 5)} for i in range(1, 61)]
    orders += [{"id": 1000 + i, "shop": "s2", "created": 1_760_000_000 + i} for i in range(5)]
    rng.shuffle(orders)
    return OrderStore(orders), sorted((o for o in orders if o["shop"] == "s1"),
                                      key=lambda o: (o["created"], o["id"]), reverse=True)


def test_same_second_orders_never_skipped():
    store, expected = busy_shop()
    for size in (1, 7, 10, 25):
        assert walk(store, "s1", size) == [o["id"] for o in expected], size


def test_all_in_one_second():
    store = OrderStore({"id": i, "shop": "s1", "created": 1_760_000_000} for i in range(1, 24))
    assert walk(store, "s1", 5) == list(range(23, 0, -1))


def test_other_shop_not_mixed_in():
    store, _ = busy_shop()
    assert walk(store, "s2", 2) == [1004, 1003, 1002, 1001, 1000]
