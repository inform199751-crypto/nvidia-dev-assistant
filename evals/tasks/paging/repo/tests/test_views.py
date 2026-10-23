from views import order_page

ORDERS = [f"A{n:03d}" for n in range(23)]


def test_first_page():
    assert order_page(ORDERS, 1)["orders"] == ORDERS[:10]


def test_page_count():
    assert order_page(ORDERS, 1)["pages"] == 3
