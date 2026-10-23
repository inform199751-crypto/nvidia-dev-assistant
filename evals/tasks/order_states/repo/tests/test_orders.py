import pytest

from shop.orders import DAY, InvalidTransition, Order


def test_happy_path_to_refund():
    o = Order("A1", 500)
    for i, action in enumerate(["pay", "ship", "deliver"]):
        o.apply(action, 1000 + i)
    o.apply("return", 1002 + 3 * DAY)
    o.apply("refund", 1002 + 4 * DAY)
    assert (o.status, o.refunded_amount, len(o.history)) == ("refunded", 500, 5)


def test_cannot_ship_unpaid():
    with pytest.raises(InvalidTransition):
        Order("A2", 100).apply("ship", 0)
