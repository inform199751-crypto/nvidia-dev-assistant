import itertools

import pytest

from shop.orders import DAY, InvalidTransition, Order

ALLOWED = {  # docs/訂單狀態.md 的表
    ("pending", "pay"): "paid", ("pending", "cancel"): "cancelled",
    ("paid", "ship"): "shipped", ("paid", "cancel"): "cancelled",
    ("shipped", "deliver"): "delivered", ("delivered", "return"): "returned",
    ("returned", "refund"): "refunded",
}
PATH = {"pending": [], "paid": ["pay"], "shipped": ["pay", "ship"], "delivered": ["pay", "ship", "deliver"],
        "returned": ["pay", "ship", "deliver", "return"], "refunded": ["pay", "ship", "deliver", "return", "refund"],
        "cancelled": ["cancel"]}
ACTIONS = ["pay", "ship", "deliver", "cancel", "return", "refund"]


def reach(status):
    o, t = Order("X", 800), 1_760_000_000
    for action in PATH[status]:
        t += 3600
        o.apply(action, t)
    return o, t


@pytest.mark.parametrize("status, action", list(itertools.product(PATH, ACTIONS)))
def test_transition_table(status, action):
    o, t = reach(status)
    before = (o.status, o.refunded_amount, list(o.history))
    if (status, action) in ALLOWED:
        o.apply(action, t + 60)
        assert o.status == ALLOWED[(status, action)]
        assert o.history[-1] == (status, action, o.status, t + 60)
    else:
        with pytest.raises(InvalidTransition):
            o.apply(action, t + 60)
        assert (o.status, o.refunded_amount, o.history) == before


def test_cancel_after_payment_refunds_in_full():
    o, t = reach("paid")
    o.apply("cancel", t + 60)
    assert o.refunded_amount == 800


def test_cancel_before_payment_refunds_nothing():
    o, t = reach("pending")
    o.apply("cancel", t + 60)
    assert o.refunded_amount == 0


def test_return_window_includes_exactly_seven_days():
    o, t = reach("delivered")
    o.apply("return", t + 7 * DAY)
    assert o.status == "returned"
    late, t2 = reach("delivered")
    with pytest.raises(InvalidTransition):
        late.apply("return", t2 + 7 * DAY + 1)
    assert late.status == "delivered"
