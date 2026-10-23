import pytest

from inventory import Inventory, OutOfStock


def test_reserve_and_ship():
    inv = Inventory()
    inv.add("A", 5)
    inv.reserve("A", "o1", 2)
    assert inv.available("A") == 3
    inv.ship("A", "o1")
    assert (inv.stock["A"], inv.available("A")) == (3, 3)


def test_two_orders_cannot_oversell():
    inv = Inventory()
    inv.add("A", 5)
    inv.reserve("A", "o1", 3)
    with pytest.raises(OutOfStock):
        inv.reserve("A", "o2", 3)
    assert inv.available("A") == 2
