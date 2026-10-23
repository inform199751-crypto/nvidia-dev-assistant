import pytest

from inventory import Inventory, OutOfStock


def stocked(qty=5):
    inv = Inventory()
    inv.add("A", qty)
    return inv


def test_release_unknown_order_is_noop():
    inv = stocked()
    inv.release("A", "nope")
    assert inv.available("A") == 5


def test_release_returns_stock():
    inv = stocked()
    inv.reserve("A", "o1", 4)
    inv.release("A", "o1")
    inv.reserve("A", "o2", 5)
    assert inv.available("A") == 0


def test_same_order_accumulates_within_available():
    inv = stocked()
    inv.reserve("A", "o1", 3)
    with pytest.raises(OutOfStock):
        inv.reserve("A", "o1", 3)
    assert inv.reserved[("A", "o1")] == 3
    inv.reserve("A", "o1", 2)
    assert inv.available("A") == 0


def test_exact_remaining_is_ok():
    inv = stocked()
    inv.reserve("A", "o1", 3)
    inv.reserve("A", "o2", 2)
    assert inv.available("A") == 0
