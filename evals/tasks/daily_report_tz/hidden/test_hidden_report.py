from datetime import date, datetime, timedelta, timezone

from orders.models import Order
from orders.reconcile import settlement_days
from orders.report import daily

TAIPEI = timezone(timedelta(hours=8))


def ts(y, m, d, hh, mm=0):
    return int(datetime(y, m, d, hh, mm, tzinfo=TAIPEI).timestamp())


def test_early_morning_orders_count_on_local_day():
    orders = [Order("A1", ts(2026, 10, 20, 1, 30), 120), Order("A2", ts(2026, 10, 20, 7, 59), 80),
              Order("A3", ts(2026, 10, 19, 23, 50), 50)]
    r = daily(orders)
    assert r[date(2026, 10, 20)]["orders"] == 2 and r[date(2026, 10, 20)]["revenue"] == 200
    assert r[date(2026, 10, 19)]["orders"] == 1 and r[date(2026, 10, 19)]["revenue"] == 50


def test_cancelled_and_refunded_not_counted_as_sales():
    orders = [Order("B1", ts(2026, 10, 21, 10), 300), Order("B2", ts(2026, 10, 21, 11), 500, "cancelled"),
              Order("B3", ts(2026, 10, 21, 12), 700, "refunded"), Order("B4", ts(2026, 10, 21, 2), 100, "cancelled")]
    assert daily(orders)[date(2026, 10, 21)] == {"orders": 1, "revenue": 300, "cancelled": 2}


def test_reconcile_still_uses_utc():
    orders = [Order("C1", ts(2026, 10, 22, 3), 100), Order("C2", ts(2026, 10, 22, 9), 100)]
    assert settlement_days(orders) == {date(2026, 10, 21): 1, date(2026, 10, 22): 1}
