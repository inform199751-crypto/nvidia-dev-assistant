from datetime import date, datetime, timedelta, timezone

from orders.models import Order
from orders.report import daily

TAIPEI = timezone(timedelta(hours=8))


def ts(y, m, d, hh, mm=0):
    return int(datetime(y, m, d, hh, mm, tzinfo=TAIPEI).timestamp())


def test_one_day():
    orders = [Order("A1", ts(2026, 10, 5, 12), 300), Order("A2", ts(2026, 10, 5, 18), 200)]
    assert daily(orders) == {date(2026, 10, 5): {"orders": 2, "revenue": 500, "cancelled": 0}}
