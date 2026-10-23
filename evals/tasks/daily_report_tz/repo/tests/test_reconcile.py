from datetime import date, datetime, timedelta, timezone

from orders.models import Order
from orders.reconcile import settlement_days

TAIPEI = timezone(timedelta(hours=8))


def test_settlement_uses_utc_days():
    early = int(datetime(2026, 10, 6, 3, 0, tzinfo=TAIPEI).timestamp())  # UTC 是 10/5 19:00
    assert settlement_days([Order("A1", early, 100)]) == {date(2026, 10, 5): 1}
