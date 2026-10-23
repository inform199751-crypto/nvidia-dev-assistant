"""訂單匯出成 CSV(給會計)。"""
import csv
import io
from datetime import datetime, timedelta, timezone

TAIPEI = timezone(timedelta(hours=8))


def to_csv(orders):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["編號", "時間", "金額", "狀態"])
    for o in orders:
        w.writerow([o.id, datetime.fromtimestamp(o.created, TAIPEI).strftime("%Y-%m-%d %H:%M"), o.amount, o.status])
    return out.getvalue()
