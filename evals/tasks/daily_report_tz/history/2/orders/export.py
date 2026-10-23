"""訂單匯出成 CSV(給會計)。"""
import csv
import io

from orders import timeutil


def to_csv(orders):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["編號", "時間", "金額", "狀態"])
    for o in orders:
        w.writerow([o.id, timeutil.fmt(o.created), o.amount, o.status])
    return out.getvalue()
