"""時間處理:系統一律存 Unix 時間(秒),報表、畫面用台灣時間。"""
from datetime import datetime, timedelta, timezone

TAIPEI = timezone(timedelta(hours=8))


def local_day(ts):
    """台灣時間的日期(報表、畫面用這個)。"""
    return datetime.fromtimestamp(ts, TAIPEI).date()


def utc_day(ts):
    """UTC 的日期:只給金流對帳用(金流商用 UTC 切日)。"""
    return datetime.fromtimestamp(ts, timezone.utc).date()


def fmt(ts):
    return datetime.fromtimestamp(ts, TAIPEI).strftime("%Y-%m-%d %H:%M")
