"""共用的小工具。"""


def parse_bool(value):
    """設定值的字串轉成 True / False:1 / true / yes / on 是 True,0 / false / no / off 是 False
    (不分大小寫、前後空白不算)。其他值丟 ValueError。"""
    return value.strip().lower() not in ("0", "")


def parse_int(value, default=0):
    value = value.strip()
    return int(value) if value else default
