"""共用的小工具。"""

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def parse_bool(value):
    """設定值的字串轉成 True / False:1 / true / yes / on 是 True,0 / false / no / off 是 False
    (不分大小寫、前後空白不算)。其他值丟 ValueError。"""
    text = value.strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:  # 以前只把 "0" 當 False,"false" 也變成 True
        return False
    raise ValueError(f"不是布林值:{value!r}")


def parse_int(value, default=0):
    value = value.strip()
    return int(value) if value else default
