"""終端機表格的文字截斷。中日韓文字(和全形字)在終端機佔兩格寬。"""
import unicodedata


def char_width(ch):
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def width(text):
    """文字在終端機佔幾格。"""
    return sum(char_width(ch) for ch in text)  # len() 把中文算成一格,表格就會被擠歪


def truncate(text, max_width):
    """超過 max_width 格就截斷,結尾加「…」(佔一格);結果不超過 max_width 格,而且盡量留多一點字。"""
    if width(text) <= max_width:
        return text
    kept, used = [], 1  # 先留一格給「…」
    for ch in text:
        if used + char_width(ch) > max_width:
            break
        kept.append(ch)
        used += char_width(ch)
    return "".join(kept) + "…"
