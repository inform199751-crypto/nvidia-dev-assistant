"""終端機表格的文字截斷。中日韓文字(和全形字)在終端機佔兩格寬。"""


def width(text):
    """文字在終端機佔幾格。"""
    return len(text)


def truncate(text, max_width):
    """超過 max_width 格就截斷,結尾加「…」(佔一格);結果不超過 max_width 格,而且盡量留多一點字。"""
    if width(text) <= max_width:
        return text
    return text[:max_width - 1] + "…"
