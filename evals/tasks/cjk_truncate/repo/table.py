"""把資料印成對齊的表格。"""
from display import truncate, width


def render(rows, widths):
    """rows:每列是字串的清單;widths:每欄的寬度(格數)。太長的欄位截斷,不夠的補空白。"""
    out = []
    for row in rows:
        cells = []
        for text, w in zip(row, widths):
            text = truncate(text, w)
            cells.append(text + " " * (w - width(text)))
        out.append(" | ".join(cells))
    return "\n".join(out)
