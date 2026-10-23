import unicodedata

from display import truncate, width
from table import render


def cells(text):
    """驗收自己算的寬度(不用 display.width,那是受測的程式)。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def test_width_counts_wide_chars():
    assert width("中文") == 4
    assert width("a中b") == 4
    assert width("ＡＢ") == 4
    assert width("abc") == 3


def test_truncate_chinese_fits_and_keeps_most():
    for text, w in [("資料庫連線逾時", 8), ("ab中文字", 6), ("中文標題很長", 5), ("中", 1)]:
        out = truncate(text, w)
        assert out.endswith("…")
        assert w - 1 <= cells(out) <= w, (text, w, out)


def test_exact_results():
    assert truncate("資料庫連線逾時", 8) == "資料庫…"
    assert truncate("ab中文字", 6) == "ab中…"
    assert truncate("中文", 4) == "中文"


def test_table_aligned_with_chinese():
    lines = render([["名稱", "狀態"], ["資料庫連線逾時", "失敗"]], [8, 4]).split("\n")
    assert [cells(line) for line in lines] == [15, 15]
