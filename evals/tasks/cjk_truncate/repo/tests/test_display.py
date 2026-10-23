from display import truncate
from table import render


def test_short_text_unchanged():
    assert truncate("hello", 10) == "hello"


def test_long_ascii_truncated():
    assert truncate("hello world", 8) == "hello w…"


def test_table_aligned():
    assert render([["id", "status"], ["1", "ok"]], [4, 6]) == "id   | status\n1    | ok    "
