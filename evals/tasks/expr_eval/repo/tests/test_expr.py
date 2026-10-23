import pytest

from expr import evaluate
from report import compute


def test_precedence():
    assert evaluate("2+3*4") == 14


def test_parentheses():
    assert evaluate("(2+3)*4") == 20


def test_subtraction_left_to_right():
    assert evaluate("2-3-4") == -5


def test_report_formula():
    assert compute("營收 - 成本 - 退款", {"營收": 100, "成本": 60, "退款": 10}) == 30


def test_bad_input():
    with pytest.raises(ValueError):
        evaluate("2+")
