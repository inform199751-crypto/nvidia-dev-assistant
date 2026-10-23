import pytest

from expr import evaluate


@pytest.mark.parametrize("text, expected", [
    ("8/4/2", 1),
    ("10-2*3-1", 3),
    ("1-1+1", 1),
    ("2*3/6*2", 2),
    ("100/10*2", 20),
    ("2*(3+4)-5", 9),
    ("-3+5", 2),
    ("1 - (2 - 3) - 4", -2),
    ("12 / 2 / 3 - 1 - 1", 0),
])
def test_left_to_right(text, expected):
    assert evaluate(text) == pytest.approx(expected)


def test_errors_still_raised():
    for text in ["(1+2", "1+*2", "3 4"]:
        with pytest.raises(ValueError):
            evaluate(text)
