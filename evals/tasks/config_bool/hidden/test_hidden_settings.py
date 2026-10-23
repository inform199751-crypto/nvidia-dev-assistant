import pytest

from app.settings import load
from app.utils import parse_bool


def test_defaults_are_off():
    s = load({})
    assert (s.debug, s.workers, s.log_sql) == (False, 4, False)


@pytest.mark.parametrize("text", ["1", "true", "TRUE", " yes ", "On"])
def test_true_values(text):
    assert parse_bool(text) is True


@pytest.mark.parametrize("text", ["0", "false", "False", "no", " OFF "])
def test_false_values(text):
    assert parse_bool(text) is False


def test_unknown_value_rejected():
    with pytest.raises(ValueError):
        parse_bool("maybe")
