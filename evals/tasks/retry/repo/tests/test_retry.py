import pytest

from retry import GaveUp, retry


def test_returns_first_success():
    assert retry(lambda: 42, sleep=lambda s: None) == 42


def test_retries_until_success():
    answers = iter([ConnectionError("down"), "ok"])

    def flaky():
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    assert retry(flaky, sleep=lambda s: None) == "ok"


def test_gives_up_with_error():
    def down():
        raise ConnectionError("down")

    with pytest.raises(GaveUp):
        retry(down, attempts=3, sleep=lambda s: None)
