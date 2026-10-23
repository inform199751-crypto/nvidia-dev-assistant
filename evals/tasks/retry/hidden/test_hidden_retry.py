import pytest

from retry import GaveUp, retry


def always_down(calls):
    def fn():
        calls.append(1)
        raise ConnectionError("down")
    return fn


def test_calls_and_cause():
    calls = []
    with pytest.raises(GaveUp) as info:
        retry(always_down(calls), attempts=3, sleep=lambda s: None)
    assert len(calls) == 3
    assert isinstance(info.value.__cause__, ConnectionError)


def test_no_wait_after_last_failure():
    slept = []
    with pytest.raises(GaveUp):
        retry(always_down([]), attempts=3, sleep=slept.append)
    assert slept == [1.0, 2.0]


def test_backoff_uses_base_delay():
    slept = []
    with pytest.raises(GaveUp):
        retry(always_down([]), attempts=4, base_delay=0.5, sleep=slept.append)
    assert slept == [0.5, 1.0, 2.0]


def test_other_errors_not_retried():
    calls, slept = [], []

    def bad():
        calls.append(1)
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        retry(bad, sleep=slept.append)
    assert (len(calls), slept) == (1, [])
