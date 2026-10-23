import pytest

from deps import CycleError, install_order


def test_chain():
    assert install_order({"app": ["web"], "web": ["core"]}) == ["core", "web", "app"]


def test_cycle_is_reported():
    with pytest.raises(CycleError):
        install_order({"a": ["b"], "b": ["c"], "c": ["a"]})
