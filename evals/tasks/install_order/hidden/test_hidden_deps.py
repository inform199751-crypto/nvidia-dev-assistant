import pytest

from deps import CycleError, install_order


def valid(deps, order):
    """驗收自己檢查:每個套件只出現一次、都排在它依賴的後面、該出現的都有。"""
    names = set(deps) | {d for ds in deps.values() for d in ds}
    assert sorted(order) == sorted(names)
    where = {name: i for i, name in enumerate(order)}
    return all(where[d] < where[name] for name, ds in deps.items() for d in ds)


def test_shared_dependency_is_not_a_cycle():
    deps = {"app": ["api", "ui"], "api": ["core"], "ui": ["core"], "core": []}
    assert valid(deps, install_order(deps))


def test_bigger_graph_without_cycle():
    deps = {"a": ["b", "c", "d"], "b": ["d", "e"], "c": ["e"], "d": ["e", "f"], "e": ["f"], "x": ["a", "f"]}
    assert valid(deps, install_order(deps))


def test_self_dependency():
    with pytest.raises(CycleError):
        install_order({"a": ["a"]})


def test_cycle_behind_valid_part():
    with pytest.raises(CycleError):
        install_order({"app": ["lib"], "lib": ["util"], "util": ["helpers"], "helpers": ["lib"]})


def test_two_separate_graphs():
    deps = {"a": ["b"], "c": ["d"], "d": []}
    assert valid(deps, install_order(deps))
