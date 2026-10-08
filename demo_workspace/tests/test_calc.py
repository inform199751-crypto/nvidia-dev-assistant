from calc import add, average


def test_add():
    assert add(2, 3) == 5


def test_average():
    assert average([2, 4, 6]) == 4
