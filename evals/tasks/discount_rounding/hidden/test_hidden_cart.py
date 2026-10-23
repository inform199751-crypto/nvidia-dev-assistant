from cart import cart_total, line_total


def test_half_rounds_up():
    assert line_total(1, 1, 0.5) == 1
    assert line_total(7, 1, 0.5) == 4


def test_float_prices():
    assert line_total(19.9, 3) == 60
    assert line_total(10.05, 1) == 10
    assert line_total(2.675, 2) == 5


def test_each_line_rounded_before_sum():
    assert cart_total([(5, 1, 0.5), (1, 1, 0.5)]) == 4
