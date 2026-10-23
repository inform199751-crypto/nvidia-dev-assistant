from cart import cart_total, line_total


def test_line_total_without_discount():
    assert line_total(120, 3) == 360


def test_line_total_rounds_half_up():
    assert line_total(5, 1, 0.5) == 3
    assert line_total(25, 1, 0.9) == 3


def test_cart_total():
    assert cart_total([(100, 2, 0.0), (80, 1, 0.25)]) == 260
