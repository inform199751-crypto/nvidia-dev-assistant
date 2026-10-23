from billing.cart import Cart
from billing.checkout import checkout
from billing.coupons import Coupon, discount


def test_paying_with_points_keeps_free_shipping():
    r = checkout(Cart().add("mug").add("beans"), "臺北市大安區", use_points=200)  # 小計 1,130
    assert (r["points"], r["shipping"], r["total"]) == (200, 0, 930)


def test_exactly_threshold_ships_free():
    r = checkout(Cart().add("mug").add("beans"), "臺北市大安區", Coupon("C130", "fixed", 130))
    assert (r["coupon"], r["shipping"], r["total"]) == (130, 0, 1000)


def test_island_threshold_with_points():
    r = checkout(Cart().add("grinder").add("mug").add("filter"), "金門縣金城鎮", use_points=300)  # 2,040
    assert (r["shipping"], r["total"]) == (0, 1740)


def test_island_exactly_threshold():
    r = checkout(Cart().add("kettle", 2).add("filter"), "澎湖縣馬公市", Coupon("C50", "fixed", 50))  # 2,050 - 50
    assert (r["shipping"], r["total"]) == (0, 2000)


def test_below_threshold_still_pays_shipping():
    r = checkout(Cart().add("kettle"), "臺北市大安區", use_points=100)
    assert (r["points"], r["shipping"], r["total"]) == (100, 80, 960)


def test_coupon_math_unchanged():
    assert discount(1250, Coupon("P", "percent", 0.9)) == 125
    assert discount(999, Coupon("P", "percent", 0.9, min_spend=1000)) == 0
    assert discount(80, Coupon("F", "fixed", 100)) == 80


def test_tax_and_points_earned():
    r = checkout(Cart().add("grinder"), "臺北市大安區")
    assert (r["tax"], r["earned"]) == (71, 15)
