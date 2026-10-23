from billing.cart import Cart
from billing.checkout import checkout
from billing.coupons import Coupon


def test_over_threshold_ships_free():
    r = checkout(Cart().add("grinder"), "臺北市中正區")
    assert (r["shipping"], r["total"]) == (0, 1500)


def test_under_threshold_pays_shipping():
    r = checkout(Cart().add("mug"), "臺北市中正區")
    assert (r["shipping"], r["total"]) == (80, 530)


def test_percent_coupon():
    r = checkout(Cart().add("grinder"), "高雄市", Coupon("OFF10", "percent", 0.9, min_spend=1000))
    assert (r["coupon"], r["total"]) == (150, 1350)


def test_island_shipping():
    r = checkout(Cart().add("grinder"), "澎湖縣馬公市")
    assert r["shipping"] == 150
