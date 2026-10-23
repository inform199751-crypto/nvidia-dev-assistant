"""結帳:算出發票上的每一行(規則見 docs/結帳規則.md)。"""
from billing import coupons, points, regions, shipping, tax


def checkout(cart, address, coupon=None, use_points=0):
    subtotal = cart.subtotal()
    off = coupons.discount(subtotal, coupon)
    after_coupon = subtotal - off
    used = points.usable(after_coupon, use_points)
    payable = after_coupon - used
    ship = shipping.fee(payable, regions.region_of(address))
    total = payable + ship
    return {"subtotal": subtotal, "coupon": off, "points": used, "shipping": ship, "total": total,
            "tax": tax.included_tax(total), "earned": points.earned(payable)}
