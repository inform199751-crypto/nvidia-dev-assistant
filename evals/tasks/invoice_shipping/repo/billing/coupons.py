"""折價券(規則見 docs/結帳規則.md 第 2 條)。"""
from dataclasses import dataclass


@dataclass
class Coupon:
    code: str
    kind: str           # "percent"(value=0.9 是 9 折)或 "fixed"(value=100 是折 100 元)
    value: float
    min_spend: int = 0  # 小計要到多少才能用


def discount(subtotal, coupon):
    """這張券可以折多少元(四捨五入到元);沒到門檻回傳 0。"""
    if coupon is None or subtotal < coupon.min_spend:
        return 0
    if coupon.kind == "percent":
        # 10/20 改過:以前是無條件捨去,會計說要四捨五入
        # TODO: .5 的時候要進位還是捨去,再跟會計確認一次
        return round(subtotal * (1 - coupon.value))
    if coupon.kind == "fixed":
        return min(int(coupon.value), subtotal)
    raise ValueError(f"不認得的折價券種類:{coupon.kind}")
