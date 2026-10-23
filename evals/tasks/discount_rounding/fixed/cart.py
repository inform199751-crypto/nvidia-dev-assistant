"""購物車金額計算。金額一律四捨五入到整數元(0.5 進位)。"""
from decimal import ROUND_HALF_UP, Decimal


def line_total(price, qty, discount=0.0):
    """單項小計:單價 × 數量 × (1 - 折扣),四捨五入到元。discount 是 0~1 的比例,例如 0.1 = 打九折。"""
    # round() 是「四捨六入五成雙」,浮點數也會讓 2.5 變成 2.4999…;金額用 Decimal 算
    amount = Decimal(str(price)) * qty * (1 - Decimal(str(discount)))
    return int(amount.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def cart_total(lines):
    """lines 是 (單價, 數量, 折扣) 的清單;每一項先各自四捨五入再加總。"""
    return sum(line_total(price, qty, discount) for price, qty, discount in lines)
