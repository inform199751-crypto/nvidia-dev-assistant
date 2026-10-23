"""購物車金額計算。金額一律四捨五入到整數元(0.5 進位)。"""


def line_total(price, qty, discount=0.0):
    """單項小計:單價 × 數量 × (1 - 折扣),四捨五入到元。discount 是 0~1 的比例,例如 0.1 = 打九折。"""
    return round(price * qty * (1 - discount))


def cart_total(lines):
    """lines 是 (單價, 數量, 折扣) 的清單;每一項先各自四捨五入再加總。"""
    return sum(line_total(price, qty, discount) for price, qty, discount in lines)
