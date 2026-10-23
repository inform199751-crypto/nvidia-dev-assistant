"""會員點數(規則見 docs/結帳規則.md 第 3 條)。"""

MAX_RATE = 0.3  # 最多折折價券折抵後金額的 30%


def usable(amount, points):
    """這次最多可以用幾點(1 點 1 元)。"""
    return max(0, min(points, int(amount * MAX_RATE)))


def earned(paid):
    """這筆消費回饋幾點:每 100 元 1 點(運費不算)。"""
    return paid // 100
