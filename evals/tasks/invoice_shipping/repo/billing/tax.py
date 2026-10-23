"""營業稅(規則見 docs/結帳規則.md 第 6 條)。"""


def included_tax(total):
    """內含 5% 營業稅是多少元。"""
    return total - round(total / 1.05)
