"""運費(規則見 docs/結帳規則.md 第 4 條)。"""

RULES = {"本島": (1000, 80), "離島": (2000, 150)}  # 地區 → (免運門檻, 運費)


def fee(amount, region="本島"):
    """amount 要給「折價券折抵後、點數折抵前」的金額。"""
    threshold, cost = RULES[region]
    return 0 if amount > threshold else cost
