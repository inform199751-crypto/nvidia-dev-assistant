"""收據上印的文字。"""

LABELS = [("subtotal", "小計"), ("coupon", "折價券"), ("points", "點數折抵"), ("shipping", "運費"), ("total", "應付金額"),
          ("tax", "內含營業稅")]


def money(n):
    return f"NT${n:,}"


def render(result):
    lines = []
    for key, label in LABELS:
        value = result[key]
        if key in ("coupon", "points") and not value:
            continue
        lines.append(f"{label:<6}{'-' if key in ('coupon', 'points') else ''}{money(value)}")
    return "\n".join(lines)
