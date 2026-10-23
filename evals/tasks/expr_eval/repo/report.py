"""月報:把公式欄位的變數換成數字再計算。"""
import re

from expr import evaluate


def compute(formula, values):
    """formula 像「營收 - 成本 - 退款」;values 是名稱 → 數字。"""
    def sub(m):
        return f"({values[m.group(0)]})"
    return evaluate(re.sub(r"[^\d\s()+\-*/.]+", sub, formula))
