"""報表公式用的四則運算(+ - * / 和括號)。照數學規則:先乘除後加減,同一級由左到右算。"""
import re

TOKEN = re.compile(r"\s*(\d+(?:\.\d+)?|[()+\-*/])")


def tokenize(text):
    tokens, pos, text = [], 0, text.strip()
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"看不懂的字元:{text[pos]!r}")
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


def evaluate(text):
    value, rest = _expr(tokenize(text))
    if rest:
        raise ValueError(f"多出來的內容:{''.join(rest)}")
    return value


# 以前右邊遞迴呼叫自己(2-3-4 變成 2-(3-4)),同一級變成由右到左;改成迴圈才是由左到右
def _expr(tokens):
    """加減:term (+|- term)*"""
    value, rest = _term(tokens)
    while rest and rest[0] in ("+", "-"):
        op = rest[0]
        right, rest = _term(rest[1:])
        value = value + right if op == "+" else value - right
    return value, rest


def _term(tokens):
    """乘除:factor (*|/ factor)*"""
    value, rest = _factor(tokens)
    while rest and rest[0] in ("*", "/"):
        op = rest[0]
        right, rest = _factor(rest[1:])
        value = value * right if op == "*" else value / right
    return value, rest


def _factor(tokens):
    """數字、負號、括號。"""
    if not tokens:
        raise ValueError("算式不完整")
    head, rest = tokens[0], tokens[1:]
    if head == "(":
        value, rest = _expr(rest)
        if not rest or rest[0] != ")":
            raise ValueError("括號沒有關")
        return value, rest[1:]
    if head == "-":
        value, rest = _factor(rest)
        return -value, rest
    if head in ("+", "*", "/", ")"):
        raise ValueError(f"這裡不能是 {head}")
    return float(head), rest
