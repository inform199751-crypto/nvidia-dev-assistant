"""QR code(只用標準函式庫):桌面版「手機連線」把網址畫成 QR code,手機相機一掃就打開。

照 ISO/IEC 18004 的作法,只做需要的部分:byte 模式、錯誤更正等級 M、版本 1–6(最多 106 個位元組,區域網路網址夠用)。
版本 7 以上要另外畫版本資訊,用不到就不做。
"""
from __future__ import annotations

# 等級 M、版本 1–6:每個區塊的更正碼數、區塊數(標準的表;索引 0 不用)
_ECC_PER_BLOCK = [0, 10, 16, 26, 18, 24, 16]
_BLOCKS = [0, 1, 1, 1, 2, 2, 4]
MAX_VERSION = 6

# 八種遮罩:回傳 0 的位置要反轉
_MASKS = [
    lambda x, y: (x + y) % 2,
    lambda x, y: y % 2,
    lambda x, y: x % 3,
    lambda x, y: (x + y) % 3,
    lambda x, y: (x // 3 + y // 2) % 2,
    lambda x, y: x * y % 2 + x * y % 3,
    lambda x, y: (x * y % 2 + x * y % 3) % 2,
    lambda x, y: ((x + y) % 2 + x * y % 3) % 2,
]


def _gf_mul(x: int, y: int) -> int:
    """GF(256) 乘法,模 x^8 + x^4 + x^3 + x^2 + 1(0x11D)。"""
    z = 0
    for i in reversed(range(8)):
        z = (z << 1) ^ ((z >> 7) * 0x11D)
        z ^= ((y >> i) & 1) * x
    return z


def _rs_divisor(degree: int) -> list[int]:
    """Reed-Solomon 生成多項式(最高次項省略),係數從高次到低次。"""
    result = [0] * (degree - 1) + [1]
    root = 1
    for _ in range(degree):
        for j in range(degree):
            result[j] = _gf_mul(result[j], root)
            if j + 1 < degree:
                result[j] ^= result[j + 1]
        root = _gf_mul(root, 0x02)
    return result


def _rs_remainder(data: list[int], divisor: list[int]) -> list[int]:
    """更正碼 = 資料多項式除以生成多項式的餘數。"""
    result = [0] * len(divisor)
    for b in data:
        factor = b ^ result.pop(0)
        result.append(0)
        for i, coef in enumerate(divisor):
            result[i] ^= _gf_mul(coef, factor)
    return result


def _raw_modules(ver: int) -> int:
    """扣掉定位、對齊、時序、格式資訊後,能放資料的格子數。"""
    result = (16 * ver + 128) * ver + 64
    if ver >= 2:
        n = ver // 7 + 2
        result -= (25 * n - 10) * n - 55
        if ver >= 7:
            result -= 36
    return result


def _capacity(ver: int) -> int:
    """資料碼字(位元組)數。"""
    return _raw_modules(ver) // 8 - _ECC_PER_BLOCK[ver] * _BLOCKS[ver]


def format_bits(mask: int) -> int:
    """15 位元的格式資訊:等級 M(代碼 00)+ 遮罩編號,加 BCH 更正碼再 XOR 0x5412。"""
    data = mask  # 等級 M 是 0b00,放在最高兩位
    rem = data
    for _ in range(10):
        rem = (rem << 1) ^ ((rem >> 9) * 0x537)
    return (data << 10 | rem) ^ 0x5412


class _Grid:
    def __init__(self, ver: int):
        self.size = 4 * ver + 17
        self.dark = [[False] * self.size for _ in range(self.size)]
        self.fixed = [[False] * self.size for _ in range(self.size)]  # 定位圖案等:不放資料、不套遮罩

    def set(self, x: int, y: int, dark: bool) -> None:
        self.dark[y][x] = dark
        self.fixed[y][x] = True


def _draw_format(g: _Grid, mask: int) -> None:
    bits = format_bits(mask)
    bit = lambda i: ((bits >> i) & 1) == 1  # noqa: E731
    n = g.size
    for i in range(6):  # 左上角那一份
        g.set(8, i, bit(i))
    g.set(8, 7, bit(6))
    g.set(8, 8, bit(7))
    g.set(7, 8, bit(8))
    for i in range(9, 15):
        g.set(14 - i, 8, bit(i))
    for i in range(8):  # 右上 + 左下那一份
        g.set(n - 1 - i, 8, bit(i))
    for i in range(8, 15):
        g.set(8, n - 15 + i, bit(i))
    g.set(8, n - 8, True)  # 固定的暗點


def function_patterns(ver: int) -> _Grid:
    """時序線、三個定位圖案、對齊圖案(版本 2–6 只有右下一個)、格式資訊的位置。"""
    g = _Grid(ver)
    n = g.size
    for i in range(n):
        g.set(6, i, i % 2 == 0)
        g.set(i, 6, i % 2 == 0)
    for cx, cy in ((3, 3), (n - 4, 3), (3, n - 4)):
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                x, y = cx + dx, cy + dy
                if 0 <= x < n and 0 <= y < n:
                    g.set(x, y, max(abs(dx), abs(dy)) not in (2, 4))
    if ver >= 2:
        c = n - 7
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                g.set(c + dx, c + dy, max(abs(dx), abs(dy)) != 1)
    _draw_format(g, 0)  # 先佔位置,選好遮罩再畫真的
    return g


def codewords(data: bytes, ver: int) -> list[int]:
    """資料 → 碼字:模式、長度、內容、結尾、補齊,分區塊加 Reed-Solomon 更正碼,再交錯排好。"""
    cap = _capacity(ver)
    bits: list[int] = []

    def put(val: int, n: int) -> None:
        bits.extend((val >> i) & 1 for i in reversed(range(n)))

    put(0b0100, 4)  # byte 模式
    put(len(data), 8)  # 版本 1–9 的長度欄位是 8 位元
    for b in data:
        put(b, 8)
    put(0, min(4, cap * 8 - len(bits)))
    put(0, -len(bits) % 8)
    pad = 0xEC
    while len(bits) < cap * 8:
        put(pad, 8)
        pad ^= 0xEC ^ 0x11
    words = [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)]

    nblocks, ecc = _BLOCKS[ver], _ECC_PER_BLOCK[ver]
    raw = _raw_modules(ver) // 8
    short = nblocks - raw % nblocks  # 前面這幾個區塊少一個資料碼字
    short_len = raw // nblocks
    divisor = _rs_divisor(ecc)
    blocks, k = [], 0
    for i in range(nblocks):
        dat = words[k:k + short_len - ecc + (0 if i < short else 1)]
        k += len(dat)
        rem = _rs_remainder(dat, divisor)
        if i < short:
            dat = dat + [0]  # 佔位,交錯時跳過
        blocks.append(dat + rem)
    out = []
    for i in range(len(blocks[0])):
        for j, b in enumerate(blocks):
            if i != short_len - ecc or j >= short:
                out.append(b[i])
    return out


def _place(g: _Grid, words: list[int]) -> None:
    """從右下角開始,兩欄一組上下來回走,跳過固定圖案;第 6 欄是時序線,整組往左挪一格。"""
    n, i, total = g.size, 0, len(words) * 8
    for right in range(n - 1, 0, -2):
        if right <= 6:
            right -= 1
        upward = ((right + 1) & 2) == 0
        for vert in range(n):
            y = n - 1 - vert if upward else vert
            for x in (right, right - 1):
                if not g.fixed[y][x] and i < total:
                    g.dark[y][x] = ((words[i >> 3] >> (7 - (i & 7))) & 1) == 1
                    i += 1


def _apply_mask(g: _Grid, mask: int) -> None:
    f = _MASKS[mask]
    for y in range(g.size):
        for x in range(g.size):
            if not g.fixed[y][x] and f(x, y) == 0:
                g.dark[y][x] = not g.dark[y][x]


def _penalty(g: _Grid) -> int:
    """選遮罩用的分數(越低越好掃):同色連續 5 格以上、2×2 同色方塊、深淺比例。省略「像定位圖案」那條,不影響正確性。"""
    n, score = g.size, 0
    lines = [g.dark[y] for y in range(n)] + [[g.dark[y][x] for y in range(n)] for x in range(n)]
    for line in lines:
        run = 1
        for a, b in zip(line, line[1:]):
            run = run + 1 if a == b else 1
            if run == 5:
                score += 3
            elif run > 5:
                score += 1
    for y in range(n - 1):
        for x in range(n - 1):
            if g.dark[y][x] == g.dark[y][x + 1] == g.dark[y + 1][x] == g.dark[y + 1][x + 1]:
                score += 3
    dark = sum(map(sum, g.dark))
    score += abs(dark * 20 - n * n * 10) // (n * n) * 10
    return score


def encode(data: bytes) -> list[list[bool]]:
    """回傳 QR code 的格子(True = 黑),不含外圍留白。"""
    ver = next((v for v in range(1, MAX_VERSION + 1) if 12 + 8 * len(data) <= _capacity(v) * 8), None)
    if ver is None:
        raise ValueError(f"太長了:QR code 最多放 {_capacity(MAX_VERSION) - 2} 個位元組")
    g = function_patterns(ver)
    _place(g, codewords(data, ver))
    best = None
    for mask in range(8):
        _apply_mask(g, mask)
        _draw_format(g, mask)
        score = _penalty(g)
        if best is None or score < best[0]:
            best = (score, mask)
        _apply_mask(g, mask)  # 還原
    _apply_mask(g, best[1])
    _draw_format(g, best[1])
    return g.dark


def svg(text: str, border: int = 4) -> str:
    """畫成 SVG:白底黑格(深色介面也一樣,相機才掃得到),外圍留 4 格白邊。"""
    grid = encode(text.encode("utf-8"))
    n = len(grid) + 2 * border
    path = "".join(f"M{x + border} {y + border}h1v1h-1z" for y, row in enumerate(grid) for x, dark in enumerate(row) if dark)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" shape-rendering="crispEdges" '
            f'role="img" aria-label="QR code"><rect width="{n}" height="{n}" fill="#fff"/>'
            f'<path d="{path}" fill="#000"/></svg>')
