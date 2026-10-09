"""小助理執行時用的部分:讀 train_models.py 訓練好的 JSON,純 Python 判斷 / 預估(不載入 numpy、scikit-learn)。

模型放在 OUTPUT_DIR/models/:
  difficulty_tree.json   ID3 決策樹:這題「一般 / 難題」
  cost_linear.json       線性回歸:這一輪大概多少 tokens、幾秒(學的是 log1p,算完再還原)
沒有檔案(還沒訓練)就回 None,小助理照原本的方式運作。
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import tools as toolbox

from . import features as feat
from .id3 import classify

TREE_FILE, COST_FILE = "difficulty_tree.json", "cost_linear.json"
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def model_dir() -> Path:
    return toolbox.OUTPUT_DIR / "models"


def load(name: str) -> dict | None:
    """讀模型;檔案改了(重新訓練)就重讀,沒有或壞掉就 None。"""
    path = model_dir() / name
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    with _lock:
        cached = _cache.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            model = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        _cache[str(path)] = (mtime, model)
        return model


def linear(model: dict, x: list[float]) -> float:
    """θ0 + Σ θi·(xi − 平均i) / 標準差i(標準化用訓練時記下的平均和標準差)。"""
    theta, mean, std = model["theta"], model["mean"], model["std"]
    return theta[0] + sum(t * (v - m) / s for t, v, m, s in zip(theta[1:], x, mean, std))


def guess(f: dict) -> tuple[str, str] | None:
    """決策樹判斷:回傳 ("hard" 或 "normal", 理由);還沒訓練就 None。"""
    tree = load(TREE_FILE)
    if not tree:
        return None
    label, path, counts = classify(tree["tree"], feat.discrete(f))
    steps = "、".join(f"{feat.NAMES.get(k, k)}={v}" for k, v in path) or "只有一個葉子"
    return str(label), f"決策樹:{steps}"


def estimate(f: dict) -> tuple[float, float] | None:
    """線性回歸預估這一輪的 (tokens, 秒);還沒訓練就 None。"""
    model = load(COST_FILE)
    if not model:
        return None
    x = feat.vector(f, model["tokens"]["names"])
    tokens = math.expm1(linear(model["tokens"], x))
    seconds = math.expm1(linear(model["seconds"], feat.vector(f, model["seconds"]["names"])))
    return max(tokens, 0.0), max(seconds, 0.0)


def describe(tokens: float, seconds: float) -> str:
    """給人看的預估:「約 1.2 萬 tokens、約 25 秒」。"""
    t = f"{tokens / 10000:.1f} 萬" if tokens >= 10000 else f"{tokens:,.0f}"
    s = f"{seconds / 60:.1f} 分鐘" if seconds >= 90 else f"{seconds:.0f} 秒"
    return f"約 {t} tokens、約 {s}"
