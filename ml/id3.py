"""ID3 決策樹:照課程「決策樹_代碼實現」的 DecisionTree.py 寫(函式名稱、用資訊熵挑特徵的步驟都一樣)。

跟課程版不一樣的地方(都是拿來用在真實資料時需要的):
1. 樹的深度、節點最少幾筆可以限制(課程「決策樹算法_實驗」講的正規化:max_depth、min_samples_split),
   不然 80 筆資料會長成一筆一個葉子,過度擬合。
2. 樣本可以有權重:難題比一般題少很多,不加權的話樹只要全部猜「一般」就有八成準,學不到難題。
3. 每個節點記下多數決的答案:預測時遇到訓練沒看過的特徵值(例如某個等級沒出現過)就用它,不會出錯。
4. 課程版在「每個特徵的資訊增益都 ≤ 0」時 bestFeature 會是 -1,labels[-1] 會拿到最後一個特徵去切;
   這裡改成直接當葉子。
5. 樹存成 {"feature", "children", ...} 的字典(可以直接存 JSON),不是課程版的 {特徵名: {值: 子樹}};
   畫圖的方式(plotTree)沿用課程版。

只用標準函式庫:小助理執行時判斷一題只要走一次樹,不用載入 numpy。
"""
from __future__ import annotations

from math import log


def calcShannonEnt(dataset: list[list], weights: list[float]) -> float:
    """資訊熵:-Σ p·log2(p),p 是每個類別的(加權)比例。"""
    total = sum(weights)
    counts: dict = {}
    for row, w in zip(dataset, weights):
        counts[row[-1]] = counts.get(row[-1], 0) + w
    return -sum(c / total * log(c / total, 2) for c in counts.values() if c > 0)


def splitDataSet(dataset: list[list], weights: list[float], axis: int, val) -> tuple[list[list], list[float]]:
    """第 axis 個特徵等於 val 的那些資料,並拿掉這一欄。"""
    rows, ws = [], []
    for row, w in zip(dataset, weights):
        if row[axis] == val:
            rows.append(row[:axis] + row[axis + 1:])
            ws.append(w)
    return rows, ws


def chooseBestFeatureToSplit(dataset: list[list], weights: list[float]) -> tuple[int, float]:
    """資訊增益最大的特徵:切之前的熵 − 切之後各組熵的加權平均。回傳 (第幾個特徵, 增益)。"""
    base = calcShannonEnt(dataset, weights)
    total = sum(weights)
    best, best_gain = -1, 0.0
    for i in range(len(dataset[0]) - 1):
        new = 0.0
        for val in {row[i] for row in dataset}:
            sub, ws = splitDataSet(dataset, weights, i, val)
            new += sum(ws) / total * calcShannonEnt(sub, ws)
        if base - new > best_gain + 1e-12:
            best, best_gain = i, base - new
    return best, best_gain


def majorityCnt(classList: list, weights: list[float]):
    counts: dict = {}
    for c, w in zip(classList, weights):
        counts[c] = counts.get(c, 0) + w
    return max(sorted(counts), key=lambda c: counts[c])  # 先排序:同票時結果固定


def _counts(classList: list) -> dict:
    out: dict = {}
    for c in classList:
        out[c] = out.get(c, 0) + 1
    return out


def createTree(dataset: list[list], labels: list[str], weights: list[float] | None = None,
               max_depth: int | None = None, min_samples_split: int = 2, _depth: int = 0) -> dict:
    """dataset 每一列是 [特徵1, 特徵2, ..., 類別];labels 是特徵名稱。回傳樹(葉子是 {"label": ...})。"""
    weights = weights or [1.0] * len(dataset)
    classList = [row[-1] for row in dataset]
    node = {"label": majorityCnt(classList, weights), "counts": _counts(classList)}
    if (classList.count(classList[0]) == len(classList) or len(dataset[0]) == 1
            or len(dataset) < min_samples_split or (max_depth is not None and _depth >= max_depth)):
        return node
    bestFeat, gain = chooseBestFeatureToSplit(dataset, weights)
    if bestFeat < 0:  # 怎麼切都沒有資訊增益
        return node
    sublabels = labels[:bestFeat] + labels[bestFeat + 1:]
    children = {}
    for value in sorted({row[bestFeat] for row in dataset}, key=str):
        sub, ws = splitDataSet(dataset, weights, bestFeat, value)
        children[str(value)] = createTree(sub, sublabels, ws, max_depth, min_samples_split, _depth + 1)
    return {"feature": labels[bestFeat], "gain": round(gain, 4), "default": node["label"],
            "counts": node["counts"], "children": children}


def classify(tree: dict, sample: dict) -> tuple[object, list[tuple[str, str]], dict]:
    """sample 是 {特徵名: 值}。回傳 (類別, 走過的路 [(特徵, 值)], 葉子的各類別筆數)。"""
    path = []
    node = tree
    while "feature" in node:
        value = str(sample.get(node["feature"]))
        path.append((node["feature"], value))
        child = node["children"].get(value)
        if child is None:  # 訓練時沒看過這個值:用這個節點的多數決
            return node["default"], path, node["counts"]
        node = child
    return node["label"], path, node["counts"]


def getNumLeafs(tree: dict) -> int:
    return sum(getNumLeafs(c) for c in tree["children"].values()) if "feature" in tree else 1


def getTreeDepth(tree: dict) -> int:
    return 1 + max(getTreeDepth(c) for c in tree["children"].values()) if "feature" in tree else 0


def createPlot(tree: dict, path, names: dict | None = None, label_names: dict | None = None,
               font=None, title: str = "", binary: set[str] = frozenset()) -> None:
    """畫樹狀圖存成 PNG,畫法沿用課程版的 plotTree(節點、箭頭、邊上寫特徵值)。binary 裡的特徵,0 / 1 寫成否 / 是。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, label_names = names or {}, label_names or {}
    leafs, depth = getNumLeafs(tree), max(getTreeDepth(tree), 1)
    fig = plt.figure(figsize=(max(8, leafs * 1.6), max(4, depth * 2.2)), facecolor="white")
    ax = plt.subplot(111, frameon=False, xticks=[], yticks=[])
    decision = dict(boxstyle="sawtooth", fc="0.85")
    leaf = dict(boxstyle="round4", fc="0.92")
    state = {"x": -0.5 / leafs, "y": 1.0}

    def node_text(n: dict) -> str:
        counts = " / ".join(f"{label_names.get(k, k)} {v}" for k, v in sorted(n["counts"].items(), key=str))
        if "feature" in n:
            return f"{names.get(n['feature'], n['feature'])}?\n({counts})"
        return f"{label_names.get(n['label'], n['label'])}\n({counts})"

    def draw(text, center, parent, style):
        ax.annotate(text, xy=parent, xycoords="axes fraction", xytext=center, textcoords="axes fraction",
                    va="center", ha="center", bbox=style, arrowprops=dict(arrowstyle="<-"), fontproperties=font,
                    fontsize=10)

    def mid(center, parent, text):
        ax.text((parent[0] + center[0]) / 2, (parent[1] + center[1]) / 2, text, va="center", ha="center",
                rotation=30, fontproperties=font, fontsize=9)

    def plot(n, parent, edge):
        if "feature" not in n:
            state["x"] += 1.0 / leafs
            draw(node_text(n), (state["x"], state["y"]), parent, leaf)
            mid((state["x"], state["y"]), parent, edge)
            return
        center = (state["x"] + (1.0 + getNumLeafs(n)) / 2.0 / leafs, state["y"])
        mid(center, parent, edge)
        draw(node_text(n), center, parent, decision)
        state["y"] -= 1.0 / depth
        for value, child in n["children"].items():
            plot(child, center, {"0": "否", "1": "是"}.get(value, value) if n["feature"] in binary else value)
        state["y"] += 1.0 / depth

    plot(tree, (0.5, 1.0), "")
    if title:  # 放在圖的上方,不要壓到最上面的節點
        fig.suptitle(title, fontproperties=font, y=1.04)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def createDataSet():
    """課程版的範例資料(貸款:年齡、有工作、有房子、信用),測試用來確認演算法跟課程版一樣。"""
    dataSet = [[0, 0, 0, 0, 'no'], [0, 0, 0, 1, 'no'], [0, 1, 0, 1, 'yes'], [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'], [1, 0, 0, 0, 'no'], [1, 0, 0, 1, 'no'], [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'], [1, 0, 1, 2, 'yes'], [2, 0, 1, 2, 'yes'], [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'], [2, 1, 0, 2, 'yes'], [2, 0, 0, 0, 'no']]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataSet, labels
