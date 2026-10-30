"""訓練小助理的小模型,並做比較實驗(專題用):

    .venv/Scripts/python train_models.py --label   # 先請 LLM 分類器標註 ml/seed_questions.txt(約 80 次呼叫)
    .venv/Scripts/python train_models.py           # 訓練 + 實驗,結果寫到 output/models/

資料:ml/seed_questions.txt 標註好的範例 + 小助理每輪寫的 routing_log.jsonl(有 features 的那些)。

實驗一「這題難不難」(分類):手寫 ID3 決策樹(照課程 DecisionTree.py)、手寫邏輯回歸(照課程 logistic_regression.py),
  跟 scikit-learn 的決策樹、邏輯回歸、隨機森林、神經網路(MLP)比較。評估照課程「模型評估方法」:
  分層 K 折交叉驗證、混淆矩陣、精確率 / 召回率 / F1、ROC 曲線與 AUC;基準是「全部猜多數類別」。
  標籤是 LLM 分類器(Super)的判斷 —— 決策樹學得夠像,就能在 models.yaml 設 classifier: tree 取代它。
實驗二「這一輪花多少」(回歸,要有真實紀錄):手寫線性回歸(照課程 linear_regression.py,梯度下降)跟 scikit-learn 的
  線性回歸、Ridge、隨機森林、MLP 比較;K 折交叉驗證看 MAE、RMSE,畫學習曲線(照課程「線性回歸實驗」)。
  tokens 和秒數差很多倍(幾千到幾十萬),所以學的是 log1p,算誤差時再還原。

訓練好的模型存成 JSON(output/models/difficulty_tree.json、cost_linear.json),小助理執行時用 ml/runtime.py 讀。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
SEEDS = HERE / "ml" / "seed_questions.txt"
MIN_CLASSIFY = 20       # 少於這麼多筆標註就不訓練分類
MIN_PER_CLASS = 4       # 每一類至少幾筆(交叉驗證的每一折才分得到)
MIN_REGRESS = 30        # 少於這麼多輪真實紀錄就不訓練回歸
LABELS = {"hard": "難題", "normal": "一般"}


# ---------- 資料 ----------

def seed_questions(path: Path = SEEDS) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def label_seeds(out_dir: Path, pause: float = 1.6) -> Path:
    """請 LLM 分類器標註範例問題,存 seed_labels.jsonl;已經標過的跳過(可以中斷後重跑)。"""
    import agent
    import difficulty
    from langchain_core.messages import SystemMessage

    from ml import features

    registry, quota, _, _ = agent.prepare()
    cfg = registry.routing.get("difficulty") or {}
    if not cfg:
        raise SystemExit("models.yaml 沒有 routing.difficulty 設定,沒有分類器可以標註")
    path = out_dir / "seed_labels.jsonl"
    done = {json.loads(line)["task"] for line in path.read_text(encoding="utf-8").splitlines()} if path.is_file() else set()
    todo = [q for q in seed_questions() if q not in done]
    print(f"範例問題 {len(done) + len(todo)} 題,已標註 {len(done)},這次要標 {len(todo)} 題")
    out_dir.mkdir(parents=True, exist_ok=True)
    spent = 0
    for i, q in enumerate(todo, 1):
        for attempt in range(2):
            try:
                text, tokens = difficulty.ask_llm(registry, quota, "train", [SystemMessage("")], q, [], cfg)
                level, reason = difficulty.parse(text)
                break
            except Exception as e:  # 免費層偶爾 429 / 503:等一下再試一次
                if attempt:
                    print(f"  {i:>3}. 失敗,跳過:{type(e).__name__}: {str(e)[:80]}")
                    level = None
                time.sleep(5)
        if level is None:
            continue
        spent += tokens
        row = {"task": q, "label": level, "reason": reason, "features": features.extract(q)}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {i:>3}. {LABELS[level]}  {q[:40]}  ({reason})")
        time.sleep(pause)  # 免費層有每分鐘次數限制
    print(f"完成,分類器共用了 {spent:,} tokens → {path}")
    return path


def load_rows(out_dir: Path, log_path: Path, hard_role: str, backend: str | None = None) -> tuple[list[dict], list[dict]]:
    """回傳 (分類資料, 回歸資料)。分類:範例標註 + 紀錄裡 LLM 分類器判斷過的輪(決策樹自己判斷的不算,不然是自己學自己);
    回歸:紀錄裡正常結束、有 tokens 和時間的輪。backend:只用這個後端的紀錄(地端測試的小模型判斷和秒數不能混進來)。"""
    import difficulty
    from ml import features

    classify, regress = [], []
    seed = out_dir / "seed_labels.jsonl"
    if seed.is_file():
        for line in seed.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)  # 範例有原文:特徵每次重算,改了特徵定義也不用重新標註
            classify.append({"features": features.extract(row["task"]), "label": row["label"], "source": "範例"})
    logs = [p for p in (log_path.with_name(log_path.name + ".1"), log_path) if p.is_file()]
    for p in logs:
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not e.get("features") or backend and difficulty.backend_of(e) != backend:
                continue
            if e.get("by") == "classifier" and e.get("engine", "llm") == "llm" and e.get("pick"):
                classify.append({"features": e["features"], "label": "hard" if e["pick"] == hard_role else "normal",
                                 "source": "紀錄"})
            if e.get("outcome") in ("done", "max_steps") and e.get("tokens") and e.get("ms"):
                regress.append({"features": e["features"], "tokens": e["tokens"], "seconds": e["ms"] / 1000})
    return classify, regress


# ---------- 畫圖 ----------

def setup_font():
    """matplotlib 預設字型沒有中文:照課程 DecisionTree.py 的作法找系統裡的中文字型。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    keys = ("msjh", "jhenghei", "mingliu", "pingfang", "heiti", "notosanscjk", "sourcehansans", "yahei", "simhei")
    for path in fm.findSystemFonts():
        if any(k in Path(path).name.lower() for k in keys):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.sans-serif"] = [prop.get_name(), *plt.rcParams["font.sans-serif"]]
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    return None


# ---------- 實驗一:分類 ----------

def _weights(y: list[int]) -> list[float]:
    """balanced 類別權重(跟 scikit-learn 的 class_weight="balanced" 一樣):n / (類別數 × 這一類的筆數)。"""
    n, pos = len(y), sum(y)
    return [n / (2 * pos) if v else n / (2 * (n - pos)) for v in y]


def classification_experiment(rows: list[dict], out: Path, max_depth: int | None = None, folds: int = 5) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression as SkLogistic
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                                 roc_auc_score, roc_curve)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    from ml import features as F
    from ml import id3
    from ml.regression import LogisticRegression

    font = setup_font()
    y = np.array([int(r["label"] == "hard") for r in rows])
    X = np.array([F.vector(r["features"]) for r in rows])
    D = [F.discrete(r["features"]) for r in rows]
    n_hard = int(y.sum())
    k = max(2, min(folds, n_hard, len(y) - n_hard))
    names = [F.NAMES.get(n, n) for n in F.VECTOR]

    def fit_id3(idx, depth):
        data = [[D[i][f] for f in F.DISCRETE] + ["hard" if y[i] else "normal"] for i in idx]
        return id3.createTree(data, list(F.DISCRETE), _weights([y[i] for i in idx]), max_depth=depth,
                              min_samples_split=4)

    def id3_predict(tree, idx):
        out = []
        for i in idx:
            label, _, counts = id3.classify(tree, D[i])
            total = sum(counts.values())
            out.append((int(label == "hard"), counts.get("hard", 0) / total if total else 0.0))
        return out

    # 每個模型:(名字, 訓練後回傳「預測 [(類別, 難題機率)]」的函式)
    def sk(model):
        def run(tr, te):
            model.fit(X[tr], y[tr])
            proba = model.predict_proba(X[te])[:, 1]
            return list(zip(model.predict(X[te]).tolist(), proba.tolist()))
        return run

    def hand_logistic(tr, te):
        m = LogisticRegression(X[tr], y[tr], sample_weight=_weights(y[tr].tolist()))
        m.train()
        p = m.predict_proba(X[te])
        return list(zip((p >= 0.5).astype(int).tolist(), p.tolist()))

    # 交叉驗證:照課程「模型評估方法」的 StratifiedKFold 寫法,每一折重新訓練、預測沒看過的那一折
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    def run_cv(run):
        pred, proba = np.zeros(len(y), int), np.zeros(len(y))
        for tr, te in cv.split(X, y):
            for i, (label, p) in zip(te, run(tr, te)):
                pred[i], proba[i] = label, p
        return pred, proba

    def sk_tree(depth):
        return DecisionTreeClassifier(criterion="entropy", max_depth=depth, class_weight="balanced", random_state=42)

    # 樹的深度實驗(課程「決策樹算法_實驗」的正規化):太淺學不到、太深過度擬合;沒指定就挑 ID3 的 F1 最好的深度
    sweep = []
    for depth in range(1, 7):
        p_id3 = run_cv(lambda tr, te, d=depth: id3_predict(fit_id3(tr, d), te))[0]
        p_sk = run_cv(sk(sk_tree(depth)))[0]
        sweep.append({"depth": depth, "id3_acc": accuracy_score(y, p_id3), "id3_f1": f1_score(y, p_id3, zero_division=0),
                      "sk_acc": accuracy_score(y, p_sk), "sk_f1": f1_score(y, p_sk, zero_division=0)})
    if max_depth is None:
        max_depth = max(sweep, key=lambda s: (round(s["id3_f1"], 3), -s["depth"]))["depth"]
    fig = plt.figure(figsize=(8, 4))
    depths = [s["depth"] for s in sweep]
    for key, style, label in (("id3_f1", "b-o", "ID3(手寫)F1"), ("id3_acc", "b--", "ID3(手寫)準確率"),
                              ("sk_f1", "g-s", "scikit-learn F1"), ("sk_acc", "g--", "scikit-learn 準確率")):
        plt.plot(depths, [s[key] for s in sweep], style, label=label)
    plt.axvline(max_depth, color="k", linestyle=":", label=f"選用深度 {max_depth}")
    plt.xlabel("樹的最大深度")
    plt.ylabel("交叉驗證分數")
    plt.title("樹的深度對結果的影響")
    plt.legend(fontsize=8)
    fig.savefig(out / "depth.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    models = [
        (f"ID3 決策樹(手寫,課程版,深度 {max_depth})", lambda tr, te: id3_predict(fit_id3(tr, max_depth), te)),
        ("邏輯回歸(手寫,課程版)", hand_logistic),
        (f"決策樹(scikit-learn,深度 {max_depth})", sk(sk_tree(max_depth))),
        ("邏輯回歸(scikit-learn)", sk(Pipeline([("std", StandardScaler()),
                                                ("lr", SkLogistic(class_weight="balanced", max_iter=1000))]))),
        ("隨機森林(scikit-learn)", sk(RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                             random_state=42))),
        ("神經網路 MLP(scikit-learn)", sk(Pipeline([("std", StandardScaler()),
                                                    ("mlp", MLPClassifier(hidden_layer_sizes=(16,), max_iter=3000,
                                                                          random_state=42))]))),
    ]
    results = {}
    for name, run in models:
        pred, proba = run_cv(run)
        results[name] = {"pred": pred, "proba": proba}
    majority = int(y.mean() >= 0.5)
    results["基準:全部猜「" + LABELS["hard" if majority else "normal"] + "」"] = {
        "pred": np.full(len(y), majority), "proba": np.full(len(y), float(majority))}

    table = []
    for name, r in results.items():
        p = r["pred"]
        auc = roc_auc_score(y, r["proba"]) if len(set(r["proba"])) > 1 else 0.5
        table.append({"model": name, "accuracy": accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0),
                      "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0), "auc": auc})

    # 圖:混淆矩陣(ID3、手寫邏輯回歸)、ROC 曲線、兩棵樹
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, list(results)[:2]):
        cm = confusion_matrix(y, results[name]["pred"], labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=16, color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["預測一般", "預測難題"])
        ax.set_yticks([0, 1], ["實際一般", "實際難題"])
        ax.set_title(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    for name, r in list(results.items())[:-1]:
        fpr, tpr, _ = roc_curve(y, r["proba"])
        plt.plot(fpr, tpr, linewidth=2, label=f"{name}(AUC {roc_auc_score(y, r['proba']):.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="隨機猜")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate(召回率)")
    plt.title("ROC 曲線(交叉驗證)")
    plt.legend(fontsize=8, loc="lower right")
    fig.savefig(out / "roc.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    tree = fit_id3(range(len(y)), max_depth)  # 正式用的:全部資料訓練
    id3.createPlot(tree, out / "tree_id3.png", F.NAMES, LABELS, font, f"ID3 決策樹(手寫,課程版,深度 {max_depth})",
                   binary={"code", "followup_short", *F.KEYWORDS})
    fig = plt.figure(figsize=(16, 8))
    plot_tree(sk_tree(max_depth).fit(X, y), feature_names=names, class_names=["一般", "難題"], filled=True,
              rounded=True, fontsize=8)
    fig.savefig(out / "tree_sklearn.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    id3_row = table[0]
    model = {"kind": "id3", "trained": datetime.now().isoformat(timespec="seconds"), "samples": len(y),
             "hard": n_hard, "max_depth": max_depth, "features": F.DISCRETE, "cv": {k_: round(v, 3)
             for k_, v in id3_row.items() if k_ != "model"}, "tree": tree}
    (out / "difficulty_tree.json").write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8")
    importance = sorted(zip(names, RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
                            .fit(X, y).feature_importances_), key=lambda t: -t[1])
    return {"n": len(y), "hard": n_hard, "folds": k, "table": table, "tree": tree, "importance": importance,
            "depth": max_depth, "sweep": sweep}


# ---------- 實驗二:回歸 ----------

def regression_experiment(rows: list[dict], out: Path, folds: int = 5) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression as SkLinear
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from ml import features as F
    from ml.regression import LinearRegression

    setup_font()
    X = np.array([F.vector(r["features"]) for r in rows])
    names = [F.NAMES.get(n, n) for n in F.VECTOR]
    cv = KFold(n_splits=min(folds, len(rows)), shuffle=True, random_state=42)
    report, saved = {}, {"kind": "linear", "trained": datetime.now().isoformat(timespec="seconds"), "samples": len(rows)}
    for target, unit in (("tokens", "tokens"), ("seconds", "秒")):
        y_raw = np.array([r[target] for r in rows], float)
        y = np.log1p(y_raw)

        def hand(tr, te):
            m = LinearRegression(X[tr], y[tr])
            m.train(alpha=0.05, num_iterations=3000)
            return m.predict(X[te]).ravel()

        def sk(model):
            def run(tr, te):
                return model.fit(X[tr], y[tr]).predict(X[te])
            return run

        models = [("線性回歸(手寫,梯度下降)", hand), ("線性回歸(scikit-learn)", sk(SkLinear())),
                  ("Ridge 正規化(alpha=1)", sk(Ridge(alpha=1.0))),
                  ("隨機森林", sk(RandomForestRegressor(n_estimators=200, random_state=42))),
                  ("神經網路 MLP", sk(Pipeline([("std", StandardScaler()),
                                               ("mlp", MLPRegressor(hidden_layer_sizes=(16,), max_iter=5000,
                                                                    random_state=42))]))),
                  ("基準:猜平均", lambda tr, te: np.full(len(te), y[tr].mean()))]
        rows_out = []
        for name, run in models:
            pred = np.zeros(len(y))
            for tr, te in cv.split(X):
                pred[te] = run(tr, te)
            back = np.expm1(pred)
            rows_out.append({"model": name, "mae": mean_absolute_error(y_raw, back),
                             "rmse": math.sqrt(mean_squared_error(y_raw, back)), "r2_log": r2_score(y, pred),
                             "pred": back})
        report[target] = {"unit": unit, "table": rows_out}

        final = LinearRegression(X, y, names=list(F.VECTOR))  # 正式用的:全部資料訓練
        _, cost = final.train(alpha=0.05, num_iterations=3000)
        saved[target] = {**final.to_json(), "mae": round(rows_out[0]["mae"], 1)}
        report[target]["coef"] = sorted(zip(names, final.theta.ravel()[1:].tolist()), key=lambda t: -abs(t[1]))

        # 圖:預測 vs 實際、梯度下降的 cost、學習曲線(照課程 plot_learning_curves)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].scatter(y_raw, rows_out[0]["pred"], s=14)
        top = max(y_raw.max(), rows_out[0]["pred"].max())
        axes[0].plot([0, top], [0, top], "k--")
        axes[0].set_xlabel(f"實際 {unit}")
        axes[0].set_ylabel(f"預估 {unit}")
        axes[0].set_title("預估 vs 實際(手寫線性回歸,交叉驗證)")
        axes[1].plot(cost)
        axes[1].set_xlabel("迭代次數")
        axes[1].set_ylabel("cost")
        axes[1].set_title("梯度下降的 cost")
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=100)
        tr_err, val_err = [], []
        sizes = range(2, len(X_tr) + 1)
        for m in sizes:
            model = SkLinear().fit(X_tr[:m], y_tr[:m])
            tr_err.append(math.sqrt(mean_squared_error(y_tr[:m], model.predict(X_tr[:m]))))
            val_err.append(math.sqrt(mean_squared_error(y_val, model.predict(X_val))))
        axes[2].plot(list(sizes), tr_err, "r-+", linewidth=2, label="train_error")
        axes[2].plot(list(sizes), val_err, "b-", linewidth=3, label="val_error")
        axes[2].set_xlabel("訓練資料筆數")
        axes[2].set_ylabel("RMSE(log)")
        axes[2].set_title("學習曲線")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(out / f"regression_{target}.png", dpi=120)
        plt.close(fig)
    (out / "cost_linear.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"n": len(rows), **report}


# ---------- 報告 ----------

def write_report(out: Path, clf: dict | None, reg: dict | None, n_reg: int) -> Path:
    from ml import features as F

    lines = [f"# 小助理的小模型:實驗報告", "", f"產生時間:{datetime.now():%Y-%m-%d %H:%M}", ""]
    if clf:
        lines += ["## 實驗一:這題「一般 / 難題」(分類)", "",
                  f"資料 {clf['n']} 筆(難題 {clf['hard']} 筆),{clf['folds']} 折分層交叉驗證。"
                  "標籤是 LLM 分類器(Super)的判斷,所以準確率代表「跟 Super 判斷一致的比例」。", "",
                  "| 模型 | 準確率 | 精確率 | 召回率 | F1 | AUC |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['model']} | {r['accuracy']:.2f} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} "
                  f"| {r['auc']:.2f} |" for r in clf["table"]]
        lines += ["", "- 精確率:判斷為難題的,有多少真的是難題(猜錯會白白用慢的模型)",
                  "- 召回率:真的難題,有多少被抓到(漏掉會用一般模型硬做)", "",
                  "![混淆矩陣](confusion_matrix.png)", "", "![ROC 曲線](roc.png)", "",
                  "### 樹的深度對結果的影響", "", "| 深度 | ID3 準確率 | ID3 F1 | scikit-learn 準確率 | scikit-learn F1 |",
                  "|---|---|---|---|---|",
                  *[f"| {s['depth']} | {s['id3_acc']:.2f} | {s['id3_f1']:.2f} | {s['sk_acc']:.2f} | {s['sk_f1']:.2f} |"
                    for s in clf["sweep"]],
                  "", f"選用深度 {clf['depth']}(ID3 的 F1 最好;一樣好就選比較淺的)。", "", "![深度實驗](depth.png)", "",
                  "### 要注意的限制", "",
                  "- 深度是用同一組交叉驗證挑的,上面表格裡選用深度的分數會稍微偏樂觀;資料多了以後應該另外留一份測試資料。",
                  "- 範例問題是人寫的、標籤是 Super 的判斷:這裡的準確率是「跟 Super 一致的比例」,不代表真的判斷對難易。"
                  "實際效果看 report_routing.py 的「決策樹跟 LLM 分類器判斷一致」(真實問題,影子模式累積)。", "",
                  "### 正式用的 ID3 決策樹(全部資料訓練)", "", "![ID3 決策樹](tree_id3.png)", "",
                  "對照:scikit-learn 的決策樹", "", "![scikit-learn 決策樹](tree_sklearn.png)", "",
                  "### 特徵重要性(隨機森林)", ""]
        lines += [f"- {name}:{v:.3f}" for name, v in clf["importance"][:8]]
    else:
        lines += ["## 實驗一:資料還不夠", "", f"要至少 {MIN_CLASSIFY} 筆標註、每一類至少 {MIN_PER_CLASS} 筆。"
                  "先跑 `python train_models.py --label` 標註範例問題。"]
    lines.append("")
    if reg:
        lines += ["## 實驗二:這一輪花多少(回歸)", "", f"資料 {reg['n']} 輪真實紀錄,5 折交叉驗證(學 log1p,誤差還原成原單位)。"]
        for target in ("tokens", "seconds"):
            r = reg[target]
            lines += ["", f"### {'tokens' if target == 'tokens' else '時間'}", "",
                      f"| 模型 | MAE({r['unit']}) | RMSE({r['unit']}) | R²(log) |", "|---|---|---|---|"]
            lines += [f"| {m['model']} | {m['mae']:,.1f} | {m['rmse']:,.1f} | {m['r2_log']:.2f} |" for m in r["table"]]
            lines += ["", f"![{target}](regression_{target}.png)", "", "手寫線性回歸的係數(標準化後,絕對值大的影響大):", ""]
            lines += [f"- {name}:{v:+.3f}" for name, v in r["coef"][:8]]
    else:
        lines += ["## 實驗二:資料還不夠", "", f"目前 {n_reg} 輪真實紀錄,要至少 {MIN_REGRESS} 輪。"
                  "小助理每回答一題會自動記一輪(models.yaml 的 routing.difficulty.mode 要是 shadow 或 on)。"]
    lines += ["", "## 特徵(只記數字,不記問題內容)", ""]
    lines += [f"- {F.NAMES.get(n, n)}" for n in F.VECTOR]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    import yaml
    from dotenv import load_dotenv

    import tools as toolbox

    load_dotenv(HERE / ".env")
    toolbox.configure(output_dir=__import__("os").getenv("AGENT_OUTPUT") or None)
    ap = argparse.ArgumentParser(description="訓練小助理的小模型(決策樹、邏輯回歸、線性回歸)並產生實驗報告")
    ap.add_argument("--label", action="store_true", help="先請 LLM 分類器標註範例問題(會呼叫模型約 80 次)")
    ap.add_argument("--max-depth", type=int, help="決策樹最大深度(正規化);不給就用交叉驗證從 1–6 挑")
    args = ap.parse_args()

    out = toolbox.OUTPUT_DIR / "models"
    out.mkdir(parents=True, exist_ok=True)
    if args.label:
        label_seeds(out)
    hard_role = (yaml.safe_load((HERE / "models.yaml").read_text(encoding="utf-8"))["routing"].get("difficulty")
                 or {}).get("hard_role", "code_deep")
    import difficulty
    backend = difficulty.current_backend()
    rows, reg_rows = load_rows(out, toolbox.OUTPUT_DIR / "routing_log.jsonl", hard_role, backend)
    print(f"紀錄只用 {backend} 後端的(LLM_BACKEND)")
    setup_font()
    n_hard = sum(r["label"] == "hard" for r in rows)
    clf = None
    if len(rows) >= MIN_CLASSIFY and min(n_hard, len(rows) - n_hard) >= MIN_PER_CLASS:
        clf = classification_experiment(rows, out, args.max_depth)
        print(f"分類:{clf['n']} 筆(難題 {clf['hard']})")
        for r in clf["table"]:
            print(f"  {r['model']:<26} 準確率 {r['accuracy']:.2f}  F1 {r['f1']:.2f}  AUC {r['auc']:.2f}")
    else:
        print(f"分類資料不夠:{len(rows)} 筆(難題 {n_hard}),要 {MIN_CLASSIFY} 筆、每類 {MIN_PER_CLASS} 筆;"
              "先跑 --label")
    reg = regression_experiment(reg_rows, out) if len(reg_rows) >= MIN_REGRESS else None
    if reg:
        for target in ("tokens", "seconds"):
            best = reg[target]["table"][0]
            print(f"回歸 {target}:手寫線性回歸 MAE {best['mae']:,.1f} {reg[target]['unit']}")
    else:
        print(f"回歸資料不夠:{len(reg_rows)} 輪,要 {MIN_REGRESS} 輪(正常使用小助理就會累積)")
    print(f"報告:{write_report(out, clf, reg, len(reg_rows))}")


if __name__ == "__main__":
    main()
