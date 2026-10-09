"""線性回歸、邏輯回歸:照課程「線性回歸_代碼」「邏輯回歸_代碼」的 class 寫(先標準化、加一欄 1、train 迭代更新 theta)。

跟課程版不一樣的地方:
1. **predict 用訓練時的平均和標準差標準化**。課程版的 predict 會對「要預測的那批資料」重新算平均和標準差;
   一次只預測一筆時(小助理就是一題一題預測),標準差是 0、平均也沒減掉,算出來的值是錯的。
2. 邏輯回歸只做二元(一般 / 難題),cost 加了 L2 正規化(課程線性回歸實驗講的 Ridge):
   資料少、又可以完全分開時,權重會一直變大,機率全變成 0 或 1。
3. 沒有多項式 / sinusoid 特徵轉換(特徵本來就是 0/1 和次數,用不到)。
4. to_json() 存成 {"names", "mean", "std", "theta"}:小助理執行時用 runtime.py 純 Python 算,不用 numpy。

訓練才需要 numpy / scipy(scikit-learn 會一起裝)。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def normalize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """課程版 utils/features/normalize.py:減平均、除以標準差(標準差是 0 的欄位除以 1)。"""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1
    return (features - mean) / std, mean, std


def _with_bias(data: np.ndarray) -> np.ndarray:
    return np.hstack((np.ones((data.shape[0], 1)), data))


class LinearRegression:
    def __init__(self, data, labels, names: list[str] | None = None):
        """1. 標準化(記下平均、標準差) 2. 加一欄 1 3. theta 從 0 開始。"""
        data = np.asarray(data, dtype=float)
        normalized, self.mean, self.std = normalize(data)
        self.data = _with_bias(normalized)
        self.labels = np.asarray(labels, dtype=float).reshape(-1, 1)
        self.names = names or [f"x{i}" for i in range(data.shape[1])]
        self.theta = np.zeros((self.data.shape[1], 1))

    def train(self, alpha: float = 0.05, num_iterations: int = 2000) -> tuple[np.ndarray, list[float]]:
        """批量梯度下降,回傳 (theta, 每次迭代的 cost)。"""
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return self.theta, cost_history

    def gradient_step(self, alpha: float) -> None:
        m = self.data.shape[0]
        delta = self.hypothesis(self.data, self.theta) - self.labels
        self.theta = self.theta - alpha * (1 / m) * (delta.T @ self.data).T

    def cost_function(self, data, labels) -> float:
        """均方誤差的一半(課程版的寫法)。課程版這裡用的是 self.data,換別的資料算 cost 也只會算到訓練資料。"""
        delta = self.hypothesis(data, self.theta) - labels
        return float((delta.T @ delta)[0, 0] / 2 / data.shape[0])

    @staticmethod
    def hypothesis(data, theta):
        return data @ theta

    def _prepare(self, data) -> np.ndarray:
        return _with_bias((np.asarray(data, dtype=float) - self.mean) / self.std)  # 用訓練時的平均和標準差

    def get_cost(self, data, labels) -> float:
        return self.cost_function(self._prepare(data), np.asarray(labels, dtype=float).reshape(-1, 1))

    def predict(self, data) -> np.ndarray:
        return self.hypothesis(self._prepare(data), self.theta)

    def to_json(self) -> dict:
        return {"names": self.names, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "theta": self.theta.ravel().tolist()}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, data, labels, names: list[str] | None = None, l2: float = 1.0,
                 sample_weight=None):
        """labels 是 0 / 1(1 = 難題)。sample_weight:類別權重(難題少,權重大一點)。"""
        data = np.asarray(data, dtype=float)
        normalized, self.mean, self.std = normalize(data)
        self.data = _with_bias(normalized)
        self.labels = np.asarray(labels, dtype=float).reshape(-1, 1)
        self.weights = (np.ones(len(self.labels)) if sample_weight is None
                        else np.asarray(sample_weight, dtype=float)).reshape(-1, 1)
        self.names = names or [f"x{i}" for i in range(data.shape[1])]
        self.l2 = l2
        self.theta = np.zeros((self.data.shape[1], 1))

    def train(self, max_iterations: int = 1000) -> tuple[np.ndarray, list[float]]:
        """跟課程版一樣用 scipy 的 L-BFGS-B 找讓 cost 最小的 theta。"""
        cost_history: list[float] = []
        n = self.data.shape[1]
        result = minimize(lambda t: self.cost_function(t.reshape(n, 1)), self.theta.ravel(), method="L-BFGS-B",
                          jac=lambda t: self.gradient_step(t.reshape(n, 1)),
                          callback=lambda t: cost_history.append(self.cost_function(t.reshape(n, 1))),
                          options={"maxiter": max_iterations})
        if not result.success:
            raise ArithmeticError(f"Can not minimize cost function: {result.message}")
        self.theta = result.x.reshape(n, 1)
        return self.theta, cost_history

    def cost_function(self, theta) -> float:
        """加權的交叉熵 + L2(不罰截距)。"""
        p = np.clip(self.hypothesis(self.data, theta), 1e-15, 1 - 1e-15)
        y, w = self.labels, self.weights
        cross = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum()
        return float(cross + self.l2 / 2 * (theta[1:] ** 2).sum() / w.sum())

    def gradient_step(self, theta) -> np.ndarray:
        p = self.hypothesis(self.data, theta)
        grad = self.data.T @ (self.weights * (p - self.labels)) / self.weights.sum()
        grad[1:] += self.l2 * theta[1:] / self.weights.sum()
        return grad.ravel()

    @staticmethod
    def hypothesis(data, theta):
        return sigmoid(data @ theta)

    def predict_proba(self, data) -> np.ndarray:
        """是難題的機率。"""
        prepared = _with_bias((np.asarray(data, dtype=float) - self.mean) / self.std)
        return self.hypothesis(prepared, self.theta).ravel()

    def predict(self, data, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(data) >= threshold).astype(int)

    def to_json(self) -> dict:
        return {"names": self.names, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "theta": self.theta.ravel().tolist()}
