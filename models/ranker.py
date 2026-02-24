import pickle
from typing import Iterable

import lightgbm as lgb
import numpy as np


class LambdaMARTRanker:
    def __init__(self) -> None:
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, group: Iterable[int]) -> None:
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
        )
        self.model.fit(X, y, group=list(group))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Ranker is not fitted")
        return self.model.predict(X)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)

