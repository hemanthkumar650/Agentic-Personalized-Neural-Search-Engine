import pickle
from typing import Dict

import numpy as np
import pandas as pd

EVENT_WEIGHT = {"view": 1.0, "click": 2.0, "cart": 3.0, "purchase": 4.0}


class UserEmbeddingModel:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.user_embeddings: Dict[str, np.ndarray] = {}

    def fit(self, interactions: pd.DataFrame, product_embeddings: Dict[str, np.ndarray]) -> None:
        for user_id, grp in interactions.groupby("user_id"):
            vec = np.zeros(self.dim, dtype=np.float32)
            total = 0.0
            for _, row in grp.iterrows():
                pid = str(row["product_id"])
                emb = product_embeddings.get(pid)
                if emb is None:
                    continue
                w = EVENT_WEIGHT.get(str(row["event_type"]).lower(), 0.0)
                vec += w * emb
                total += w
            if total > 0:
                vec = vec / total
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.user_embeddings[str(user_id)] = vec

    def score(self, user_id: str, product_embedding: np.ndarray | None) -> float:
        if product_embedding is None:
            return 0.0
        user = self.user_embeddings.get(str(user_id))
        if user is None:
            return 0.0
        return float(np.dot(user, product_embedding))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"dim": self.dim, "embeddings": self.user_embeddings}, f)

    @classmethod
    def load(cls, path: str) -> "UserEmbeddingModel":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        model = cls(payload["dim"])
        model.user_embeddings = payload["embeddings"]
        return model

