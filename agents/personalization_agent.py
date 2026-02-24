from typing import Dict, List

import numpy as np

from models.user_embedding import UserEmbeddingModel


class PersonalizationAgent:
    def __init__(self, user_embedding_model: UserEmbeddingModel, product_embedding_map: Dict[str, np.ndarray]) -> None:
        self.user_embedding_model = user_embedding_model
        self.product_embedding_map = product_embedding_map

    def rerank(self, user_id: str, ranked_items: List[dict], personalization_weight: float = 0.3) -> List[dict]:
        for item in ranked_items:
            pid = str(item["product_id"])
            p_vec = self.product_embedding_map.get(pid)
            p_score = self.user_embedding_model.score(user_id, p_vec)
            item["personalization_score"] = float(p_score)

            base = float(item.get("raw_model_score", item.get("hybrid_score", 0.0)))
            item["score"] = base + personalization_weight * p_score

            explanation = item.get("explanation", {})
            explanation["personalization_score"] = float(p_score)
            item["explanation"] = explanation

        return sorted(ranked_items, key=lambda x: x["score"], reverse=True)

