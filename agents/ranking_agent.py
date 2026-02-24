from typing import Dict, List

import pandas as pd

from models.ranker import LambdaMARTRanker
from utils.feature_engineering import FeatureContext, build_feature_row, to_feature_matrix


class RankingAgent:
    def __init__(self, model_path: str = "models/ranker.pkl") -> None:
        self.ranker = LambdaMARTRanker()
        self.model_path = model_path

    def load(self) -> None:
        self.ranker.load(self.model_path)

    def rerank(
        self,
        query: str,
        user_id: str,
        candidates: List[dict],
        products: pd.DataFrame,
        feature_context: FeatureContext,
    ) -> List[dict]:
        product_map: Dict[str, pd.Series] = {str(r["product_id"]): r for _, r in products.iterrows()}

        feature_rows = []
        for c in candidates:
            pid = str(c["product_id"])
            feat = build_feature_row(
                query=query,
                user_id=user_id,
                product_row=product_map[pid],
                bm25_score=float(c["bm25_score"]),
                cosine_similarity=float(c["cosine_similarity"]),
                hybrid_score=float(c["hybrid_score"]),
                context=feature_context,
            )
            feature_rows.append(feat)

        X = to_feature_matrix(feature_rows)
        scores = self.ranker.predict(X)

        for i, c in enumerate(candidates):
            c["raw_model_score"] = float(scores[i])
            c["explanation"] = feature_rows[i]

        return sorted(candidates, key=lambda x: x["raw_model_score"], reverse=True)

