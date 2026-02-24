from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd


class RecommendationAgent:
    """
    Basic item-to-item recommender using user co-occurrence in interaction history.
    This keeps the logic lightweight but production-realistic for a baseline module.
    """

    def __init__(self) -> None:
        self.co_counts: Dict[str, Counter] = defaultdict(Counter)
        self.user_recent_items: Dict[str, List[str]] = {}

    def fit(self, interactions: pd.DataFrame, max_recent_items: int = 20) -> None:
        data = interactions[interactions["event_type"].isin(["click", "cart", "purchase"])].copy()
        data = data.sort_values("timestamp")

        user_groups = data.groupby("user_id")["product_id"].apply(list).to_dict()
        for user_id, items in user_groups.items():
            deduped = list(dict.fromkeys([str(x) for x in items]))
            self.user_recent_items[str(user_id)] = deduped[-max_recent_items:]

            for i in range(len(deduped)):
                a = deduped[i]
                for j in range(i + 1, len(deduped)):
                    b = deduped[j]
                    if a == b:
                        continue
                    self.co_counts[a][b] += 1
                    self.co_counts[b][a] += 1

    def recommend(self, user_id: str, top_k: int = 10) -> List[dict]:
        user_items = self.user_recent_items.get(str(user_id), [])
        if not user_items:
            return []

        scores = Counter()
        user_set = set(user_items)
        for item in user_items:
            for neighbor, cnt in self.co_counts.get(item, {}).items():
                if neighbor in user_set:
                    continue
                scores[neighbor] += cnt

        if not scores:
            return []

        max_score = max(scores.values())
        ranked = scores.most_common(top_k)
        return [
            {
                "product_id": pid,
                "score": float(raw / max_score) if max_score > 0 else 0.0,
                "reason": "item_cooccurrence",
            }
            for pid, raw in ranked
        ]

