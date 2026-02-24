"""
User segmentation from interaction history.
Segments: by engagement level (event count), by preferred category, and by recency.
"""

from typing import Any, Dict

import pandas as pd


class SegmentationAgent:
    def __init__(self) -> None:
        self.user_to_segment: Dict[str, str] = {}
        self.segment_counts: Dict[str, int] = {}
        self.user_details: Dict[str, Dict[str, Any]] = {}

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> None:
        if interactions.empty:
            self.user_to_segment = {}
            self.segment_counts = {}
            self.user_details = {}
            return

        cat_map = products.set_index("product_id")["category"].to_dict() if not products.empty else {}
        interactions = interactions[interactions["event_type"].isin(["click", "cart", "purchase"])].copy()

        # Per user: event count, preferred category, last event
        agg = interactions.groupby("user_id").agg(
            event_count=("product_id", "count"),
            last_ts=("timestamp", "max"),
        ).reset_index()
        agg["user_id"] = agg["user_id"].astype(str)

        # Preferred category = most frequent category in user's interactions
        user_cats = (
            interactions.assign(
                user_id=interactions["user_id"].astype(str),
                category=interactions["product_id"].map(lambda x: cat_map.get(x, "unknown")),
            )
            .groupby("user_id")["category"]
            .apply(lambda s: s.mode().iloc[0] if len(s) else "unknown")
            .to_dict()
        )

        # Engagement terciles
        counts = agg["event_count"].values
        if len(counts) >= 3:
            t1, t2 = pd.Series(counts).quantile([1 / 3, 2 / 3]).values
        else:
            t1, t2 = 0, max(counts) if len(counts) else 1

        def engagement_segment(c: int) -> str:
            if c <= t1:
                return "low_engagement"
            if c <= t2:
                return "mid_engagement"
            return "high_engagement"

        self.user_to_segment = {}
        self.user_details = {}
        for _, row in agg.iterrows():
            uid = str(row["user_id"])
            eng = engagement_segment(int(row["event_count"]))
            pref_cat = user_cats.get(uid, "unknown")
            seg = f"{eng}_({pref_cat})"
            self.user_to_segment[uid] = seg
            self.user_details[uid] = {
                "segment": seg,
                "engagement": eng,
                "preferred_category": pref_cat,
                "event_count": int(row["event_count"]),
            }

        self.segment_counts = {}
        for s in self.user_to_segment.values():
            self.segment_counts[s] = self.segment_counts.get(s, 0) + 1

    def get_segment(self, user_id: str) -> str:
        return self.user_to_segment.get(str(user_id), "unknown")

    def get_user_details(self, user_id: str) -> Dict[str, Any]:
        return self.user_details.get(str(user_id), {})

    def list_segments(self) -> Dict[str, int]:
        return dict(self.segment_counts)
