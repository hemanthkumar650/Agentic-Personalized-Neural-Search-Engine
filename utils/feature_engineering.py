from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

EVENT_WEIGHT = {"view": 1.0, "click": 2.0, "cart": 3.0, "purchase": 4.0}


@dataclass
class FeatureContext:
    popularity: Dict[str, float]
    user_category_pref: Dict[str, Dict[str, float]]


def build_feature_context(products: pd.DataFrame, interactions: pd.DataFrame) -> FeatureContext:
    popularity_raw = interactions["product_id"].value_counts().to_dict()
    max_pop = max(popularity_raw.values()) if popularity_raw else 1.0
    popularity = {pid: val / max_pop for pid, val in popularity_raw.items()}

    merged = interactions.merge(products[["product_id", "category"]], on="product_id", how="left")
    merged["w"] = merged["event_type"].map(EVENT_WEIGHT).fillna(0.0)

    user_category_pref = {}
    for user_id, grp in merged.groupby("user_id"):
        cat_scores = grp.groupby("category")["w"].sum().to_dict()
        total = sum(cat_scores.values()) or 1.0
        user_category_pref[user_id] = {k: v / total for k, v in cat_scores.items()}

    return FeatureContext(popularity=popularity, user_category_pref=user_category_pref)


def price_match_indicator(query: str, price: float) -> float:
    q = query.lower()
    if "cheap" in q or "budget" in q:
        return float(price < 100)
    if "premium" in q or "expensive" in q:
        return float(price >= 300)
    return 0.0


def build_feature_row(
    query: str,
    user_id: str,
    product_row: pd.Series,
    bm25_score: float,
    cosine_similarity: float,
    hybrid_score: float,
    context: FeatureContext,
) -> Dict[str, float]:
    pid = str(product_row["product_id"])
    category = str(product_row["category"])
    return {
        "bm25_score": float(bm25_score),
        "cosine_similarity": float(cosine_similarity),
        "hybrid_score": float(hybrid_score),
        "product_popularity": float(context.popularity.get(pid, 0.0)),
        "user_category_preference": float(context.user_category_pref.get(user_id, {}).get(category, 0.0)),
        "query_length": float(len(query.split())),
        "price_match_indicator": float(price_match_indicator(query, float(product_row["price"]))),
    }


def to_feature_matrix(rows: list[dict]) -> np.ndarray:
    cols = [
        "bm25_score",
        "cosine_similarity",
        "hybrid_score",
        "product_popularity",
        "user_category_preference",
        "query_length",
        "price_match_indicator",
    ]
    return np.asarray([[r[c] for c in cols] for r in rows], dtype=np.float32)

