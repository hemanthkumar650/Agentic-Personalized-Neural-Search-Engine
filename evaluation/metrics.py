from typing import Iterable, List, Set

import numpy as np


def dcg_at_k(relevances: List[float], k: int) -> float:
    rel = np.asarray(relevances[:k], dtype=np.float32)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum((2**rel - 1) * discounts))


def ndcg_at_k(relevances: List[float], k: int) -> float:
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal


def mrr_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    for i, pid in enumerate(ranked_ids[:k], start=1):
        if pid in relevant_ids:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]).intersection(relevant_ids)) / len(relevant_ids)


def precision_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked_ids[:k]).intersection(relevant_ids)) / k


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))

