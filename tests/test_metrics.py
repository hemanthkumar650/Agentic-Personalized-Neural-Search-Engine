from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_range() -> None:
    score = ndcg_at_k([3, 2, 1], 3)
    assert 0.0 <= score <= 1.0


def test_mrr() -> None:
    assert mrr_at_k(["a", "b", "c"], {"b"}, 3) == 0.5


def test_recall() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "x"}, 3) == 0.5

