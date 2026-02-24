import pandas as pd

from utils.preprocessing import build_user_click_history, clean_interactions, clean_products


def test_product_text_creation() -> None:
    df = pd.DataFrame(
        {
            "product_id": ["p1"],
            "title": ["Hello"],
            "description": ["World"],
            "category": ["cat"],
            "price": [10],
        }
    )
    out = clean_products(df)
    assert out.loc[0, "product_text"] == "hello world"


def test_user_history() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "product_id": ["p1", "p2", "p3"],
            "event_type": ["view", "click", "purchase"],
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )
    cleaned = clean_interactions(df)
    history = build_user_click_history(cleaned)
    assert history["u1"] == ["p2"]

