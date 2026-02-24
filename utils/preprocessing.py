import pandas as pd
from typing import Dict, List, Tuple


def load_data(
    products_path: str = "data/products.csv",
    interactions_path: str = "data/interactions.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    products = pd.read_csv(products_path)
    interactions = pd.read_csv(interactions_path)

    required_products = {"product_id", "title", "description", "category", "price"}
    required_interactions = {"user_id", "product_id", "event_type", "timestamp"}

    if not required_products.issubset(products.columns):
        missing = sorted(required_products - set(products.columns))
        raise ValueError(f"products.csv missing columns: {missing}")

    if not required_interactions.issubset(interactions.columns):
        missing = sorted(required_interactions - set(interactions.columns))
        raise ValueError(f"interactions.csv missing columns: {missing}")

    return clean_products(products), clean_interactions(interactions)


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["category"] = df["category"].fillna("unknown").astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(df["price"].median())
    df = df.dropna(subset=["product_id"]).drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    df["product_text"] = (df["title"] + " " + df["description"]).str.strip().str.lower()
    return df


def clean_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    df = interactions.copy()
    df = df.dropna(subset=["user_id", "product_id", "event_type", "timestamp"]).copy()
    df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
    df = df[df["event_type"].isin(["view", "click", "cart", "purchase"])]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def build_user_click_history(interactions: pd.DataFrame) -> Dict[str, List[str]]:
    clicks = interactions[interactions["event_type"].isin(["click", "cart", "purchase"])].copy()
    return clicks.groupby("user_id")["product_id"].apply(list).to_dict()
