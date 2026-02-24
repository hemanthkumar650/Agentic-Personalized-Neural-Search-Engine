import json
from datetime import datetime
from pathlib import Path

import numpy as np

from agents.retrieval_agent import RetrievalAgent
from models.ranker import LambdaMARTRanker
from utils.experiment_tracking import log_experiment
from utils.feature_engineering import build_feature_context, build_feature_row, to_feature_matrix
from utils.preprocessing import load_data

EVENT_REL = {"view": 0.0, "click": 1.0, "cart": 2.0, "purchase": 3.0}


def train() -> None:
    products, interactions = load_data()
    retrieval = RetrievalAgent()
    retrieval.load("models/retrieval.pkl")
    context = build_feature_context(products, interactions)
    product_map = {str(r["product_id"]): r for _, r in products.iterrows()}

    X_rows = []
    y = []
    group = []

    supervised = interactions[interactions["event_type"].isin(["click", "cart", "purchase"])]
    for _, row in supervised.iterrows():
        user_id = str(row["user_id"])
        target_pid = str(row["product_id"])
        query = str(product_map[target_pid]["title"]).lower()
        candidates = retrieval.retrieve(query, top_k=10, alpha=0.5)
        count = 0
        for c in candidates:
            prod = product_map[str(c.product_id)]
            X_rows.append(
                build_feature_row(
                    query=query,
                    user_id=user_id,
                    product_row=prod,
                    bm25_score=c.bm25_score,
                    cosine_similarity=c.cosine_similarity,
                    hybrid_score=c.hybrid_score,
                    context=context,
                )
            )
            y.append(EVENT_REL[row["event_type"]] if c.product_id == target_pid else 0.0)
            count += 1
        if count > 0:
            group.append(count)

    X = to_feature_matrix(X_rows)
    y = np.asarray(y, dtype=np.float32)

    ranker = LambdaMARTRanker()
    ranker.fit(X, y, group)
    Path("models").mkdir(exist_ok=True)
    ranker.save("models/ranker.pkl")

    Path("runs").mkdir(exist_ok=True)
    artifact = Path("runs") / f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_train_ranker.json"
    train_meta = {"samples": int(X.shape[0]), "queries": len(group)}
    artifact.write_text(json.dumps(train_meta, indent=2), encoding="utf-8")
    log_experiment(
        run_type="train_ranker",
        metrics={},
        metadata={"train_samples": str(X.shape[0]), "train_queries": str(len(group))},
    )
    print("Saved models/ranker.pkl")
    print(f"Saved run artifact: {artifact}")


if __name__ == "__main__":
    train()
