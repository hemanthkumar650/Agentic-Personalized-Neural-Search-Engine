import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.personalization_agent import PersonalizationAgent
from agents.ranking_agent import RankingAgent
from agents.retrieval_agent import RetrievalAgent
from models.user_embedding import UserEmbeddingModel
from utils.feature_engineering import build_feature_context
from utils.preprocessing import load_data


def run_eda_error_analysis(top_n_errors: int = 20) -> None:
    products, interactions = load_data()
    Path("runs/analysis").mkdir(parents=True, exist_ok=True)

    # EDA summaries
    eda = {
        "num_products": int(len(products)),
        "num_interactions": int(len(interactions)),
        "num_users": int(interactions["user_id"].nunique()),
        "category_distribution": products["category"].value_counts().head(15).to_dict(),
        "event_distribution": interactions["event_type"].value_counts().to_dict(),
        "price_summary": products["price"].describe().to_dict(),
    }
    Path("runs/analysis/eda_summary.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")

    retrieval = RetrievalAgent()
    retrieval.load("models/retrieval.pkl")
    ranker = RankingAgent("models/ranker.pkl")
    ranker.load()
    user_model = UserEmbeddingModel.load("models/user_embeddings.pkl")
    personalization = PersonalizationAgent(user_model, retrieval.embedding_by_product())
    feature_context = build_feature_context(products, interactions)
    product_map = {str(r["product_id"]): r for _, r in products.iterrows()}

    eval_rows = interactions[interactions["event_type"].isin(["click", "cart", "purchase"])].copy()
    failures = []
    for _, row in eval_rows.iterrows():
        user_id = str(row["user_id"])
        target_pid = str(row["product_id"])
        query = str(product_map[target_pid]["title"]).lower()

        cands = [r.__dict__.copy() for r in retrieval.retrieve(query, top_k=10, alpha=0.5)]
        ranked = ranker.rerank(query, user_id, cands, products, feature_context)
        final = personalization.rerank(user_id, ranked, personalization_weight=0.3)
        ranked_ids = [x["product_id"] for x in final]

        if target_pid in ranked_ids:
            rank_pos = ranked_ids.index(target_pid) + 1
            miss = 0
        else:
            rank_pos = 999
            miss = 1

        failures.append(
            {
                "user_id": user_id,
                "target_product_id": target_pid,
                "query": query,
                "target_rank": rank_pos,
                "missed_top10": miss,
            }
        )

    fail_df = pd.DataFrame(failures).sort_values(["missed_top10", "target_rank"], ascending=[False, False])
    fail_df.head(top_n_errors).to_csv("runs/analysis/top_errors.csv", index=False)
    summary = {
        "total_eval_samples": int(len(fail_df)),
        "top10_miss_rate": float(fail_df["missed_top10"].mean()) if len(fail_df) else 0.0,
        "mean_target_rank": float(fail_df["target_rank"].replace(999, 11).mean()) if len(fail_df) else 0.0,
    }
    Path("runs/analysis/error_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved runs/analysis/eda_summary.json")
    print("Saved runs/analysis/top_errors.csv")
    print("Saved runs/analysis/error_summary.json")


if __name__ == "__main__":
    run_eda_error_analysis()
