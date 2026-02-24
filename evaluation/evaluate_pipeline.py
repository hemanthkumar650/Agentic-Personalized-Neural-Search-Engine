import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.personalization_agent import PersonalizationAgent
from agents.ranking_agent import RankingAgent
from agents.retrieval_agent import RetrievalAgent
from evaluation.metrics import mean, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from models.user_embedding import UserEmbeddingModel
from utils.experiment_tracking import log_experiment
from utils.feature_engineering import build_feature_context
from utils.preprocessing import load_data


def evaluate() -> dict:
    products, interactions = load_data()
    retrieval = RetrievalAgent()
    retrieval.load("models/retrieval.pkl")

    ranking = RankingAgent("models/ranker.pkl")
    ranking.load()

    user_model = UserEmbeddingModel.load("models/user_embeddings.pkl")
    personalization = PersonalizationAgent(user_model, retrieval.embedding_by_product())
    feature_context = build_feature_context(products, interactions)
    product_map = {str(r["product_id"]): r for _, r in products.iterrows()}

    ndcg_vals, mrr_vals, recall_vals, precision_vals = [], [], [], []
    eval_rows = interactions[interactions["event_type"].isin(["click", "cart", "purchase"])]

    for _, row in eval_rows.iterrows():
        user_id = str(row["user_id"])
        target_pid = str(row["product_id"])
        query = str(product_map[target_pid]["title"]).lower()

        retrieved = retrieval.retrieve(query, top_k=10, alpha=0.5)
        candidates = [r.__dict__.copy() for r in retrieved]
        ranked = ranking.rerank(query, user_id, candidates, products, feature_context)
        final = personalization.rerank(user_id, ranked, personalization_weight=0.3)

        ranked_ids = [x["product_id"] for x in final]
        relevant = {target_pid}
        rel = [1.0 if pid == target_pid else 0.0 for pid in ranked_ids]

        ndcg_vals.append(ndcg_at_k(rel, 10))
        mrr_vals.append(mrr_at_k(ranked_ids, relevant, 10))
        recall_vals.append(recall_at_k(ranked_ids, relevant, 10))
        precision_vals.append(precision_at_k(ranked_ids, relevant, 10))

    result = {
        "NDCG@10": mean(ndcg_vals),
        "MRR": mean(mrr_vals),
        "Recall@10": mean(recall_vals),
        "Precision@10": mean(precision_vals),
    }

    Path("runs").mkdir(exist_ok=True)
    out_file = Path("runs") / f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_evaluate_pipeline.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log_experiment(run_type="evaluate_pipeline", metrics=result)
    print(json.dumps(result, indent=2))
    print(f"[Eval] Saved run artifact: {out_file}")
    return result


if __name__ == "__main__":
    evaluate()
