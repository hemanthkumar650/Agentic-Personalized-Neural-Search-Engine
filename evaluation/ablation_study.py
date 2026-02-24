import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

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


def _compute_metrics(ranked_ids: List[str], relevant_id: str) -> Dict[str, float]:
    relevant = {relevant_id}
    rel = [1.0 if pid == relevant_id else 0.0 for pid in ranked_ids]
    return {
        "NDCG@10": ndcg_at_k(rel, 10),
        "MRR": mrr_at_k(ranked_ids, relevant, 10),
        "Recall@10": recall_at_k(ranked_ids, relevant, 10),
        "Precision@10": precision_at_k(ranked_ids, relevant, 10),
    }


def run_ablation() -> Dict[str, Dict[str, float]]:
    products, interactions = load_data()
    retrieval = RetrievalAgent()
    retrieval.load("models/retrieval.pkl")

    ranking = RankingAgent("models/ranker.pkl")
    ranking.load()

    user_model = UserEmbeddingModel.load("models/user_embeddings.pkl")
    personalization = PersonalizationAgent(user_model, retrieval.embedding_by_product())
    feature_context = build_feature_context(products, interactions)
    product_map = {str(r["product_id"]): r for _, r in products.iterrows()}

    strategies: Dict[str, Callable[[str, str], List[str]]] = {}

    def bm25_only(query: str, user_id: str) -> List[str]:
        cands = [r.__dict__.copy() for r in retrieval.retrieve(query, top_k=10, alpha=1.0)]
        return [x["product_id"] for x in sorted(cands, key=lambda x: x["bm25_score"], reverse=True)]

    def dense_only(query: str, user_id: str) -> List[str]:
        cands = [r.__dict__.copy() for r in retrieval.retrieve(query, top_k=10, alpha=0.0)]
        return [x["product_id"] for x in sorted(cands, key=lambda x: x["cosine_similarity"], reverse=True)]

    def hybrid(query: str, user_id: str) -> List[str]:
        cands = [r.__dict__.copy() for r in retrieval.retrieve(query, top_k=10, alpha=0.5)]
        return [x["product_id"] for x in sorted(cands, key=lambda x: x["hybrid_score"], reverse=True)]

    def ranker(query: str, user_id: str) -> List[str]:
        cands = [r.__dict__.copy() for r in retrieval.retrieve(query, top_k=10, alpha=0.5)]
        ranked = ranking.rerank(query, user_id, cands, products, feature_context)
        return [x["product_id"] for x in ranked]

    def personalized(query: str, user_id: str) -> List[str]:
        cands = [r.__dict__.copy() for r in retrieval.retrieve(query, top_k=10, alpha=0.5)]
        ranked = ranking.rerank(query, user_id, cands, products, feature_context)
        final = personalization.rerank(user_id, ranked, personalization_weight=0.3)
        return [x["product_id"] for x in final]

    strategies["BM25"] = bm25_only
    strategies["Dense"] = dense_only
    strategies["Hybrid"] = hybrid
    strategies["Ranker"] = ranker
    strategies["Personalized"] = personalized

    eval_rows = interactions[interactions["event_type"].isin(["click", "cart", "purchase"])]
    aggregated: Dict[str, Dict[str, List[float]]] = {
        k: {"NDCG@10": [], "MRR": [], "Recall@10": [], "Precision@10": []} for k in strategies
    }

    for _, row in eval_rows.iterrows():
        user_id = str(row["user_id"])
        target_pid = str(row["product_id"])
        query = str(product_map[target_pid]["title"]).lower()
        for name, fn in strategies.items():
            ranked_ids = fn(query, user_id)
            metrics = _compute_metrics(ranked_ids, target_pid)
            for mk, mv in metrics.items():
                aggregated[name][mk].append(mv)

    summary: Dict[str, Dict[str, float]] = {}
    for name, metric_lists in aggregated.items():
        summary[name] = {metric_name: mean(values) for metric_name, values in metric_lists.items()}
        log_experiment(run_type=f"ablation_{name.lower()}", metrics=summary[name], metadata={"strategy": name})

    Path("runs").mkdir(exist_ok=True)
    out = Path("runs") / f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_ablation.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[Ablation] Saved artifact: {out}")
    return summary


if __name__ == "__main__":
    run_ablation()
