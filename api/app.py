import os
import time
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.personalization_agent import PersonalizationAgent
from agents.recommendation_agent import RecommendationAgent
from agents.ranking_agent import RankingAgent
from agents.retrieval_agent import RetrievalAgent
from agents.segmentation_agent import SegmentationAgent
from models.user_embedding import UserEmbeddingModel
from utils.feature_engineering import build_feature_context
from utils.preprocessing import load_data

app = FastAPI(title="Agentic Personalized Neural Search Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    ranker_available: bool
    personalization_available: bool


class SearchResponse(BaseModel):
    query: str
    user_id: str
    strategy: str
    search_id: str
    personalized: bool
    num_results: int
    latency_ms: float
    results: List[Dict[str, Any]]


class RecommendationResponse(BaseModel):
    user_id: str
    num_results: int
    results: List[Dict[str, Any]]


class SegmentResponse(BaseModel):
    user_id: str
    segment: str
    engagement: str
    preferred_category: str
    event_count: int


class SegmentsListResponse(BaseModel):
    segments: Dict[str, int]


class ContentSimilarResponse(BaseModel):
    product_id: str
    similar: List[Dict[str, Any]]


class ConversationRequest(BaseModel):
    user_id: str = ""
    message: str = ""


class ConversationResponse(BaseModel):
    intent: str
    query: str | None = None
    results: List[Dict[str, Any]]
    num_results: int = 0


class EventLogRequest(BaseModel):
    event_type: str
    search_id: str = ""
    user_id: str = ""
    query: str = ""
    product_id: str
    position: int = 0
    metadata: Dict[str, Any] = {}


products_df: pd.DataFrame | None = None
feature_context = None
retrieval: RetrievalAgent | None = None
ranking: RankingAgent | None = None
personalization: PersonalizationAgent | None = None
recommendation: RecommendationAgent | None = None
segmentation: SegmentationAgent | None = None
log_dir = Path("runs/logs")
events_file = log_dir / "events.jsonl"
impressions_file = log_dir / "impressions.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_jsonl(path: Path, max_rows: int = 2000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines[-max_rows:]:
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@app.on_event("startup")
def startup() -> None:
    global products_df, feature_context, retrieval, ranking, personalization, recommendation, segmentation

    products_df, interactions_df = load_data()
    feature_context = build_feature_context(products_df, interactions_df)

    retrieval = RetrievalAgent()
    retrieval.load("models/retrieval.pkl")

    ranking = RankingAgent("models/ranker.pkl")
    ranking.load()

    user_model = UserEmbeddingModel.load("models/user_embeddings.pkl")
    personalization = PersonalizationAgent(user_model, retrieval.embedding_by_product())

    recommendation = RecommendationAgent()
    recommendation.fit(interactions_df)

    segmentation = SegmentationAgent()
    segmentation.fit(interactions_df, products_df)


@app.get("/")
def root() -> dict:
    return {
        "name": "Agentic Personalized Neural Search Engine",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "search": "/search?q=wireless+headphones&user_id=user_1&top_k=5",
        "recommend": "/recommend?user_id=user_1&top_k=5",
        "user_segment": "/user/{user_id}/segment",
        "segments": "/segments",
        "content_similar": "/content/similar?product_id=8&top_k=5",
        "conversation": "POST /conversation",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        ranker_available=ranking is not None,
        personalization_available=personalization is not None,
    )


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    user_id: str = Query(""),
    top_k: int = Query(10, ge=1, le=50),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
    personalization_weight: float = Query(0.3, ge=0.0, le=2.0),
    strategy: str = Query("auto"),
) -> SearchResponse:
    if retrieval is None or ranking is None or personalization is None or products_df is None:
        raise RuntimeError("Artifacts not loaded")

    t0 = time.perf_counter()
    mode = os.getenv("SEARCH_RANKING_MODE", "ranker").strip().lower()
    strategy = strategy.strip().lower()
    if strategy == "auto":
        strategy = "personalized" if user_id else mode
    valid = {"bm25", "dense", "hybrid", "ranker", "personalized"}
    if strategy not in valid:
        strategy = "ranker"
    retrieved = retrieval.retrieve(q, top_k=max(top_k, 10), alpha=alpha)
    candidates = [r.__dict__.copy() for r in retrieved]

    if strategy == "bm25":
        ranked = sorted(candidates, key=lambda x: x["bm25_score"], reverse=True)
        for item in ranked:
            item["raw_model_score"] = item["bm25_score"]
            item["score"] = item["bm25_score"]
            item["explanation"] = {"bm25_score": item["bm25_score"]}
    elif strategy == "dense":
        ranked = sorted(candidates, key=lambda x: x["cosine_similarity"], reverse=True)
        for item in ranked:
            item["raw_model_score"] = item["cosine_similarity"]
            item["score"] = item["cosine_similarity"]
            item["explanation"] = {"cosine_similarity": item["cosine_similarity"]}
    elif strategy == "hybrid":
        ranked = sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)
        for item in ranked:
            item["raw_model_score"] = item["hybrid_score"]
            item["score"] = item["hybrid_score"]
            item["explanation"] = {
                "bm25_score": item["bm25_score"],
                "cosine_similarity": item["cosine_similarity"],
                "hybrid_score": item["hybrid_score"],
            }
    else:
        ranked = ranking.rerank(q, user_id, candidates, products_df, feature_context)

    personalized = bool(user_id) and strategy == "personalized"
    if strategy == "personalized":
        ranked = personalization.rerank(user_id, ranked, personalization_weight)
    else:
        for item in ranked:
            item["score"] = float(item.get("raw_model_score", item.get("hybrid_score", 0.0)))

    product_map = {str(r["product_id"]): r for _, r in products_df.iterrows()}
    results = []
    for i, item in enumerate(ranked[:top_k], start=1):
        p = product_map[str(item["product_id"])]
        results.append(
            {
                "rank": i,
                "product_id": str(p["product_id"]),
                "title": str(p["title"]),
                "category": str(p["category"]),
                "price": float(p["price"]),
                "score": float(item["score"]),
                "explanation": item.get("explanation", {}),
            }
        )

    latency = round((time.perf_counter() - t0) * 1000.0, 2)
    search_id = str(uuid.uuid4())
    _append_jsonl(
        impressions_file,
        {
            "timestamp": _now_iso(),
            "search_id": search_id,
            "query": q,
            "user_id": user_id,
            "strategy": strategy,
            "latency_ms": latency,
            "top_k": top_k,
            "results": [
                {
                    "rank": r["rank"],
                    "product_id": r["product_id"],
                    "category": r["category"],
                    "score": r["score"],
                }
                for r in results
            ],
        },
    )
    return SearchResponse(
        query=q,
        user_id=user_id,
        strategy=strategy,
        search_id=search_id,
        personalized=personalized,
        num_results=len(results),
        latency_ms=latency,
        results=results,
    )


@app.get("/recommend", response_model=RecommendationResponse)
def recommend(user_id: str = Query(..., min_length=1), top_k: int = Query(10, ge=1, le=50)) -> RecommendationResponse:
    if recommendation is None or products_df is None:
        raise RuntimeError("Artifacts not loaded")

    recs = recommendation.recommend(user_id=user_id, top_k=top_k)
    product_map = {str(r["product_id"]): r for _, r in products_df.iterrows()}
    enriched = []
    for i, rec in enumerate(recs, start=1):
        p = product_map.get(str(rec["product_id"]))
        if p is None:
            continue
        enriched.append(
            {
                "rank": i,
                "product_id": str(p["product_id"]),
                "title": str(p["title"]),
                "category": str(p["category"]),
                "price": float(p["price"]),
                "score": float(rec["score"]),
                "reason": rec["reason"],
            }
        )

    return RecommendationResponse(user_id=user_id, num_results=len(enriched), results=enriched)


@app.get("/user/{user_id}/segment", response_model=SegmentResponse)
def get_user_segment(user_id: str) -> SegmentResponse:
    if segmentation is None:
        raise RuntimeError("Segmentation not loaded")
    details = segmentation.get_user_details(user_id)
    if not details:
        return SegmentResponse(
            user_id=user_id,
            segment="unknown",
            engagement="unknown",
            preferred_category="unknown",
            event_count=0,
        )
    return SegmentResponse(
        user_id=user_id,
        segment=details.get("segment", "unknown"),
        engagement=details.get("engagement", "unknown"),
        preferred_category=details.get("preferred_category", "unknown"),
        event_count=details.get("event_count", 0),
    )


@app.get("/segments", response_model=SegmentsListResponse)
def list_segments() -> SegmentsListResponse:
    if segmentation is None:
        return SegmentsListResponse(segments={})
    return SegmentsListResponse(segments=segmentation.list_segments())


@app.get("/content/similar", response_model=ContentSimilarResponse)
def content_similar(
    product_id: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
) -> ContentSimilarResponse:
    if retrieval is None or products_df is None:
        raise RuntimeError("Artifacts not loaded")
    similar_ids = retrieval.similar_by_content(product_id, top_k=top_k)
    product_map = {str(r["product_id"]): r for _, r in products_df.iterrows()}
    similar = []
    for pid, score in similar_ids:
        p = product_map.get(str(pid))
        if p is None:
            continue
        similar.append(
            {
                "product_id": str(p["product_id"]),
                "title": str(p["title"]),
                "category": str(p["category"]),
                "price": float(p["price"]),
                "content_similarity": round(score, 4),
            }
        )
    return ContentSimilarResponse(product_id=product_id, similar=similar)


def _conversation_intent(message: str) -> str:
    m = message.strip().lower()
    if not m:
        return "search"
    recommend_keywords = ("recommend", "suggest", "for me", "what should i", "what can i buy", "show me something")
    if any(k in m for k in recommend_keywords):
        return "recommend"
    return "search"


@app.post("/conversation", response_model=ConversationResponse)
def conversation(payload: ConversationRequest = Body(...)) -> ConversationResponse:
    if retrieval is None or ranking is None or personalization is None or recommendation is None or products_df is None:
        raise RuntimeError("Artifacts not loaded")
    user_id = (payload.user_id or "").strip() or "user_1"
    message = (payload.message or "").strip()
    intent = _conversation_intent(message)
    query = message if intent == "search" else None
    if intent == "recommend":
        recs = recommendation.recommend(user_id=user_id, top_k=10)
        product_map = {str(r["product_id"]): r for _, r in products_df.iterrows()}
        results = []
        for i, rec in enumerate(recs, start=1):
            p = product_map.get(str(rec["product_id"]))
            if p is None:
                continue
            results.append(
                {
                    "rank": i,
                    "product_id": str(p["product_id"]),
                    "title": str(p["title"]),
                    "category": str(p["category"]),
                    "price": float(p["price"]),
                    "score": float(rec["score"]),
                }
            )
        return ConversationResponse(intent="recommend", results=results, num_results=len(results))
    if not message:
        return ConversationResponse(intent="search", query=None, results=[], num_results=0)
    t0 = time.perf_counter()
    strategy = "personalized" if user_id else "ranker"
    candidates = retrieval.retrieve(message, top_k=20, alpha=0.5)
    candidates = [{"product_id": c.product_id, "bm25_score": c.bm25_score, "cosine_similarity": c.cosine_similarity, "hybrid_score": c.hybrid_score} for c in candidates]
    ranked = ranking.rerank(message, user_id, candidates, products_df, feature_context)
    ranked = personalization.rerank(user_id, ranked, personalization_weight=0.3)
    product_map = {str(r["product_id"]): r for _, r in products_df.iterrows()}
    results = []
    for i, item in enumerate(ranked[:10], start=1):
        p = product_map.get(str(item["product_id"]))
        if p is None:
            continue
        results.append(
            {
                "rank": i,
                "product_id": str(p["product_id"]),
                "title": str(p["title"]),
                "category": str(p["category"]),
                "price": float(p["price"]),
                "score": float(item.get("score", 0)),
            }
        )
    return ConversationResponse(intent="search", query=message, results=results, num_results=len(results))


@app.post("/events")
def log_event(payload: EventLogRequest = Body(...)) -> Dict[str, Any]:
    event_type = payload.event_type.strip().lower()
    if event_type not in {"view", "click", "cart", "purchase"}:
        return {"ok": False, "error": "Invalid event_type"}

    row = {
        "timestamp": _now_iso(),
        "event_type": event_type,
        "search_id": payload.search_id,
        "user_id": payload.user_id,
        "query": payload.query,
        "product_id": payload.product_id,
        "position": payload.position,
        "metadata": payload.metadata,
    }
    _append_jsonl(events_file, row)
    return {"ok": True}


@app.get("/analytics/drift")
def drift_summary(window: int = Query(500, ge=50, le=5000)) -> Dict[str, Any]:
    if products_df is None:
        raise RuntimeError("Artifacts not loaded")

    logs = _load_jsonl(impressions_file, max_rows=window)
    if not logs:
        return {"num_impressions": 0, "message": "No impression logs yet"}

    baseline = products_df["category"].value_counts(normalize=True).to_dict()
    recent_counts: Dict[str, int] = {}
    strategy_latency: Dict[str, List[float]] = {}
    for row in logs:
        strategy = str(row.get("strategy", "unknown"))
        strategy_latency.setdefault(strategy, []).append(float(row.get("latency_ms", 0.0)))
        for item in row.get("results", []):
            cat = str(item.get("category", "unknown"))
            recent_counts[cat] = recent_counts.get(cat, 0) + 1

    total_recent = sum(recent_counts.values()) or 1
    recent_dist = {k: v / total_recent for k, v in recent_counts.items()}
    all_cats = set(baseline.keys()).union(recent_dist.keys())
    l1_drift = 0.5 * sum(abs(recent_dist.get(c, 0.0) - baseline.get(c, 0.0)) for c in all_cats)
    latency_breakdown = {
        k: round(sum(v) / len(v), 2) for k, v in strategy_latency.items() if len(v) > 0
    }

    return {
        "num_impressions": len(logs),
        "category_drift_l1": round(l1_drift, 4),
        "recent_category_distribution": recent_dist,
        "baseline_category_distribution": baseline,
        "avg_latency_by_strategy_ms": latency_breakdown,
    }
