import os
import time
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.personalization_agent import PersonalizationAgent
from agents.ranking_agent import RankingAgent
from agents.retrieval_agent import RetrievalAgent
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
    personalized: bool
    num_results: int
    latency_ms: float
    results: List[Dict[str, Any]]


products_df: pd.DataFrame | None = None
feature_context = None
retrieval: RetrievalAgent | None = None
ranking: RankingAgent | None = None
personalization: PersonalizationAgent | None = None


@app.on_event("startup")
def startup() -> None:
    global products_df, feature_context, retrieval, ranking, personalization

    products_df, interactions_df = load_data()
    feature_context = build_feature_context(products_df, interactions_df)

    retrieval = RetrievalAgent()
    retrieval.load("models/retrieval.pkl")

    ranking = RankingAgent("models/ranker.pkl")
    ranking.load()

    user_model = UserEmbeddingModel.load("models/user_embeddings.pkl")
    personalization = PersonalizationAgent(user_model, retrieval.embedding_by_product())


@app.get("/")
def root() -> dict:
    return {
        "name": "Agentic Personalized Neural Search Engine",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "search": "/search?q=wireless+headphones&user_id=user_1&top_k=5",
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
) -> SearchResponse:
    if retrieval is None or ranking is None or personalization is None or products_df is None:
        raise RuntimeError("Artifacts not loaded")

    t0 = time.perf_counter()
    mode = os.getenv("SEARCH_RANKING_MODE", "ranker").strip().lower()
    retrieved = retrieval.retrieve(q, top_k=max(top_k, 10), alpha=alpha)
    candidates = [r.__dict__.copy() for r in retrieved]

    if mode == "hybrid":
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

    personalized = bool(user_id)
    if personalized:
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
    return SearchResponse(
        query=q,
        user_id=user_id,
        personalized=personalized,
        num_results=len(results),
        latency_ms=latency,
        results=results,
    )

