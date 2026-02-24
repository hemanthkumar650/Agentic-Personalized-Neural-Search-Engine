# Technical Design: Agentic Personalized Neural Search Engine

## Problem
Modern search needs both relevance and personalization:
- Retrieve the right candidates quickly.
- Rank candidates with richer behavioral features.
- Adapt ranking for different users on the same query.

## Approach
1. Retrieval Layer
- Sparse retrieval with BM25 on `title + description`.
- Dense retrieval with sentence embeddings and FAISS cosine-like similarity.
- Hybrid retrieval merges sparse + dense scores.

2. Ranking Layer
- LambdaMART model trained on interaction-derived supervision.
- Feature set:
  - lexical (`bm25_score`)
  - semantic (`cosine_similarity`)
  - blended (`hybrid_score`)
  - behavior priors (`product_popularity`, `user_category_preference`)
  - query metadata (`query_length`)
  - heuristic commercial feature (`price_match_indicator`)

3. Personalization Layer
- User embedding computed as weighted average of interacted product embeddings.
- Final rerank applies user-product affinity as personalization score.

4. Recommendation Module
- Item-to-item co-occurrence recommender from user interaction histories.
- Adds non-query discovery capability (`/recommend` endpoint).

## Data and Labels
- Uses `data/products.csv` and `data/interactions.csv`.
- Labels are implicit from events (`view/click/cart/purchase`) with weights.

## Serving Design
- FastAPI startup loads retrieval index, ranker, and user embeddings.
- `/search` endpoint supports hybrid/ranker mode and optional personalization.
- `/recommend` endpoint serves personalized item-to-item recommendations.

## Evaluation
- Metrics: `NDCG@10`, `MRR`, `Recall@10`, `Precision@10`.
- `evaluation/ablation_study.py` compares:
  - BM25
  - Dense
  - Hybrid
  - Ranker
  - Personalized

## Tradeoffs
- Pros:
  - Modular architecture, easy to iterate each stage.
  - Strong observability with metrics and run artifacts.
  - Clear transition path from prototype to larger datasets.
- Cons:
  - Event-derived labels are noisy and biased by exposure.
  - Query simulation from product titles is simplistic.
  - Small dataset size limits robust generalization.

## Known Limitations
- Limited negative sampling and no hard-negative mining.
- No online feedback loop/real-time model updates.
- Recommendation is co-occurrence baseline, not sequence-aware.
