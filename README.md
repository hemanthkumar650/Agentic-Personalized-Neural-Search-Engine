# Agentic Personalized Neural Search Engine

Production-style ML system for search relevance, personalization, and recommendations.

## What I achieved in this project
- Built an end-to-end retrieval -> ranking -> personalization pipeline using real data from `data/`.
- Implemented hybrid retrieval (`BM25 + dense`) and FAISS vector indexing.
- Trained LambdaMART ranker with explicit, interpretable ranking features.
- Added user-behavior personalization so the same query can rank differently by user.
- Added an item-to-item recommendation module for non-query product discovery.
- Added evaluation, ablation, experiment tracking, and error-analysis utilities.
- Served everything through FastAPI and visualized outputs in a Next.js frontend.

## System architecture
1. Retrieval
- BM25 (`rank-bm25`) on `title + description`.
- Dense retrieval with Sentence Transformers.
- FAISS vector index for semantic nearest neighbors.
- Hybrid score combining sparse and dense signals.

2. Ranking
- LightGBM LambdaMART.
- Feature set:
`bm25_score`, `cosine_similarity`, `hybrid_score`, `product_popularity`,
`user_category_preference`, `query_length`, `price_match_indicator`.

3. Personalization
- User embeddings from weighted interaction history (`view/click/cart/purchase`).
- Personalized reranking via user-product embedding affinity.

4. Recommendation
- Item-to-item co-occurrence recommender (`agents/recommendation_agent.py`).
- Exposed via `/recommend`.

## Repository structure
```text
agentic-personalized-neural-search/
├── agents/                  # retrieval, ranking, personalization, recommendation agents
├── api/                     # FastAPI service
├── data/                    # input datasets
├── docs/                    # technical design docs
├── evaluation/              # metrics, pipeline evaluation, ablation, error analysis
├── frontend/                # Next.js + TypeScript dashboard
├── models/                  # model wrappers
├── runs/                    # experiment artifacts
├── tests/                   # unit tests
├── utils/                   # preprocessing, feature engineering, experiment tracking
├── build_index.py           # builds retrieval index + user embeddings
├── train_ranker.py          # trains LambdaMART
├── main.py                  # backend/frontend/fullstack launcher
└── requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
python build_index.py
python train_ranker.py
python main.py --mode fullstack
```

Manual start:
```bash
uvicorn api.app:app --reload --port 8000
cd frontend && npm install && npm run dev
```

## API endpoints
- `GET /health`
- `GET /search?q=wireless+headphones&user_id=user_1&top_k=5&strategy=ranker`
- `GET /recommend?user_id=user_1&top_k=5`
- `POST /events` for click/cart/purchase/view logging
- `GET /analytics/drift` for quick drift + latency diagnostics from logs

## Evaluation and analysis
Main evaluation:
```bash
python evaluation/evaluate_pipeline.py
```

Ablation study:
```bash
python evaluation/ablation_study.py
```

EDA + error analysis:
```bash
python evaluation/eda_error_analysis.py
```

Run artifacts are written to `runs/`.

## Experiment tracking
- Each training/evaluation/ablation run appends to:
`runs/experiment_summary.csv`
- This gives a compact comparison table across multiple runs.
- Online event logs are written to:
`runs/logs/impressions.jsonl` and `runs/logs/events.jsonl`

## Ablation template (README table)
After running `evaluation/ablation_study.py`, copy key values here:

| Strategy | NDCG@10 | MRR | Recall@10 | Precision@10 |
|---|---:|---:|---:|---:|
| BM25 | - | - | - | - |
| Dense | - | - | - | - |
| Hybrid | - | - | - | - |
| Ranker | - | - | - | - |
| Personalized | - | - | - | - |

## Technical design
- Detailed design and tradeoffs:
`docs/technical_design.md`

## Current limitations
- Labels are implicit from behavior events, not curated relevance judgments.
- Training queries are title-derived and can bias offline metrics.
- Dataset is small, so learned ranking/personalization can be unstable.
- Recommendation module is a strong baseline but not sequence-aware.
