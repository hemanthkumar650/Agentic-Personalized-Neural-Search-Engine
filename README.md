# Agentic Personalized Neural Search Engine

Production-style ML search system for relevance + personalization.

## What this project achieves
- Loads real product and interaction data from `data/`.
- Uses hybrid retrieval: BM25 + dense embeddings.
- Trains LambdaMART on ranking features.
- Personalizes ranking with user behavior embeddings.
- Exposes search through FastAPI and a Next.js UI.

## Pipeline
1. Retrieval
- BM25 (`rank-bm25`)
- Sentence Transformer dense search
- FAISS index (numpy fallback if FAISS unavailable)
- Hybrid scoring

2. Ranking
- LightGBM LambdaMART
- Features:
`bm25_score`, `cosine_similarity`, `hybrid_score`, `product_popularity`,
`user_category_preference`, `query_length`, `price_match_indicator`

3. Personalization
- User embeddings from interaction history
- User-specific reranking for same query

## Run
```bash
pip install -r requirements.txt
python build_index.py
python train_ranker.py
uvicorn api.app:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

## API
- `GET /health`
- `GET /search?q=wireless+headphones&user_id=user_1&top_k=5`

## Evaluation
```bash
python evaluation/evaluate_pipeline.py
```
Computes `NDCG@10`, `MRR`, `Recall@10`, `Precision@10`.

## Limitations
- Labels come from events, not manual relevance judgments.
- Training queries are generated from product titles.
- Small dataset limits generalization of ranker/personalization.
