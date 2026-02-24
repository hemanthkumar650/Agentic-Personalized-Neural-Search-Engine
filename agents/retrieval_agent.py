import pickle
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


@dataclass
class RetrievalResult:
    product_id: str
    bm25_score: float
    cosine_similarity: float
    hybrid_score: float


class RetrievalAgent:
    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.encoder = SentenceTransformer(encoder_name)
        self.products: pd.DataFrame | None = None
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi | None = None
        self.embeddings: np.ndarray | None = None
        self.id_by_pos: List[str] = []
        self.index = None

    def fit(self, products: pd.DataFrame) -> None:
        self.products = products.copy().reset_index(drop=True)
        self.id_by_pos = self.products["product_id"].astype(str).tolist()
        self.tokenized_corpus = [txt.split() for txt in self.products["product_text"].tolist()]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        embeddings = self.encoder.encode(
            self.products["product_text"].tolist(),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "products": self.products,
                    "tokenized_corpus": self.tokenized_corpus,
                    "embeddings": self.embeddings,
                    "id_by_pos": self.id_by_pos,
                },
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.products = payload["products"]
        self.tokenized_corpus = payload["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.embeddings = payload["embeddings"]
        self.id_by_pos = payload["id_by_pos"]
        if faiss is not None and self.embeddings is not None:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[RetrievalResult]:
        if self.bm25 is None or self.embeddings is None:
            raise RuntimeError("Retrieval agent not initialized")

        bm25_scores = np.asarray(self.bm25.get_scores(query.lower().split()), dtype=np.float32)
        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_norm = bm25_scores

        query_emb = np.asarray(self.encoder.encode([query], normalize_embeddings=True), dtype=np.float32)

        if self.index is not None:
            sims, idxs = self.index.search(query_emb, len(self.id_by_pos))
            dense = np.zeros(len(self.id_by_pos), dtype=np.float32)
            dense[idxs[0]] = sims[0]
        else:
            dense = (self.embeddings @ query_emb[0]).astype(np.float32)

        dmin, dmax = float(dense.min()), float(dense.max())
        dense_norm = (dense - dmin) / (dmax - dmin) if dmax > dmin else dense
        hybrid = alpha * bm25_norm + (1.0 - alpha) * dense_norm

        top_idx = np.argsort(-hybrid)[:top_k]
        out = []
        for i in top_idx:
            out.append(
                RetrievalResult(
                    product_id=self.id_by_pos[i],
                    bm25_score=float(bm25_norm[i]),
                    cosine_similarity=float(dense_norm[i]),
                    hybrid_score=float(hybrid[i]),
                )
            )
        return out

    def embedding_by_product(self) -> Dict[str, np.ndarray]:
        if self.embeddings is None:
            return {}
        return {pid: self.embeddings[i] for i, pid in enumerate(self.id_by_pos)}

