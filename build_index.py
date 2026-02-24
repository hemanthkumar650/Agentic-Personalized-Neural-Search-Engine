from pathlib import Path

from agents.retrieval_agent import RetrievalAgent
from models.user_embedding import UserEmbeddingModel
from utils.preprocessing import load_data


def main() -> None:
    products, interactions = load_data()
    retrieval = RetrievalAgent()
    retrieval.fit(products)

    Path("models").mkdir(exist_ok=True)
    retrieval.save("models/retrieval.pkl")

    product_embeddings = retrieval.embedding_by_product()
    dim = next(iter(product_embeddings.values())).shape[0] if product_embeddings else 384
    user_model = UserEmbeddingModel(dim=dim)
    user_model.fit(interactions, product_embeddings)
    user_model.save("models/user_embeddings.pkl")
    print("Saved models/retrieval.pkl and models/user_embeddings.pkl")


if __name__ == "__main__":
    main()

