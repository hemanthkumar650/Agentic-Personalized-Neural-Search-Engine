import pandas as pd

from agents.recommendation_agent import RecommendationAgent


def test_recommendation_agent_returns_neighbors() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "product_id": ["p1", "p2", "p3", "p1", "p4"],
            "event_type": ["click", "purchase", "click", "click", "purchase"],
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01", "2026-01-02"]
            ),
        }
    )
    agent = RecommendationAgent()
    agent.fit(interactions)
    recs = agent.recommend("u2", top_k=3)
    assert len(recs) > 0
    assert recs[0]["product_id"] in {"p2", "p3"}

