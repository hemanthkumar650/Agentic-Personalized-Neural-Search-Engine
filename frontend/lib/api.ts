const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

function handleFetchError(err: unknown, context: string): never {
  if (err instanceof TypeError && err.message === "Failed to fetch") {
    throw new Error(
      `Cannot reach the API at ${API_BASE}. Start the backend (e.g. run "python main.py --mode backend" or "uvicorn api.app:app --port 8000") and ensure NEXT_PUBLIC_API_BASE in .env.local points to it.`
    );
  }
  throw err instanceof Error ? err : new Error(String(err));
}

export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!res.ok) throw new Error("Backend health check failed");
    return res.json();
  } catch (e) {
    handleFetchError(e, "health");
  }
}

export async function runSearch(params: {
  q: string;
  user_id: string;
  top_k: number;
  alpha: number;
  personalization_weight: number;
  strategy?: string;
}) {
  const qs = new URLSearchParams({
    q: params.q,
    user_id: params.user_id,
    top_k: String(params.top_k),
    alpha: String(params.alpha),
    personalization_weight: String(params.personalization_weight),
    strategy: params.strategy || "auto",
  });
  try {
    const res = await fetch(`${API_BASE}/search?${qs.toString()}`, { cache: "no-store" });
    if (!res.ok) throw new Error("Search failed");
    return res.json();
  } catch (e) {
    handleFetchError(e, "search");
  }
}

export async function compareStrategies(params: {
  q: string;
  user_id: string;
  top_k: number;
  alpha: number;
  personalization_weight: number;
}) {
  const strategies = ["bm25", "dense", "hybrid", "ranker", "personalized"];
  try {
    const runs = await Promise.all(
      strategies.map(async (strategy) => {
        const data = await runSearch({ ...params, strategy });
        return { strategy, data };
      })
    );
    return runs;
  } catch (e) {
    handleFetchError(e, "compare");
  }
}

export async function logUserEvent(payload: {
  event_type: "view" | "click" | "cart" | "purchase";
  search_id: string;
  user_id: string;
  query: string;
  product_id: string;
  position: number;
  metadata?: Record<string, unknown>;
}) {
  try {
    const res = await fetch(`${API_BASE}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Event logging failed");
    return res.json();
  } catch (e) {
    handleFetchError(e, "events");
  }
}

export async function fetchRecommend(params: { user_id: string; top_k: number }) {
  const qs = new URLSearchParams({
    user_id: params.user_id,
    top_k: String(params.top_k),
  });
  try {
    const res = await fetch(`${API_BASE}/recommend?${qs.toString()}`, { cache: "no-store" });
    if (!res.ok) throw new Error("Recommendations failed");
    return res.json();
  } catch (e) {
    handleFetchError(e, "recommend");
  }
}

export async function fetchDrift() {
  try {
    const res = await fetch(`${API_BASE}/analytics/drift`, { cache: "no-store" });
    if (!res.ok) throw new Error("Drift fetch failed");
    return res.json();
  } catch (e) {
    handleFetchError(e, "drift");
  }
}

export async function fetchUserSegment(userId: string) {
  try {
    const res = await fetch(`${API_BASE}/user/${encodeURIComponent(userId)}/segment`, { cache: "no-store" });
    if (!res.ok) throw new Error("Segment fetch failed");
    return res.json();
  } catch (e) {
    handleFetchError(e, "segment");
  }
}

export async function fetchSegments() {
  try {
    const res = await fetch(`${API_BASE}/segments`, { cache: "no-store" });
    if (!res.ok) throw new Error("Segments fetch failed");
    return res.json();
  } catch (e) {
    handleFetchError(e, "segments");
  }
}

export async function fetchContentSimilar(productId: string, topK: number = 5) {
  try {
    const res = await fetch(
      `${API_BASE}/content/similar?product_id=${encodeURIComponent(productId)}&top_k=${topK}`,
      { cache: "no-store" }
    );
    if (!res.ok) throw new Error("Content similar failed");
    return res.json();
  } catch (e) {
    handleFetchError(e, "content/similar");
  }
}

export async function sendConversation(params: { user_id: string; message: string }) {
  try {
    const res = await fetch(`${API_BASE}/conversation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: params.user_id, message: params.message }),
    });
    if (!res.ok) throw new Error("Conversation failed");
    return res.json();
  } catch (e) {
    handleFetchError(e, "conversation");
  }
}
