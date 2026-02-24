const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("Backend health check failed");
  return res.json();
}

export async function runSearch(params: {
  q: string;
  user_id: string;
  top_k: number;
  alpha: number;
  personalization_weight: number;
}) {
  const qs = new URLSearchParams({
    q: params.q,
    user_id: params.user_id,
    top_k: String(params.top_k),
    alpha: String(params.alpha),
    personalization_weight: String(params.personalization_weight),
  });
  const res = await fetch(`${API_BASE}/search?${qs.toString()}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Search failed");
  return res.json();
}
