"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import {
  compareStrategies,
  fetchContentSimilar,
  fetchDrift,
  fetchHealth,
  fetchRecommend,
  fetchUserSegment,
  logUserEvent,
  runSearch,
  sendConversation,
} from "../lib/api";
import LatencyChart from "./latency-chart";
import type { LatencyDatum } from "./latency-chart";
import type { HistogramMode } from "./score-histogram";

const ScoreHistogram = dynamic(() => import("./score-histogram"), { ssr: false });

type SearchResult = {
  rank: number;
  product_id: string;
  title: string;
  category: string;
  price: number;
  score: number;
};

type RecommendResult = SearchResult & { reason?: string };

type RecommendResponse = {
  user_id: string;
  num_results: number;
  results: RecommendResult[];
};

type SearchResponse = {
  query: string;
  user_id: string;
  strategy: string;
  search_id: string;
  personalized: boolean;
  num_results: number;
  latency_ms: number;
  results: SearchResult[];
};

type SegmentResponse = {
  user_id: string;
  segment: string;
  engagement: string;
  preferred_category: string;
  event_count: number;
};

type ContentSimilarItem = { product_id: string; title: string; category: string; price: number; content_similarity: number };
type ContentSimilarResponse = { product_id: string; similar: ContentSimilarItem[] };

type ConversationMessage = { role: "user" | "assistant"; text: string; results?: unknown[]; intent?: string };

export default function SearchDashboard() {
  const [health, setHealth] = useState<any>(null);
  const [drift, setDrift] = useState<any>(null);
  const [query, setQuery] = useState("wireless headphones");
  const [userId, setUserId] = useState("user_1");
  const [topK, setTopK] = useState(5);
  const [alpha, setAlpha] = useState(0.5);
  const [pw, setPw] = useState(0.3);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [compare, setCompare] = useState<Array<{ strategy: string; data: SearchResponse }> | null>(null);
  const [eventStatus, setEventStatus] = useState("");
  const [apiError, setApiError] = useState<string | null>(null);
  const [recommendTopK, setRecommendTopK] = useState(10);
  const [recommendations, setRecommendations] = useState<RecommendResponse | null>(null);
  const [recommendLoading, setRecommendLoading] = useState(false);
  const [scoreChartMode, setScoreChartMode] = useState<HistogramMode>("deviation");
  const [userSegment, setUserSegment] = useState<SegmentResponse | null>(null);
  const [contentSimilarProductId, setContentSimilarProductId] = useState("");
  const [contentSimilar, setContentSimilar] = useState<ContentSimilarResponse | null>(null);
  const [conversationMessages, setConversationMessages] = useState<ConversationMessage[]>([]);
  const [conversationInput, setConversationInput] = useState("");
  const [conversationLoading, setConversationLoading] = useState(false);

  async function onHealth() {
    setApiError(null);
    try {
      setHealth(await fetchHealth());
    } catch (e) {
      setApiError(e instanceof Error ? e.message : String(e));
    }
  }

  async function onSearch() {
    setApiError(null);
    try {
      const data = await runSearch({ q: query, user_id: userId, top_k: topK, alpha, personalization_weight: pw });
      setResult(data);
      setCompare(null);
    } catch (e) {
      setApiError(e instanceof Error ? e.message : String(e));
    }
  }

  async function onCompare() {
    setApiError(null);
    try {
      const runs = await compareStrategies({ q: query, user_id: userId, top_k: topK, alpha, personalization_weight: pw });
      setCompare(runs);
      const selected = runs.find((x) => x.strategy === "personalized") || runs[0];
      setResult(selected.data);
    } catch (e) {
      setApiError(e instanceof Error ? e.message : String(e));
    }
  }

  async function onDrift() {
    setApiError(null);
    try {
      setDrift(await fetchDrift());
    } catch (e) {
      setApiError(e instanceof Error ? e.message : String(e));
    }
  }

  async function onRecommend() {
    setApiError(null);
    setRecommendLoading(true);
    try {
      const data = await fetchRecommend({ user_id: userId, top_k: recommendTopK });
      setRecommendations(data);
    } catch (e) {
      setApiError(e instanceof Error ? e.message : String(e));
    } finally {
      setRecommendLoading(false);
    }
  }

  async function onLoadSegment() {
    setApiError(null);
    try {
      const data = await fetchUserSegment(userId);
      setUserSegment(data);
    } catch (e) {
      setApiError(e instanceof Error ? e.message : String(e));
    }
  }

  async function onContentSimilar() {
    if (!contentSimilarProductId.trim()) return;
    setApiError(null);
    try {
      const data = await fetchContentSimilar(contentSimilarProductId.trim(), 5);
      setContentSimilar(data);
    } catch (e) {
      setApiError(e instanceof Error ? e.message : String(e));
    }
  }

  async function onConversationSend() {
    const msg = conversationInput.trim();
    if (!msg || conversationLoading) return;
    setConversationMessages((prev) => [...prev, { role: "user", text: msg }]);
    setConversationInput("");
    setConversationLoading(true);
    setApiError(null);
    try {
      const data = await sendConversation({ user_id: userId, message: msg });
      setConversationMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.intent === "recommend" ? "Here are some recommendations for you." : `Search results for "${data.query}".`,
          results: data.results,
          intent: data.intent,
        },
      ]);
    } catch (e) {
      setApiError(e instanceof Error ? e.message : String(e));
      setConversationMessages((prev) => [...prev, { role: "assistant", text: "Sorry, something went wrong.", results: [] }]);
    } finally {
      setConversationLoading(false);
    }
  }

  async function track(event_type: "click" | "cart" | "purchase", row: SearchResult) {
    if (!result) return;
    await logUserEvent({
      event_type,
      search_id: result.search_id,
      user_id: userId,
      query,
      product_id: row.product_id,
      position: row.rank,
      metadata: { strategy: result.strategy, score: row.score },
    });
    setEventStatus(`Logged ${event_type} for ${row.product_id}`);
  }

  const scoreChartData = useMemo(() => {
    const results = result?.results || [];
    if (results.length === 0) return [];
    const scores = results.map((r) => Number(r.score) ?? 0);
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    return results.map((r, i) => {
      const id = r.product_id;
      const name = /^\d+$/.test(String(id)) ? `P${String(id).padStart(3, "0")}` : id;
      const rawScore = scores[i];
      return { name, score: rawScore - mean, rawScore };
    });
  }, [result]);

  const latencyData = useMemo((): LatencyDatum[] => {
    if (!compare) return [];
    return compare.map((x) => ({
      strategy: x.strategy,
      latency_ms: x.data.latency_ms,
    }));
  }, [compare]);

  return (
    <main className="mx-auto max-w-7xl space-y-6 p-6">
      <h1 className="text-4xl font-semibold">Agentic Personalized Neural Search</h1>

      <div className="flex flex-wrap gap-3">
        <button className="rounded bg-slate-900 px-4 py-2 text-white" onClick={onHealth}>
          Check Backend Health
        </button>
        <button className="rounded bg-indigo-700 px-4 py-2 text-white" onClick={onDrift}>
          Drift Summary
        </button>
        {eventStatus && <div className="rounded bg-emerald-100 px-3 py-2 text-sm">{eventStatus}</div>}
        {apiError && (
          <div className="rounded bg-red-100 px-4 py-3 text-sm text-red-800" role="alert">
            <strong>API error:</strong> {apiError}
          </div>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {health && <pre className="overflow-auto rounded bg-white p-3 text-xs">{JSON.stringify(health, null, 2)}</pre>}
        {drift && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
                <div className="text-xs font-medium uppercase text-slate-500">Impressions</div>
                <div className="mt-1 text-xl font-semibold">{drift.num_impressions ?? 0}</div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
                <div className="text-xs font-medium uppercase text-slate-500">Category drift (L1)</div>
                <div className="mt-1 text-xl font-semibold">{(drift.category_drift_l1 ?? 0).toFixed(4)}</div>
              </div>
              {drift.avg_latency_by_strategy_ms && Object.keys(drift.avg_latency_by_strategy_ms).length > 0 && (
                <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm sm:col-span-2 sm:col-start-1">
                  <div className="text-xs font-medium uppercase text-slate-500">Avg latency by strategy (ms)</div>
                  <div className="mt-1 flex flex-wrap gap-2 text-sm">
                    {Object.entries(drift.avg_latency_by_strategy_ms).map(([s, ms]) => (
                      <span key={s} className="rounded bg-slate-100 px-2 py-0.5">
                        {s}: <strong>{Number(ms).toFixed(0)}</strong>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <details className="rounded bg-white p-2">
              <summary className="cursor-pointer text-xs text-slate-500">Raw drift JSON</summary>
              <pre className="mt-2 overflow-auto p-2 text-xs">{JSON.stringify(drift, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>

      <div className="grid gap-4 rounded bg-white p-4 md:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold text-slate-700">User segmentations</h3>
          <p className="mb-2 text-xs text-slate-500">Segment for current user (engagement + preferred category).</p>
          <div className="flex gap-2">
            <button
              type="button"
              className="rounded bg-slate-700 px-3 py-1.5 text-sm text-white"
              onClick={onLoadSegment}
            >
              Load segment
            </button>
          </div>
          {userSegment && (
            <div className="mt-2 rounded border border-slate-200 bg-slate-50 p-3 text-sm">
              <div><strong>Segment:</strong> {userSegment.segment}</div>
              <div><strong>Engagement:</strong> {userSegment.engagement}</div>
              <div><strong>Preferred category:</strong> {userSegment.preferred_category}</div>
              <div><strong>Event count:</strong> {userSegment.event_count}</div>
            </div>
          )}
        </div>
        <div>
          <h3 className="mb-2 text-sm font-semibold text-slate-700">Content intelligence</h3>
          <p className="mb-2 text-xs text-slate-500">Similar products by content (title + description embedding).</p>
          <div className="flex gap-2">
            <input
              className="flex-1 rounded border p-2 text-sm"
              placeholder="Product ID (e.g. P002 or 2)"
              value={contentSimilarProductId}
              onChange={(e) => setContentSimilarProductId(e.target.value)}
            />
            <button
              type="button"
              className="rounded bg-slate-700 px-3 py-1.5 text-sm text-white"
              onClick={onContentSimilar}
            >
              Similar
            </button>
          </div>
          {contentSimilar && (
            <div className="mt-2 max-h-40 overflow-auto rounded border border-slate-200 text-xs">
              <table className="w-full">
                <thead><tr><th className="p-1 text-left">Product</th><th className="p-1">Similarity</th></tr></thead>
                <tbody>
                  {contentSimilar.similar.map((s) => (
                    <tr key={s.product_id} className="border-t border-slate-100">
                      <td className="p-1">{s.title}</td>
                      <td className="p-1">{s.content_similarity.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <div className="space-y-3 rounded bg-white p-4">
        <input className="w-full rounded border p-2" value={query} onChange={(e) => setQuery(e.target.value)} />
        <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
          <input className="rounded border p-2" value={userId} onChange={(e) => setUserId(e.target.value)} />
          <input className="rounded border p-2" type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
          <input className="rounded border p-2" type="number" step="0.1" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
          <input className="rounded border p-2" type="number" step="0.1" value={pw} onChange={(e) => setPw(Number(e.target.value))} />
          <div className="flex gap-2">
            <button className="w-full rounded bg-teal-700 px-4 py-2 text-white" onClick={onSearch}>
              Run Search
            </button>
            <button className="w-full rounded bg-purple-700 px-4 py-2 text-white" onClick={onCompare}>
              Compare
            </button>
          </div>
        </div>
      </div>

      <div className="rounded bg-white p-4">
        <h2 className="mb-3 text-xl font-semibold">Recommendations</h2>
        <p className="mb-3 text-sm text-slate-600">Item-to-item recommendations for the current user (no query).</p>
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <label className="text-sm">Top-K</label>
          <input
            className="w-20 rounded border p-2 text-sm"
            type="number"
            min={1}
            max={50}
            value={recommendTopK}
            onChange={(e) => setRecommendTopK(Number(e.target.value) || 10)}
          />
          <button
            className="rounded bg-amber-600 px-4 py-2 text-white disabled:opacity-50"
            onClick={onRecommend}
            disabled={recommendLoading}
          >
            {recommendLoading ? "Loading…" : "Load recommendations"}
          </button>
        </div>
        {recommendations && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr>
                  <th className="p-2">Rank</th>
                  <th className="p-2">Product</th>
                  <th className="p-2">Category</th>
                  <th className="p-2">Price</th>
                  <th className="p-2">Score</th>
                  {recommendations.results.some((r) => r.reason) && <th className="p-2">Reason</th>}
                </tr>
              </thead>
              <tbody>
                {recommendations.results.map((r) => (
                  <tr key={r.product_id} className="border-t border-slate-100">
                    <td className="p-2">{r.rank}</td>
                    <td className="p-2">{r.title ?? "-"}</td>
                    <td className="p-2">{r.category ?? "-"}</td>
                    <td className="p-2">${typeof r.price === "number" ? r.price.toFixed(2) : Number(r.price).toFixed(2)}</td>
                    <td className="p-2">{typeof r.score === "number" ? r.score.toFixed(4) : r.score}</td>
                    {recommendations.results.some((x) => x.reason) && <td className="p-2 text-xs text-slate-500">{r.reason ?? "-"}</td>}
                  </tr>
                ))}
                {recommendations.results.length === 0 && (
                  <tr>
                    <td colSpan={6} className="p-3 text-slate-500">No recommendations for this user.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="rounded bg-white p-4">
        <h2 className="mb-2 text-xl font-semibold">Conversational commerce</h2>
        <p className="mb-3 text-sm text-slate-600">
          Type a search query (e.g. &quot;wireless headphones&quot;) or ask for recommendations (e.g. &quot;suggest something for me&quot;). Uses current user.
        </p>
        <div className="mb-3 max-h-64 overflow-y-auto rounded border border-slate-200 bg-slate-50/50 p-3">
          {conversationMessages.length === 0 && (
            <div className="text-center text-xs text-slate-400">No messages yet. Send a query or ask for recommendations.</div>
          )}
          {conversationMessages.map((m, i) => (
            <div key={i} className={`mb-3 ${m.role === "user" ? "text-right" : ""}`}>
              <div
                className={`inline-block max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                  m.role === "user" ? "bg-teal-600 text-white" : "bg-white text-slate-800 shadow"
                }`}
              >
                {m.text}
              </div>
              {m.results && m.results.length > 0 && (
                <div className="mt-2 rounded border border-slate-200 bg-white p-2 text-left text-xs">
                  <div className="font-semibold text-slate-600">Results ({m.results.length})</div>
                  <ul className="mt-1 list-inside list-disc space-y-0.5">
                    {(m.results as { title?: string; product_id?: string }[]).slice(0, 5).map((r, j) => (
                      <li key={j}>{r.title ?? r.product_id}</li>
                    ))}
                    {m.results.length > 5 && <li>… and {m.results.length - 5} more</li>}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            className="flex-1 rounded border p-2 text-sm"
            placeholder="Search or ask for recommendations..."
            value={conversationInput}
            onChange={(e) => setConversationInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onConversationSend()}
          />
          <button
            type="button"
            className="rounded bg-teal-600 px-4 py-2 text-white disabled:opacity-50"
            onClick={onConversationSend}
            disabled={conversationLoading}
          >
            {conversationLoading ? "…" : "Send"}
          </button>
        </div>
      </div>

      {!result && (
        <div className="rounded border border-slate-200 bg-slate-50 p-6 text-center text-sm text-slate-500">
          Run <strong>Run Search</strong> or <strong>Compare</strong> above — results and the score histogram will appear here.
        </div>
      )}

      {compare && (
        <div className="rounded bg-white p-4">
          <h2 className="mb-3 text-xl font-semibold">Strategy Comparison (Top-1 + Latency)</h2>
          <table className="mb-4 w-full text-left text-sm">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Top-1 Product</th>
                <th>Top-1 Score</th>
                <th>Latency (ms)</th>
              </tr>
            </thead>
            <tbody>
              {compare.map((row) => {
                const top = row.data.results?.[0];
                const score = top?.score;
                const scoreStr = typeof score === "number" ? score.toFixed(4) : score != null ? Number(score).toFixed(4) : "-";
                return (
                  <tr key={row.strategy}>
                    <td className="font-semibold">{row.strategy}</td>
                    <td>{top?.title ?? "-"}</td>
                    <td>{scoreStr}</td>
                    <td>{typeof row.data.latency_ms === "number" ? row.data.latency_ms.toFixed(2) : row.data.latency_ms}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="space-y-2">
            <h3 className="text-base font-semibold text-slate-800">Latency by strategy</h3>
            <LatencyChart data={latencyData} />
          </div>
        </div>
      )}

      {result && (
        <div className="space-y-4 rounded bg-white p-4">
          <div className="text-sm">
            Strategy: <b>{result.strategy}</b> | Search ID: <b>{result.search_id}</b> | Latency: <b>{typeof result.latency_ms === "number" ? result.latency_ms.toFixed(2) : result.latency_ms} ms</b>
            {result.num_results !== undefined && (
              <> | Results: <b>{result.num_results}</b></>
            )}
          </div>
          <div className="space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-lg font-semibold text-slate-800">Score distribution (histogram)</h2>
              <div className="flex rounded border border-slate-200 bg-slate-50 p-0.5">
                <button
                  type="button"
                  className={`rounded px-3 py-1 text-sm ${scoreChartMode === "deviation" ? "bg-teal-600 text-white" : "text-slate-600"}`}
                  onClick={() => setScoreChartMode("deviation")}
                >
                  Deviation from mean
                </button>
                <button
                  type="button"
                  className={`rounded px-3 py-1 text-sm ${scoreChartMode === "raw" ? "bg-teal-600 text-white" : "text-slate-600"}`}
                  onClick={() => setScoreChartMode("raw")}
                >
                  Raw scores
                </button>
              </div>
            </div>
            {scoreChartMode === "deviation" && (
              <p className="text-xs text-slate-500">
                Bars show score <em>relative to the mean</em>: above the line = above-average relevance, below = below-average.
              </p>
            )}
            {scoreChartData.length === 0 ? (
              <div className="rounded border border-dashed border-slate-300 bg-slate-50 p-6 text-center text-sm text-slate-600">
                No results to plot for this query/strategy.
              </div>
            ) : (
              <ScoreHistogram data={scoreChartData} mode={scoreChartMode} />
            )}
          </div>
          <table className="w-full text-left">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Product</th>
                <th>Category</th>
                <th>Price</th>
                <th>Score</th>
                <th>Track</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((r) => (
                <tr
                  key={r.product_id}
                  className={r.rank === 1 ? "border-l-4 border-l-teal-600 bg-teal-50/70" : ""}
                >
                  <td>{r.rank === 1 ? <span className="font-semibold text-teal-700">1 (top)</span> : r.rank}</td>
                  <td>{r.title ?? "-"}</td>
                  <td>{r.category ?? "-"}</td>
                  <td>${typeof r.price === "number" ? r.price.toFixed(2) : Number(r.price).toFixed(2)}</td>
                  <td>{typeof r.score === "number" ? r.score.toFixed(4) : Number(r.score || 0).toFixed(4)}</td>
                  <td className="space-x-2">
                    <button
                      type="button"
                      className="rounded bg-slate-600 px-2 py-1 text-xs text-white"
                      onClick={async () => {
                        setContentSimilarProductId(r.product_id);
                        setApiError(null);
                        try {
                          const data = await fetchContentSimilar(r.product_id, 5);
                          setContentSimilar(data);
                        } catch (e) {
                          setApiError(e instanceof Error ? e.message : String(e));
                        }
                      }}
                    >
                      similar
                    </button>
                    <button className="rounded bg-slate-700 px-2 py-1 text-xs text-white" onClick={() => track("click", r)}>
                      click
                    </button>
                    <button className="rounded bg-amber-600 px-2 py-1 text-xs text-white" onClick={() => track("cart", r)}>
                      cart
                    </button>
                    <button className="rounded bg-emerald-700 px-2 py-1 text-xs text-white" onClick={() => track("purchase", r)}>
                      buy
                    </button>
                  </td>
                </tr>
              ))}
              {result.results.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-3 text-sm text-slate-600">
                    No results returned.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
