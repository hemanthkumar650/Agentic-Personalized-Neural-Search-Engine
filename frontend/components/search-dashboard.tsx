"use client";

import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fetchHealth, runSearch } from "../lib/api";

export default function SearchDashboard() {
  const [health, setHealth] = useState<any>(null);
  const [query, setQuery] = useState("wireless headphones");
  const [userId, setUserId] = useState("user_1");
  const [topK, setTopK] = useState(5);
  const [alpha, setAlpha] = useState(0.5);
  const [pw, setPw] = useState(0.3);
  const [result, setResult] = useState<any>(null);

  async function onHealth() {
    setHealth(await fetchHealth());
  }

  async function onSearch() {
    setResult(await runSearch({ q: query, user_id: userId, top_k: topK, alpha, personalization_weight: pw }));
  }

  const chartData = (result?.results || []).map((r: any) => ({ name: r.product_id, score: r.score }));

  return (
    <main className="mx-auto max-w-6xl space-y-6 p-6">
      <h1 className="text-4xl font-semibold">Agentic Personalized Neural Search</h1>

      <div className="flex gap-3">
        <button className="rounded bg-slate-900 px-4 py-2 text-white" onClick={onHealth}>
          Check Backend Health
        </button>
        {health && <pre className="rounded bg-white p-2 text-sm">{JSON.stringify(health, null, 2)}</pre>}
      </div>

      <div className="space-y-3 rounded bg-white p-4">
        <input className="w-full rounded border p-2" value={query} onChange={(e) => setQuery(e.target.value)} />
        <div className="grid grid-cols-4 gap-2">
          <input className="rounded border p-2" value={userId} onChange={(e) => setUserId(e.target.value)} />
          <input className="rounded border p-2" type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
          <input className="rounded border p-2" type="number" step="0.1" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
          <input className="rounded border p-2" type="number" step="0.1" value={pw} onChange={(e) => setPw(Number(e.target.value))} />
        </div>
        <button className="rounded bg-teal-700 px-4 py-2 text-white" onClick={onSearch}>
          Run Search
        </button>
      </div>

      {result && (
        <div className="space-y-4 rounded bg-white p-4">
          <div>
            Latency: <b>{result.latency_ms} ms</b>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="score" fill="#0f766e" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <table className="w-full text-left">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Product</th>
                <th>Category</th>
                <th>Price</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((r: any) => (
                <tr key={r.product_id}>
                  <td>{r.rank}</td>
                  <td>{r.title}</td>
                  <td>{r.category}</td>
                  <td>${r.price}</td>
                  <td>{r.score.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
