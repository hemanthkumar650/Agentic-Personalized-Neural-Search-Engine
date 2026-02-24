"use client";

import { useEffect, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from "recharts";

export type LatencyDatum = { strategy: string; latency_ms: number };

const CHART_HEIGHT = 220;
const MIN_WIDTH = 400;

export default function LatencyChart({ data }: { data: LatencyDatum[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;
    const el = containerRef.current;
    const update = () => {
      const w = el.offsetWidth || MIN_WIDTH;
      setWidth(Math.max(w, MIN_WIDTH));
    };
    update();
    requestAnimationFrame(update);
    const ro = new ResizeObserver(update);
    ro.observe(el);
    const t = setTimeout(update, 100);
    return () => {
      ro.disconnect();
      clearTimeout(t);
    };
  }, [data.length]);

  if (data.length === 0) return null;

  return (
    <div
      ref={containerRef}
      className="w-full overflow-hidden rounded border border-slate-200 bg-slate-50/50"
      style={{ minHeight: CHART_HEIGHT + 24 }}
    >
      {width > 0 && (
        <BarChart
          width={width}
          height={CHART_HEIGHT}
          data={data}
          margin={{ top: 16, right: 16, bottom: 24, left: 24 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#cbd5e1" />
          <XAxis dataKey="strategy" tick={{ fontSize: 12 }} stroke="#64748b" />
          <YAxis tick={{ fontSize: 12 }} stroke="#64748b" label={{ value: "Latency (ms)", angle: -90, position: "insideLeft" }} />
          <Tooltip
            formatter={(value: number) => [`${value} ms`, "Latency"]}
            contentStyle={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 6 }}
          />
          <Bar dataKey="latency_ms" fill="#4f46e5" />
        </BarChart>
      )}
    </div>
  );
}
