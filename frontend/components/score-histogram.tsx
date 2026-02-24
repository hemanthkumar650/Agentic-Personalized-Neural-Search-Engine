"use client";

import { useEffect, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from "recharts";

export type HistogramDatum = { name: string; score: number; rawScore?: number };

const CHART_HEIGHT = 260;
const MIN_CHART_WIDTH = 400;

export type HistogramMode = "deviation" | "raw";

export default function ScoreHistogram({
  data,
  mode = "deviation",
}: {
  data: HistogramDatum[];
  mode?: HistogramMode;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: CHART_HEIGHT });

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;
    const el = containerRef.current;
    const update = () => {
      const w = el.offsetWidth || MIN_CHART_WIDTH;
      setSize((prev) => ({
        width: Math.max(w, MIN_CHART_WIDTH),
        height: prev.height,
      }));
    };
    update();
    const raf = requestAnimationFrame(update);
    const ro = new ResizeObserver(update);
    ro.observe(el);
    const t = setTimeout(update, 100);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(t);
      ro.disconnect();
    };
  }, [data.length]);

  if (data.length === 0) return null;

  const isRaw = mode === "raw";
  const chartData = isRaw
    ? data.map((d) => ({ ...d, score: d.rawScore ?? d.score }))
    : data;
  const domain: [number, number] = isRaw
    ? [0, Math.max(0.1, ...chartData.map((d) => d.score))]
    : (() => {
        const yExtent = Math.max(0.5, ...data.map((d) => Math.abs(d.score)));
        return [-yExtent, yExtent];
      })();

  return (
    <div
      ref={containerRef}
      className="w-full overflow-hidden rounded border border-slate-200 bg-slate-50/50"
      style={{ minHeight: CHART_HEIGHT + 40 }}
    >
      {size.width > 0 && (
        <BarChart
          width={size.width}
          height={CHART_HEIGHT}
          data={chartData}
          margin={{ top: 16, right: 16, bottom: 24, left: 24 }}
          layout="horizontal"
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#cbd5e1" />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#64748b" />
          <YAxis
            domain={domain}
            tick={{ fontSize: 12 }}
            tickCount={5}
            stroke="#64748b"
            tickFormatter={(v) => (typeof v === "number" ? v.toFixed(2) : String(v))}
          />
          <Tooltip
            formatter={(value: number, _name: string, props: { payload: HistogramDatum }) => [
              isRaw
                ? Number(value).toFixed(4)
                : `${Number(value).toFixed(4)}${props.payload.rawScore != null ? ` (raw: ${props.payload.rawScore.toFixed(4)})` : ""}`,
              isRaw ? "Score" : "Deviation from mean",
            ]}
            labelFormatter={(label) => `Product ${label}`}
            contentStyle={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 6 }}
          />
          <Bar dataKey="score" fill="#0f766e" isAnimationActive={true} />
        </BarChart>
      )}
    </div>
  );
}
