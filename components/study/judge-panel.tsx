"use client";

import { useState } from "react";

import { BASE_PATH, conversationHref, precisionRecall, PRESET_THRESHOLD, type ValidationPair } from "@/lib/study";

const W = 840;
const H = 120;
const X = (v: number) => 20 + v * (W - 40);

export function JudgePanel({
  pairs,
  keyword,
}: {
  pairs: ValidationPair[];
  keyword?: { precision: number; recall: number };
}) {
  const [t, setT] = useState(PRESET_THRESHOLD);
  const m = precisionRecall(pairs, t);
  const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "—");
  const positives = pairs.filter((p) => p.label === 1).length;

  return (
    <div className="space-y-4">
      <label className="flex flex-wrap items-center gap-3 text-[12px] text-text-secondary">
        Threshold
        <input
          type="range"
          min={0.05}
          max={0.95}
          step={0.05}
          value={t}
          onChange={(e) => setT(Number(e.target.value))}
          className="w-56 accent-[var(--color-accent)]"
          aria-label="Pushback threshold"
        />
        <span className="num text-text-primary">{t.toFixed(2)}</span>
        {t === PRESET_THRESHOLD && <span className="text-text-muted">set in advance</span>}
        {t === 0.8 && <span className="text-[var(--color-status-warning)]">chosen after the fact on this sample</span>}
      </label>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Jev pushback scores of the 60 validation messages">
        <text x={20} y={22} fontSize={11} fill="var(--color-text-muted)">Labelled pushback ({positives})</text>
        <text x={20} y={74} fontSize={11} fill="var(--color-text-muted)">Labelled not pushback ({pairs.length - positives})</text>
        <line x1={X(t)} x2={X(t)} y1={10} y2={H - 18} stroke="var(--color-accent)" strokeDasharray="4 3" />
        {pairs.map((p, i) => (
          <a key={i} href={`${BASE_PATH}${conversationHref(p.run_id)}`}>
            <circle
              cx={X(p.score)}
              cy={(p.label === 1 ? 38 : 90) + ((i * 7) % 11) - 5}
              r={4.5}
              fill={p.score >= t ? "var(--color-accent)" : "var(--color-surface-1)"}
              stroke="var(--color-accent)"
            >
              <title>{`score ${p.score.toFixed(2)} · ${p.label ? "pushback" : "not pushback"} · open the conversation`}</title>
            </circle>
          </a>
        ))}
        <text x={20} y={H - 3} fontSize={10.5} fill="var(--color-text-muted)">0</text>
        <text x={W - 28} y={H - 3} fontSize={10.5} fill="var(--color-text-muted)">1.0</text>
      </svg>

      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Metric label="Precision" value={fmt(m.precision)} />
        <Metric label="Recall" value={fmt(m.recall)} />
        <Metric label="Flagged" value={`${m.flagged} of ${pairs.length}`} />
        <Metric
          label="Keyword baseline"
          value={keyword ? `${keyword.precision.toFixed(2)} / ${keyword.recall.toFixed(2)}` : "—"}
          hint="precision / recall"
        />
      </div>
      <p className="text-[11.5px] leading-relaxed text-text-muted">
        60 user messages, stratified on the highest of all 5 user-message questions (including &quot;says it is
        done&quot;): 30 scored at least 0.5 there and 30 below. The pushback score uses 4 of them, so fewer than 30 are
        flagged at 0.5. Labels come from a separate LLM that
        did not see the scores, not from humans. Precision on a stratified sample is not the population rate, and recall
        of 1.00 on {positives} positives has a wide interval. Filled dots are flagged at the current threshold; click a dot
        to open its conversation.
      </p>
    </div>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-md bg-surface-2 px-3 py-2">
      <div className="label-caps">{label}</div>
      <div className="num mt-1 text-[16px]">{value}</div>
      {hint && <div className="text-[11px] text-text-muted">{hint}</div>}
    </div>
  );
}
