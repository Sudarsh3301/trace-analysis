"use client";

import { useState } from "react";

import type { PooledRow, Study } from "@/lib/study";
import { fmtDelta } from "@/lib/ui";

import { Segmented } from "./auroc-panel";

const EXPOSURES = [
  ["all_in_context", "All required documents in context"],
  ["all_opened", "All required documents deliberately opened"],
] as const;
type Exposure = (typeof EXPOSURES)[number][0];

const W = 840;
const H = 220;
const PAD = 8;

export function RetrievalPanel({ tasks, rows }: { tasks: Study["retrieval_tasks"]; rows: PooledRow[] }) {
  const [exp, setExp] = useState<Exposure>("all_in_context");
  const data = [...tasks[exp]].sort((a, b) => b.diff - a.diff);
  const higher = data.filter((d) => d.diff > 0).length;
  const lower = data.filter((d) => d.diff < 0).length;
  const sign = rows.find((r) => r.section === `3.3 ${exp}` && r.quantity === "tasks_pass_higher");
  const p = sign?.note?.match(/p=([\d.]+)/)?.[1];
  const mh = (note: string) =>
    rows.find((r) => r.section === `3.3 ${exp}` && r.quantity === "MH_risk_diff_fail_complete_minus_incomplete" && r.note === note);
  const mixed = mh("mixed strata only");
  const all = mh("all strata");

  const step = (W - 2 * PAD) / data.length;
  const y = (v: number) => H / 2 - (v * (H / 2 - PAD));

  return (
    <div className="space-y-4">
      <Segmented value={exp} options={EXPOSURES.map((e) => [...e] as [string, string])} onChange={(v) => setExp(v as Exposure)} />

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${higher} tasks above zero, ${lower} below`}>
        <line x1={0} x2={W} y1={H / 2} y2={H / 2} stroke="var(--color-border-strong)" />
        {data.map((d, i) => (
          <rect
            key={d.task}
            x={PAD + i * step + step * 0.15}
            width={step * 0.7}
            y={Math.min(y(d.diff), H / 2)}
            height={Math.max(1, Math.abs(y(d.diff) - H / 2))}
            fill={d.diff > 0 ? "var(--color-accent)" : d.diff < 0 ? "var(--color-series-2)" : "var(--color-text-muted)"}
          >
            <title>{`${d.task}: ${d.diff > 0 ? "+" : ""}${d.diff.toFixed(3)} (${d.units} model${d.units === 1 ? "" : "s"} with mixed outcomes)`}</title>
          </rect>
        ))}
        <text x={W - 4} y={14} fontSize={11} textAnchor="end" fill="var(--color-text-muted)">
          ↑ Passed runs had the documents more often
        </text>
        <text x={4} y={H - 6} fontSize={11} fill="var(--color-text-muted)">
          ↓ Failed runs had them more often
        </text>
      </svg>

      <div className="grid gap-2 md:grid-cols-3">
        <Box label="Tasks, passed higher / failed higher / tied">
          <span className="num">
            {higher} / {lower} / {data.length - higher - lower}
          </span>
          {p && <span className="text-text-muted"> · sign test p = {Number(p).toFixed(4)}</span>}
        </Box>
        <Box label="Change in failure risk, same model and task">
          <span className="num">{mixed ? fmtDelta(mixed.estimate) : "—"}</span>
          {mixed?.ci_low != null && (
            <span className="num text-text-muted">
              {" "}
              [{fmtDelta(mixed.ci_low)}, {fmtDelta(mixed.ci_high ?? NaN)}]
            </span>
          )}
          <div className="text-[11px] text-text-muted">Mantel–Haenszel, strata with mixed outcomes</div>
        </Box>
        <Box label="Over all strata">
          <span className="num">{all ? fmtDelta(all.estimate) : "—"}</span>
          {all?.ci_low != null && (
            <span className="num text-text-muted">
              {" "}
              [{fmtDelta(all.ci_low)}, {fmtDelta(all.ci_high ?? NaN)}]
            </span>
          )}
          <div className="text-[11px] text-text-muted">Includes strata where all runs pass or all fail</div>
        </Box>
      </div>
    </div>
  );
}

function Box({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-md bg-surface-2 px-3 py-2 text-[12.5px]">
      <div className="label-caps mb-1">{label}</div>
      {children}
    </div>
  );
}
