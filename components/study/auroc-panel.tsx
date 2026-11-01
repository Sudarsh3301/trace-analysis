"use client";

import { useState } from "react";

import { kOf, MEAN, type AurocRow, type StudyModel } from "@/lib/study";
import { cn, fmtDelta } from "@/lib/ui";

const LABELS: Record<string, string> = {
  C: "Task prior: other models' failure rate on the task",
  "C+P": "Task prior + this model's other trials",
  "C+F": "Task prior + trace facts",
  "C+P+F": "Task prior + own trials + trace facts",
  "C+J": "Task prior + Jev step judgments",
  "C+J2": "Task prior + Jev, 2 features (max stuck, max premature)",
  "C+F+J": "Task prior + trace facts + Jev",
  J: "Jev step judgments alone",
  F: "Trace facts alone",
};
const DELTA_LABELS: Record<string, string> = {
  "C+F - C": "Adding trace facts to the task prior",
  "C+P+F - C+P": "Adding trace facts to the prior and own trials",
  "C+J - C": "Adding Jev step judgments to the task prior",
  "C+F+J - C+F": "Adding Jev on top of trace facts",
  "C+J2 - C": "Adding Jev, 2 features, to the task prior",
};
const SOURCES = {
  facts: { label: "Trace facts · 20 models", sets: ["C", "C+F", "C+P", "C+P+F"], ks: ["2", "all"] },
  jev: { label: "Jev judgments · 4 models", sets: ["C", "C+J", "C+J2", "C+F", "C+F+J", "J", "F"], ks: ["2", "4"] },
} as const;
const K_LABEL: Record<string, string> = { "2": "First 2 user turns", "4": "First 4 user turns", all: "All turns" };
const LO = 0.5; // AUROC axis starts at chance

export function AurocPanel({
  models,
  prior,
  jev,
  modelSelect = true,
}: {
  models: StudyModel[];
  prior: AurocRow[];
  jev: AurocRow[];
  /** Off on the Overview: the headline shows the mean over models only. */
  modelSelect?: boolean;
}) {
  const [source, setSource] = useState<keyof typeof SOURCES>("facts");
  const [k, setK] = useState<string>("2");
  const [model, setModel] = useState(MEAN);
  const src = SOURCES[source];
  const table = source === "facts" ? prior : jev;
  const choices = models.filter((m) => source === "facts" || m.jev);

  const pick = (s: string) =>
    table.find((r) => r.model === model && kOf(r) === k && r.feature_set === s && r.section !== "delta");
  // The headline (no model selector) shows trace signals only; the own-trials prior is a second task-level prior.
  const shown = (s: string) => modelSelect || !s.includes("P");
  const rows = src.sets
    .filter(shown)
    .map((s) => ({ s, r: pick(s) }))
    .filter((x) => x.r) as { s: string; r: AurocRow }[];
  const base = rows.find((x) => x.s === "C")?.r.auroc;
  const deltas = table.filter(
    (r) =>
      r.model === model &&
      kOf(r) === k &&
      (r.section === "delta" || r.feature_set.startsWith("delta")) &&
      shown(r.feature_set.replace(/^delta /, "")),
  );
  const x = (v: number) => `${(100 * Math.max(0, v - LO)) / (1 - LO)}%`;

  const switchSource = (s: keyof typeof SOURCES) => {
    setSource(s);
    setK("2");
    if (s === "jev" && model !== MEAN && !models.find((m) => m.key === model)?.jev) setModel(MEAN);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Segmented
          value={source}
          options={Object.entries(SOURCES).map(([v, o]) => [v, o.label])}
          onChange={(v) => switchSource(v as keyof typeof SOURCES)}
        />
        <Segmented value={k} options={src.ks.map((v) => [v, K_LABEL[v]])} onChange={setK} />
        {modelSelect ? (
        <label className="flex items-center gap-2 text-[12px] text-text-secondary">
          Model
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="rounded-md border border-border bg-surface-2 px-2 py-1 text-[12px] text-text-primary"
          >
            <option value={MEAN}>Mean of {choices.length} models</option>
            {choices.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        ) : (
          <span className="text-[12px] text-text-muted">Mean of {choices.length} models</span>
        )}
      </div>

      <div className="space-y-2.5">
        {rows.map(({ s, r }) => (
          <div key={s} className="grid grid-cols-[minmax(0,300px)_1fr_120px] items-center gap-3">
            <span className={cn("text-[12px]", s === "C" ? "font-medium text-text-primary" : "text-text-secondary")}>
              {LABELS[s]}
            </span>
            <div className="relative h-4 rounded-sm bg-surface-2">
              <div
                className="absolute inset-y-0 left-0 rounded-sm"
                style={{
                  width: x(r.auroc),
                  background: s === "C" ? "var(--color-accent)" : "var(--color-border-strong)",
                }}
              />
              <div
                aria-hidden
                className="absolute top-1/2 h-px bg-text-primary"
                style={{ left: x(r.ci_low), width: `calc(${x(r.ci_high)} - ${x(r.ci_low)})` }}
              />
              {base !== undefined && (
                <div
                  aria-hidden
                  className="absolute -inset-y-1 border-l border-dashed border-[var(--color-accent)]"
                  style={{ left: x(base) }}
                />
              )}
            </div>
            <span className="num text-[12px] text-text-secondary">
              {r.auroc.toFixed(3)} <span className="text-text-muted">[{r.ci_low.toFixed(2)}, {r.ci_high.toFixed(2)}]</span>
            </span>
          </div>
        ))}
        <div className="grid grid-cols-[minmax(0,300px)_1fr_120px] gap-3 text-[10.5px] text-text-muted">
          <span />
          <span className="num flex justify-between">
            <span>0.5 chance</span>
            <span>1.0</span>
          </span>
          <span />
        </div>
      </div>

      {deltas.length > 0 && (
        <ul className="space-y-1 border-t border-border pt-3 text-[12px] text-text-secondary">
          {deltas.map((d) => {
            const name = d.feature_set.replace(/^delta /, "");
            const crosses = d.ci_low <= 0 && d.ci_high >= 0;
            return (
              <li key={name}>
                {DELTA_LABELS[name] ?? name}:{" "}
                <span className={cn("num", d.auroc < 0 ? "text-[var(--color-status-critical)]" : "text-text-primary")}>
                  {fmtDelta(d.auroc)}
                </span>{" "}
                <span className="num text-text-muted">
                  [{fmtDelta(d.ci_low)}, {fmtDelta(d.ci_high)}]{crosses ? " · interval includes 0" : ""}
                </span>
              </li>
            );
          })}
        </ul>
      )}
      <p className="text-[11px] leading-relaxed text-text-muted">
        AUROC for predicting failure, logistic regression, cross-validated by task. Lines are 95% intervals from a
        bootstrap over tasks. The dashed line marks the task prior.
      </p>
    </div>
  );
}

export function Segmented({
  value,
  options,
  onChange,
}: {
  value: string;
  options: (readonly [string, string])[];
  onChange: (v: string) => void;
}) {
  return (
    <div role="radiogroup" className="inline-flex rounded-md border border-border bg-surface-2 p-0.5">
      {options.map(([v, label]) => (
        <button
          key={v}
          type="button"
          role="radio"
          aria-checked={value === v}
          onClick={() => onChange(v)}
          className={cn(
            "rounded px-2 py-1 text-[11.5px] transition-colors",
            value === v ? "bg-surface-1 text-text-primary shadow-sm" : "text-text-secondary hover:text-text-primary",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
