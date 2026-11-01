"use client";

import { useState } from "react";

import type { SilentRow, WindowRow } from "@/lib/study";
import { fmtDelta } from "@/lib/ui";

import { Segmented } from "./auroc-panel";

const THRESHOLDS = [
  ["0.5", "Threshold 0.5 · set in advance"],
  ["0.8", "Threshold 0.8 · exploratory, chosen after the fact"],
] as const;

type Pair = { label: string; passed: number; failed: number; lo?: [number, number]; hi?: [number, number] };

export function PushbackPanel({ silent, window }: { silent: SilentRow[]; window: WindowRow[] }) {
  const [t, setT] = useState("0.5");
  const th = Number(t);
  const g = (group: string) => silent.find((r) => r.threshold === th && r.model === "ALL" && r.group === group);
  const passed = g("passed");
  const failed = g("failed");
  const perMsg = window.find((r) => r.threshold === th && r.measure.startsWith("pushback per user message"));
  if (!passed || !failed || !perMsg) return null;

  const pairs: Pair[] = [
    {
      label: "No pushback before the decisive action",
      passed: passed.share_no_pushback_before_decisive,
      failed: failed.share_no_pushback_before_decisive,
      lo: [passed.ci_low_before_decisive, failed.ci_low_before_decisive],
      hi: [passed.ci_high_before_decisive, failed.ci_high_before_decisive],
    },
    {
      label: "No pushback in the whole conversation",
      passed: passed.share_no_pushback_whole_run,
      failed: failed.share_no_pushback_whole_run,
      lo: [passed.ci_low_whole_run, failed.ci_low_whole_run],
      hi: [passed.ci_high_whole_run, failed.ci_high_whole_run],
    },
    { label: "Pushback per user message, before the decisive action", passed: perMsg.passed, failed: perMsg.failed },
  ];

  return (
    <div className="space-y-4">
      <Segmented value={t} options={THRESHOLDS.map((x) => [...x] as [string, string])} onChange={setT} />
      <div className="grid gap-4 md:grid-cols-3">
        {pairs.map((p) => (
          <div key={p.label} className="space-y-2">
            <div className="text-[12px] text-text-secondary">{p.label}</div>
            {(["passed", "failed"] as const).map((side, i) => (
              <div key={side} className="grid grid-cols-[52px_1fr_44px] items-center gap-2">
                <span className="text-[11px] text-text-muted">{side === "passed" ? `Passed` : `Failed`}</span>
                <div className="relative h-3.5 rounded-sm bg-surface-2">
                  <div
                    className="absolute inset-y-0 left-0 rounded-sm"
                    style={{
                      width: `${100 * p[side]}%`,
                      background: side === "failed" ? "var(--color-status-critical)" : "var(--color-status-good)",
                      opacity: 0.75,
                    }}
                  />
                  {p.lo && p.hi && (
                    <div
                      aria-hidden
                      className="absolute top-1/2 h-px bg-text-primary"
                      style={{ left: `${100 * p.lo[i]}%`, width: `${100 * (p.hi[i] - p.lo[i])}%` }}
                    />
                  )}
                </div>
                <span className="num text-right text-[12px]">{p[side].toFixed(2)}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
      <p className="num text-[12px] text-text-secondary">
        Pushback per message, failed minus passed: {fmtDelta(perMsg.failed_minus_passed)}{" "}
        <span className="text-text-muted">
          [{fmtDelta(perMsg.ci_low)}, {fmtDelta(perMsg.ci_high)}]
        </span>
        . Failed conversations do not show more pushback before the mistake.
      </p>
      <p className="text-[11px] leading-relaxed text-text-muted">
        {passed.n_runs + failed.n_runs} conversations of 4 models. Jev checked every natural user message for a
        correction, a repeated request, a complaint or a request for a human. The decisive action is the first extra
        or wrong write action, by the agent or by the customer through a customer tool, or the end of the run if
        no such write happened.
      </p>
    </div>
  );
}
