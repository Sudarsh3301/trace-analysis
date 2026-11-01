"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { Badge } from "@/components/ui/primitives";
import { conversationHref, fetchRuns, type RunRow, type StudyModel } from "@/lib/study";
import { cn, fmtProbability } from "@/lib/ui";

type Cell = { m: number; t: number };

const shade = (frac: number) =>
  `color-mix(in oklab, var(--color-accent) ${Math.round(frac * 100)}%, var(--color-surface-2))`;

export function TaskHeatmap({
  models,
  tasks,
  passes,
  trials,
}: {
  models: StudyModel[];
  tasks: { id: string; pass_rate: number }[];
  passes: number[][];
  trials: number[][];
}) {
  // strongest model on top; tasks arrive hardest first
  const order = useMemo(
    () => models.map((_, i) => i).sort((a, b) => models[b].pass_rate - models[a].pass_rate),
    [models],
  );
  const [hover, setHover] = useState<Cell | null>(null);
  const [selected, setSelected] = useState<Cell | null>(null);
  const [runs, setRuns] = useState<RunRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selected || runs) return;
    fetchRuns().then(setRuns, (e: Error) => setError(e.message));
  }, [selected, runs]);

  const shown = hover ?? selected;

  return (
    <div className="space-y-3">
      <p className="num min-h-[18px] text-[12px] text-text-secondary" aria-live="polite">
        {shown
          ? `${models[shown.m].label} · ${tasks[shown.t].id} · ${passes[shown.m][shown.t]} of ${trials[shown.m][shown.t]} runs passed · all models on this task: ${Math.round(100 * tasks[shown.t].pass_rate)}%`
          : "Hover a cell to read it. Click a cell to see its runs."}
      </p>

      <div className="overflow-x-auto">
        <div
          className="grid min-w-[880px] gap-y-[2px]"
          style={{ gridTemplateColumns: `150px repeat(${tasks.length}, minmax(0, 1fr))` }}
          onMouseLeave={() => setHover(null)}
        >
          {order.map((m) => (
            <Row key={models[m].key}>
              <div className="truncate pr-2 text-right text-[11px] leading-[14px] text-text-secondary">
                {models[m].label}
              </div>
              {tasks.map((task, t) => {
                const n = trials[m][t];
                const isSel = selected?.m === m && selected.t === t;
                return (
                  <button
                    key={task.id}
                    type="button"
                    aria-label={`${models[m].label}, ${task.id}: ${passes[m][t]} of ${n} passed`}
                    aria-pressed={isSel}
                    onMouseEnter={() => setHover({ m, t })}
                    onFocus={() => setHover({ m, t })}
                    onClick={() => setSelected(isSel ? null : { m, t })}
                    className={cn(
                      "h-[14px] border-r border-[var(--color-surface-1)] outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]",
                      isSel && "ring-2 ring-[var(--color-text-primary)]",
                    )}
                    style={{ background: n ? shade(passes[m][t] / n) : "transparent" }}
                  />
                );
              })}
            </Row>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-text-muted">
        <span>Tasks sorted hardest (left) to easiest (right); models by pass rate.</span>
        <span className="flex items-center gap-1">
          Runs passed
          {[0, 1, 2, 3, 4].map((p) => (
            <span key={p} className="flex items-center gap-1">
              <span aria-hidden className="inline-block size-3 rounded-[2px]" style={{ background: shade(p / 4) }} />
              <span className="num">{p}/4</span>
            </span>
          ))}
        </span>
      </div>

      {selected && (
        <CellRuns
          model={models[selected.m]}
          task={tasks[selected.t].id}
          runs={runs?.filter((r) => r.model === selected.m && r.task_id === selected.t) ?? null}
          error={error}
          jevNames={models.filter((x) => x.jev).map((x) => x.label).join(", ")}
        />
      )}
    </div>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return <div className="contents">{children}</div>;
}

function CellRuns({
  model,
  task,
  runs,
  error,
  jevNames,
}: {
  model: StudyModel;
  task: string;
  runs: RunRow[] | null;
  error: string | null;
  jevNames: string;
}) {
  return (
    <div className="panel overflow-hidden">
      <div className="border-b border-border px-3.5 py-2.5 text-[12.5px] font-medium">
        {model.label} on {task}
        {!model.jev && (
          <span className="ml-2 text-[11px] font-normal text-text-muted">
            Jev judgments exist only for {jevNames}.
          </span>
        )}
      </div>
      {error ? (
        <p className="px-3.5 py-3 text-[12px] text-[var(--color-status-critical)]">Could not load runs: {error}</p>
      ) : !runs ? (
        <p className="px-3.5 py-3 text-[12px] text-text-muted">Loading runs…</p>
      ) : (
        <table className="w-full text-left text-[12px]">
          <thead className="label-caps">
            <tr className="[&>th]:px-3.5 [&>th]:py-2 [&>th]:font-semibold">
              <th>Trial</th>
              <th>Outcome</th>
              <th>Tool calls</th>
              <th>Required docs in context</th>
              {model.jev && <th>Max pushback before decisive point</th>}
              {model.jev && <th>Max stuck</th>}
            </tr>
          </thead>
          <tbody>
            {runs
              .sort((a, b) => Number(a.trial) - Number(b.trial))
              .map((r) => {
                const failed = r.reward === 0;
                const silent = model.jev && failed && Number(r.pushback_max_before) < 0.5;
                return (
                  <tr key={String(r.run_id)} className="border-t border-border [&>td]:px-3.5 [&>td]:py-2">
                    <td className="num">
                      <Link
                        href={conversationHref(String(r.run_id))}
                        className="text-[var(--color-accent)] underline-offset-2 hover:underline"
                      >
                        Trial {Number(r.trial) + 1} →
                      </Link>
                    </td>
                    <td className="space-x-1">
                      <Badge tone={failed ? "critical" : "good"}>{failed ? "failed" : "passed"}</Badge>
                      {silent && <Badge tone="warning">silent failure</Badge>}
                    </td>
                    <td className="num">{String(r.n_tool_calls)}</td>
                    <td className="num">
                      {String(r.required_in_context)} / {String(r.required_docs_total)}
                    </td>
                    {model.jev && <td className="num">{fmtProbability(Number(r.pushback_max_before))}</td>}
                    {model.jev && (
                      <td className="num">{r.stuck_max === null ? "—" : fmtProbability(Number(r.stuck_max))}</td>
                    )}
                  </tr>
                );
              })}
          </tbody>
        </table>
      )}
    </div>
  );
}
