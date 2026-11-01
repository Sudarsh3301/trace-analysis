"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { Badge } from "@/components/ui/primitives";
import { conversationHref, fetchRuns, silentFailures, type RunRow, type StudyModel } from "@/lib/study";
import { fmtProbability } from "@/lib/ui";

const KINDS: Record<string, string> = {
  wrong_write: "wrong write",
  unexpected_write: "unexpected write",
  end_of_run: "expected write missing",
};
const PAGE = 20;

export function SilentFailures({ models, tasks }: { models: StudyModel[]; tasks: string[] }) {
  const [runs, setRuns] = useState<RunRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState("all");
  const [kind, setKind] = useState("all");
  const [shown, setShown] = useState(PAGE);

  useEffect(() => {
    fetchRuns().then(setRuns, (e: Error) => setError(e.message));
  }, []);

  const failed = useMemo(() => runs?.filter((r) => r.reward === 0) ?? [], [runs]);
  const silent = useMemo(() => (runs ? silentFailures(runs) : []), [runs]);
  const list = silent.filter(
    (r) => (model === "all" || String(r.model) === model) && (kind === "all" || r.decisive_kind === kind),
  );

  if (error) return <p className="text-[12px] text-[var(--color-status-critical)]">Could not load runs: {error}</p>;
  if (!runs) return <p className="text-[12px] text-text-muted">Loading runs…</p>;

  // runs with no messages have no decisive point and no Jev checks: they are left out, as in the analysis
  const withActor = failed.filter((r) => r.decisive_actor !== null);
  const byCustomer = withActor.filter((r) => r.decisive_actor === "customer").length;
  const jevIdx = models.flatMap((m, i) => (m.jev ? [i] : []));
  const jevFailed = failed.filter((r) => r.pushback_max_before !== null);

  return (
    <div className="space-y-4">
      <div className="grid gap-2 md:grid-cols-2">
        <div className="rounded-md bg-surface-2 px-3 py-2 text-[12.5px] leading-relaxed">
          <div className="label-caps mb-1">Silent failures</div>
          <b className="num">{silent.length}</b> of <span className="num">{jevFailed.length}</span> failed conversations
          of the 4 judged models show no pushback of 0.5 or more before the decisive action.
        </div>
        <div className="rounded-md bg-surface-2 px-3 py-2 text-[12.5px] leading-relaxed">
          <div className="label-caps mb-1">Decided by the customer</div>
          In <b className="num">{byCustomer}</b> of <span className="num">{withActor.length.toLocaleString("en-US")}</span>{" "}
          failed conversations of all 20 models ({Math.round((100 * byCustomer) / withActor.length)}%), the decisive write is a customer
          action through a customer tool. This is a property of the benchmark: the simulated customer has its own tools,
          and no agent trace can show their mistake in advance.
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 text-[12px] text-text-secondary">
        <label className="flex items-center gap-2">
          Model
          <select value={model} onChange={(e) => (setModel(e.target.value), setShown(PAGE))} className="rounded-md border border-border bg-surface-2 px-2 py-1 text-text-primary">
            <option value="all">All 4 judged models</option>
            {jevIdx.map((i) => (
              <option key={i} value={String(i)}>
                {models[i].label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          Decisive point
          <select value={kind} onChange={(e) => (setKind(e.target.value), setShown(PAGE))} className="rounded-md border border-border bg-surface-2 px-2 py-1 text-text-primary">
            <option value="all">Any</option>
            {Object.entries(KINDS).map(([k, l]) => (
              <option key={k} value={k}>
                {l}
              </option>
            ))}
          </select>
        </label>
        <span className="num text-text-muted">{list.length} conversations</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[620px] text-left text-[12px]">
          <thead className="label-caps">
            <tr className="[&>th]:px-2 [&>th]:py-2 [&>th]:font-semibold">
              <th>Model</th>
              <th>Task</th>
              <th>Decisive point</th>
              <th>Max pushback before it</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {list.slice(0, shown).map((r) => (
              <tr key={String(r.run_id)} className="border-t border-border [&>td]:px-2 [&>td]:py-1.5">
                <td>{models[Number(r.model)].label}</td>
                <td className="num">
                  {tasks[Number(r.task_id)]} · trial {Number(r.trial) + 1}
                </td>
                <td className="space-x-1">
                  <span>{KINDS[String(r.decisive_kind)] ?? "—"}</span>
                  {r.decisive_actor === "customer" && <Badge>customer</Badge>}
                </td>
                <td className="num">{fmtProbability(Number(r.pushback_max_before))}</td>
                <td className="text-right">
                  <Link href={conversationHref(String(r.run_id))} className="text-[var(--color-accent)] hover:underline">
                    Open →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {shown < list.length && (
        <button
          type="button"
          onClick={() => setShown(shown + PAGE)}
          className="rounded-md border border-border bg-surface-2 px-2.5 py-1 text-[12px] hover:bg-surface-3"
        >
          Show {Math.min(PAGE, list.length - shown)} more
        </button>
      )}
    </div>
  );
}
