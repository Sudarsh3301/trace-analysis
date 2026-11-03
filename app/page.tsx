import Link from "next/link";
import type { ReactNode } from "react";
import { ArrowRight } from "lucide-react";

import { Page } from "@/components/ui/page";
import { auroc, DATASET_URL, pooled, REPO_URL, study } from "@/lib/study";

const LO = 0.5; // scores start at a coin flip

export default function Overview() {
  const { models, tasks, trials, tables } = study;
  const conversations = trials.flat().reduce((a, b) => a + b, 0);
  const s = (set: string) => auroc(tables.auroc_jev, "2", set)?.auroc ?? 0;
  const board = [
    { label: "How often other models fail this task", value: s("C"), lead: true },
    { label: "Task + trace + LLM judge, combined", value: s("C+F+J") },
    { label: "Trace: tool calls, documents fetched", value: s("F") },
    { label: "LLM judge reading each step", value: s("J") },
    { label: "Coin flip", value: 0.5 },
  ];
  const signs = pooled("3.3 all_in_context", "tasks_pass_higher");
  const [, failHigher, ties] = signs?.note?.match(/fail_higher=(\d+), ties=(\d+)/)?.map(Number) ?? [];
  const passHigher = signs?.estimate ?? 0;
  const silent = tables.silent.find((r) => r.threshold === 0.5 && r.model === "ALL" && r.group === "failed");
  const silentPct = Math.round(100 * (silent?.share_no_pushback_before_decisive ?? 0));
  const alwaysFail = tasks.filter((t) => t.pass_rate === 0).length;

  return (
    <Page
      title="Knowing the task predicts agent failure better than reading the trace"
      description={`I tested the signals agent-monitoring tools read (tool calls, documents fetched, an LLM judging each step, user complaints) against ${conversations.toLocaleString("en-US")} real outcomes from ${models.length} AI models. One simple number about the task beat all of them.`}
    >
      <p className="text-[12px] text-text-secondary">
        A study by Sudarsh, AI/ML engineer · {models.length} models · {conversations.toLocaleString("en-US")} customer-support
        conversations · {tasks.length} tasks ·{" "}
        <a href={REPO_URL} className="text-[var(--color-accent)] hover:underline">
          Code
        </a>{" "}
        ·{" "}
        <a href={DATASET_URL} className="text-[var(--color-accent)] hover:underline">
          Data
        </a>
      </p>

      <section className="panel px-4 py-4 md:px-5">
        <h2 className="text-[16px] font-semibold tracking-tight">Which signal best predicts a failed conversation?</h2>
        <p className="mt-0.5 text-[12px] text-text-secondary">Prediction score: 0.5 is a coin flip, 1.0 is perfect.</p>
        <div className="mt-4 space-y-3">
          {board.map((b) => (
            <div key={b.label} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 md:grid-cols-[260px_1fr_48px]">
              <span className={b.lead ? "text-[13px] font-semibold" : "text-[13px] text-text-secondary"}>{b.label}</span>
              <span className={`num text-right text-[14px] md:order-last ${b.lead ? "font-semibold" : "text-text-secondary"}`}>
                {b.value.toFixed(2)}
              </span>
              <div className="col-span-2 h-5 rounded bg-surface-2 md:col-span-1">
                <div
                  className="h-full rounded"
                  style={{
                    width: `${(100 * (b.value - LO)) / (1 - LO)}%`,
                    background: b.lead ? "var(--color-accent)" : "var(--color-border-strong)",
                  }}
                />
              </div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-[13.5px] font-medium">
          Adding the trace to the task&apos;s failure rate did not help. It made the prediction slightly worse.
        </p>
        <p className="mt-1 text-[11px] text-text-muted">4 models where every signal was run, first 2 user turns.</p>
      </section>

      <section className="grid gap-3 lg:grid-cols-3">
        <Insight href="/task-prior" title="Some tasks are just hard">
          <div className="grid grid-cols-[repeat(14,minmax(0,1fr))] gap-[3px]" aria-hidden>
            {tasks.map((t) => (
              <div
                key={t.id}
                className="aspect-square rounded-[2px]"
                style={{ background: `color-mix(in srgb, var(--color-accent) ${Math.round(100 * (1 - t.pass_rate))}%, var(--color-surface-2))` }}
              />
            ))}
          </div>
          <p>
            Each square is a task; the stronger the blue, the more models fail it. <b>{alwaysFail}</b> tasks beat every model, every
            time.
          </p>
        </Insight>

        <Insight href="/retrieval" title="The right documents matter">
          <div className="flex h-7 overflow-hidden rounded text-[11px] font-medium text-white" aria-hidden>
            <div className="flex items-center justify-center" style={{ flex: passHigher, background: "var(--color-status-good)" }}>
              {passHigher}
            </div>
            <div className="flex items-center justify-center text-text-secondary" style={{ flex: ties, background: "var(--color-surface-3)" }}>
              {ties}
            </div>
            <div className="flex items-center justify-center" style={{ flex: failHigher, background: "var(--color-status-critical)" }}>
              {failHigher}
            </div>
          </div>
          <p>
            Same task, same model: in <b>{passHigher}</b> tasks the successful runs had found the right documents more
            often. In <b>{failHigher}</b>, the failed runs had; in {ties}, no difference.
          </p>
        </Insight>

        <Insight href="/pushback" title="Users rarely warn you">
          <div className="grid max-w-[200px] grid-cols-10 gap-[3px]" aria-hidden>
            {Array.from({ length: 100 }, (_, i) => (
              <div
                key={i}
                className="aspect-square rounded-full"
                style={{ background: i < silentPct ? "var(--color-status-critical)" : "var(--color-surface-3)" }}
              />
            ))}
          </div>
          <p>
            In <b>{silentPct} of 100</b> failed conversations, the user never complained before the agent made its
            mistake.
          </p>
        </Insight>
      </section>

      <div className="panel border-l-4 border-l-[var(--color-accent)] px-4 py-3">
        <div className="label-caps">What this means</div>
        <p className="mt-1 text-[14px] leading-relaxed">
          Before you trust an agent-failure detector, check it against how hard the task is. Most of what it sees may be
          task difficulty.
        </p>
      </div>

      <p className="text-[11.5px] text-text-muted">
        Public τ&#8209;bench banking-support data ·{" "}
        <Link href="/method" className="text-[var(--color-accent)] hover:underline">
          How it was tested&nbsp;→
        </Link>
      </p>
    </Page>
  );
}

function Insight({ href, title, children }: { href: string; title: string; children: ReactNode }) {
  return (
    <Link href={href} className="panel group flex flex-col gap-3 px-4 py-3.5 transition-colors hover:border-border-strong">
      <div className="flex items-center justify-between">
        <h3 className="text-[14px] font-semibold tracking-tight">{title}</h3>
        <ArrowRight size={14} className="text-text-muted transition-transform group-hover:translate-x-0.5" />
      </div>
      <div className="flex flex-1 flex-col gap-3 text-[12.5px] leading-relaxed text-text-secondary">{children}</div>
    </Link>
  );
}
