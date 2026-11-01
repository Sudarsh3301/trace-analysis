import Link from "next/link";
import type { ReactNode } from "react";
import { ArrowRight } from "lucide-react";

import { AurocPanel } from "@/components/study/auroc-panel";
import { Note, Page, Section } from "@/components/ui/page";
import { Stat } from "@/components/ui/primitives";
import { auroc, DATASET_URL, pooled, REPO_URL, study } from "@/lib/study";
import { fmtDelta } from "@/lib/ui";

export default function Overview() {
  const { models, tasks, trials, tables, prior_check: pc } = study;
  const prior = auroc(tables.auroc_prior, "all", "C");
  const addFacts = auroc(tables.auroc_prior, "all", "delta C+F - C");
  const jevAlone = auroc(tables.auroc_jev, "2", "J");
  const factsAlone = auroc(tables.auroc_jev, "2", "F");
  const mh = pooled("3.3 all_in_context", "MH_risk_diff_fail_complete_minus_incomplete", "mixed strata only");
  const silent = tables.silent.find((r) => r.threshold === 0.5 && r.model === "ALL" && r.group === "failed");
  const conversations = trials.flat().reduce((a, b) => a + b, 0);

  return (
    <Page
      title="What agent traces do and don't tell you"
      description="If you know which task an agent faced, you can predict whether it will fail better than with anything you can read in its trace."
    >
      <p className="text-[12px] text-text-secondary">
        A study by Sudarsh, AI/ML engineer. Built solo on public τ-bench data. ·{" "}
        <a href={REPO_URL} className="text-[var(--color-accent)] hover:underline">
          Code
        </a>{" "}
        ·{" "}
        <a href={DATASET_URL} className="text-[var(--color-accent)] hover:underline">
          Data
        </a>
      </p>

      <div className="panel border-l-4 border-l-[var(--color-accent)] px-4 py-3">
        <div className="label-caps">For practitioners</div>
        <p className="mt-1 text-[13px] leading-relaxed">
          If you build or evaluate failure predictors for agents, report a task-difficulty baseline first. On this
          benchmark, no trace signal beats it: not tool and retrieval counts, not LLM-judge step judgments, not user
          pushback.
        </p>
      </div>

      <section className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <Stat label="Models" value={models.length} />
        <Stat label="Conversations" value={conversations.toLocaleString("en-US")} />
        <Stat label="Tasks" value={tasks.length} hint="τ-bench banking support" />
        <Stat label="Trials" value="4" hint="per model and task; labels re-scored" />
      </section>

      <Section
        title="Does the trace beat the task prior?"
        description="AUROC for predicting failure. The task prior is how often the other models failed the same task. Adding signals from the trace does not raise it."
      >
        <AurocPanel models={models} prior={tables.auroc_prior} jev={tables.auroc_jev} modelSelect={false} />
        <div className="mt-4 grid gap-3 border-t border-border pt-3 md:grid-cols-2">
          <div className="space-y-1">
            <div className="label-caps">Where the prior applies</div>
            <p className="text-[12px] leading-relaxed text-text-secondary">
              The prior needs results from other runs on the same task, so it exists in eval and regression suites,
              where tasks repeat. For a new task in production there is no prior. There, the trace alone is what you
              have: trace facts give AUROC{" "}
              <b className="num">{factsAlone?.auroc.toFixed(3)}</b> and Jev step judgments{" "}
              <b className="num">{jevAlone?.auroc.toFixed(3)}</b> (first 2 turns, 4 models). That is weak.
            </p>
          </div>
          <div className="space-y-1">
            <div className="label-caps">Not driven by impossible tasks</div>
            <p className="text-[12px] leading-relaxed text-text-secondary">
              {pc.always_fail_tasks} tasks fail in every run of every model. Without them, the prior still reaches
              AUROC <b className="num">{pc.auroc_without_always_fail.toFixed(3)}</b> (
              {pc.n_without.toLocaleString("en-US")} runs), against{" "}
              <span className="num">{pc.auroc_all.toFixed(3)}</span> with them, pooled over all runs.
            </p>
          </div>
        </div>
      </Section>

      <section className="grid gap-2 lg:grid-cols-3">
        <Finding href="/task-prior" title="The task comes first">
          Other models&apos; failure rate on the task: AUROC <b className="num">{prior?.auroc.toFixed(3)}</b>. Adding
          trace facts: <span className="num">{addFacts ? fmtDelta(addFacts.auroc) : "—"}</span>.
        </Finding>
        <Finding href="/retrieval" title="Within a task, retrieval matters">
          When all required documents reached the context, failure risk changes by{" "}
          <b className="num">{mh ? fmtDelta(mh.estimate) : "—"}</b> within the same model and task. An association, not a
          cause.
        </Finding>
        <Finding href="/pushback" title="Pushback is not an early warning">
          <b className="num">{silent ? Math.round(100 * silent.share_no_pushback_before_decisive) : "—"}%</b> of failed
          conversations show no user pushback before the decisive action. Per message, failed runs show less.
        </Finding>
      </section>

      <div className="panel px-4 py-3">
        <div className="label-caps">Limits</div>
        <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[12px] leading-relaxed text-text-secondary">
          <li>The customer is an LLM simulator that follows a task script, not a real person.</li>
          <li>One benchmark and one domain (banking support); prediction uses logistic regression only.</li>
          <li>The pushback judge was validated on 60 messages labelled by another LLM, not by humans.</li>
        </ul>
        <Link href="/method" className="mt-2 inline-flex items-center gap-1 text-[12px] text-[var(--color-accent)] hover:underline">
          Method and all limits <ArrowRight size={12} />
        </Link>
      </div>

      <Note>
        Data: τ-bench (tau2-bench) banking_knowledge trajectories, Sierra, MIT. Labels re-scored with tau2-bench v1.0.1.
        Judgments: TypeSafe Jev 1.14.
      </Note>
    </Page>
  );
}

function Finding({ href, title, children }: { href: string; title: string; children: ReactNode }) {
  return (
    <Link href={href} className="panel group block space-y-1.5 px-3.5 py-3 transition-colors hover:border-border-strong">
      <div className="flex items-center justify-between">
        <p className="text-[13px] leading-snug font-medium">{title}</p>
        <ArrowRight size={14} className="text-text-muted transition-transform group-hover:translate-x-0.5" />
      </div>
      <p className="text-[12px] leading-relaxed text-text-secondary">{children}</p>
    </Link>
  );
}
