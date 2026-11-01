import type { ReactNode } from "react";

import { Page, Section } from "@/components/ui/page";
import { DATASET_URL, REPO_URL, study } from "@/lib/study";

const DIMENSIONS: Record<string, string> = {
  taskAlignment: "Task alignment",
  progress: "Progress",
  repetition: "Repetition",
  stuck: "Stuck",
  needsVerification: "Needs verification",
  prematureCompletion: "Premature completion",
  unexpectedDirection: "Unexpected direction",
  phase: "Phase (choice)",
};

export default function MethodPage() {
  const { models, tasks, trials, tables } = study;
  const flips = tables.auroc_jev.filter((r) => r.section === "flip_rate");
  const usage = tables.jev_usage.find((r) => r.part === "total");
  const conversations = trials.flat().reduce((a, b) => a + b, 0);

  return (
    <Page title="Method and limits" description="What was measured, how, and where the result stops.">
      <Section title="Data">
        <Prose>
          <p>
            τ-bench (tau2-bench) public leaderboard trajectories for the <code>banking_knowledge</code> domain: {tasks.length}{" "}
            tasks, about 700 policy documents, an LLM-simulated customer (gpt-5.2), and grading by the final database state.
            {" "}{models.length} models × {tasks.length} tasks × 4 trials = {conversations.toLocaleString("en-US")} conversations.
          </p>
          <p>
            <b>Labels re-scored.</b> Several older trajectory files carried rewards from before tau2-bench v1.0.1 fixed
            banking grading. Every file was re-scored with the v1.0.1 grading code; re-scored pass rates match the
            leaderboard within 0.3 points. 6 files were excluded: 4 that v1.0.1 cannot load and 2 that stay off the
            leaderboard after re-scoring. The pooled results repeat on the 12 files whose original labels already matched.
          </p>
          <p className="text-text-muted">Models: {models.map((m) => m.label).join(", ")}.</p>
        </Prose>
      </Section>

      <Section title="Signals tested">
        <Prose>
          <p>
            <b>Trace facts</b> (code, no model): tool counts, search and write calls, required documents surfaced, in
            context and opened, repeated identical calls, transfers to a human, message lengths. Computed on the first k
            user turns (k = 2, 4, 6, all), using only messages up to turn k.
          </p>
          <p>
            <b>Step judgments</b> (TypeSafe Jev 1.14, 4 models): 8 questions on each of the first 6 agent steps. The task
            given to Jev is the customer&apos;s first message; no ground truth is sent.
          </p>
          <p>
            <b>Pushback checks</b> (Jev 1.14, 4 models): one request per natural user message, with the previous agent and
            user messages. Pushback is the highest of <i>corrects</i>, <i>repeats</i>, <i>complains</i> and{" "}
            <i>asks for a human</i>.
          </p>
          <p>
            <b>Decisive point:</b> the first extra or wrong write action (by the agent, or by the customer through a
            customer tool), or the end of the run if no such write happened.
          </p>
          {usage && (
            <p className="text-text-muted">
              Jev usage: {usage.calls.toLocaleString("en-US")} calls, ${usage.cost_usd.toFixed(2)} in total.
            </p>
          )}
        </Prose>
      </Section>

      <Section title="Prediction and uncertainty">
        <Prose>
          <p>
            Logistic regression with fixed C = 1 and no tuning. Cross-validation is grouped by task, so no task is in both
            the training and the test fold. The task prior for a run is that task&apos;s pass rate among the other models.
            Adding weak features to an untuned model can lower AUROC slightly; that is why some changes are negative.
          </p>
          <p>95% intervals come from 2,000 bootstrap resamples of tasks (seed 0).</p>
        </Prose>
      </Section>

      {flips.length > 0 && (
        <Section
          title="Repeatability of the judge"
          description="The same 20 step payloads were sent 3 times. Share of steps whose answer changed between the identical calls:"
        >
          <div className="grid grid-cols-2 gap-1.5 text-[12px] md:grid-cols-4">
            {flips.map((r) => (
              <div key={r.feature_set} className="flex justify-between rounded-md bg-surface-2 px-2.5 py-1.5">
                <span className="text-text-secondary">{DIMENSIONS[r.feature_set] ?? r.feature_set}</span>
                <span className="num">{r.auroc.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </Section>
      )}

      <Section title="Limits">
        <Prose>
          <ul className="list-disc space-y-1 pl-4">
            <li>
              <b>Simulated users.</b> The customer is an LLM that follows the task script; real customers differ, and the
              simulator cannot know bank policy, so policy mistakes can go unnoticed by design.
            </li>
            <li>
              <b>The task prior needs repeated tasks.</b> It exists in eval and regression suites, not for a new task in
              production.
            </li>
            <li>
              <b>One domain</b> (banking support) and one benchmark.
            </li>
            <li>
              <b>The retrieval result is an association</b>, not a cause.
            </li>
            <li>
              <b>Validation labels come from an LLM</b>, on 60 stratified messages; the 0.8 threshold was chosen on that
              same sample.
            </li>
            <li>The &quot;opened&quot; document measure is loose: any later mention of a document id counts.</li>
            <li>Prediction uses logistic regression only; a stronger model might extract more from the trace.</li>
          </ul>
        </Prose>
      </Section>

      <Section title="Related work">
        <Prose>
          <ul className="list-disc space-y-1 pl-4">
            <li>
              <b>Failure attribution</b> in the sense of{" "}
              <a href="https://arxiv.org/abs/2505.00212">Who&amp;When (ICML 2025)</a> finds which agent and which step caused
              a failure. This study asks a different question: whether trace signals predict failure at all, against a
              task-difficulty baseline.
            </li>
            <li>
              <a href="https://arxiv.org/abs/2609.02783">EarlyEval</a> evaluates failure predictors on new agents but seen
              tasks, without a per-task baseline. On τ-bench banking that baseline alone reaches AUROC 0.86.
            </li>
            <li>
              <a href="https://arxiv.org/abs/2608.09153">TRACE</a> detects user dissatisfaction and traces it back to a
              cause, evaluated on 75 synthetic traces. Here, pushback flags successful runs at least as often as failing
              ones before the mistake.
            </li>
            <li>
              <a href="https://arxiv.org/abs/2603.04370">τ-Knowledge</a> (the benchmark) and{" "}
              <a href="https://arxiv.org/abs/2601.17087">Lost in Simulation</a> (limits of simulated users).
            </li>
          </ul>
        </Prose>
      </Section>

      <Section title="Reproduce">
        <Prose>
          <p>
            Every number on this site is exported from the analysis tables by <code>python -m analysis.export_site</code>.
            The full pipeline, from downloading trajectories to re-scoring and the Jev calls (cached by payload hash), is
            in the <a href={REPO_URL}>repository README</a>. The conversations with Jev annotations are a{" "}
            <a href={DATASET_URL}>Hugging Face dataset</a>.
          </p>
        </Prose>
      </Section>
    </Page>
  );
}

function Prose({ children }: { children: ReactNode }) {
  return (
    <div className="space-y-2 text-[12.5px] leading-relaxed text-text-secondary [&_a]:text-[var(--color-accent)] [&_a:hover]:underline [&_b]:text-text-primary [&_code]:font-mono [&_code]:text-[11.5px]">
      {children}
    </div>
  );
}
