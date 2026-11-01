import { AurocPanel } from "@/components/study/auroc-panel";
import { TaskHeatmap } from "@/components/study/task-heatmap";
import { Note, Page, Section } from "@/components/ui/page";
import { Stat } from "@/components/ui/primitives";
import { pooled, study } from "@/lib/study";

export default function TaskPriorPage() {
  const { models, tasks, passes, trials, tables, prior_check: pc } = study;
  const taskOnly = pooled("3.4 AUROC of priors", "task_only (other models' fail rate)");
  const modelOnly = pooled("3.4 AUROC of priors", "model_only (model fail rate)");
  const ci = (r?: { ci_low: number | null; ci_high: number | null }) =>
    r?.ci_low != null && r.ci_high != null ? `[${r.ci_low.toFixed(3)}, ${r.ci_high.toFixed(3)}]` : undefined;

  return (
    <Page
      title="The task comes first"
      description="Which task an agent faced predicts its failure far better than which model it was, or anything in its trace."
    >
      <section className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <Stat label="Task prior, AUROC" value={taskOnly?.estimate.toFixed(3) ?? "—"} hint={ci(taskOnly)} />
        <Stat label="Model identity, AUROC" value={modelOnly?.estimate.toFixed(3) ?? "—"} hint={ci(modelOnly)} />
        <Stat
          label="Tasks every run fails"
          value={`${pc.always_fail_tasks} of ${tasks.length}`}
          hint={`Without them the prior gives ${pc.auroc_without_always_fail.toFixed(3)}`}
        />
        <Stat label="Tasks every run passes" value={tasks.filter((t) => t.pass_rate === 1).length} />
      </section>
      <Note>
        Pooled over all {pc.n_all.toLocaleString("en-US")} runs. The task prior is the task&apos;s failure rate among the
        other 19 models; model identity is the model&apos;s own failure rate. Brackets: 95% intervals, bootstrap over
        tasks.
      </Note>

      <Section
        title="Every run, by model and task"
        description="Vertical bands are tasks that almost every model fails or passes. Click a cell to see its 4 runs, then open a conversation."
      >
        <TaskHeatmap models={models} tasks={tasks} passes={passes} trials={trials} />
      </Section>

      <Section
        title="Adding trace signals to the prior, by model"
        description="The same comparison as on the Overview, per model. Select a model to see that the pattern holds for each one."
      >
        <AurocPanel models={models} prior={tables.auroc_prior} jev={tables.auroc_jev} />
        <div className="mt-3">
          <Note>
            &quot;This model&apos;s other trials&quot; is a second task-level prior: the same model&apos;s results on its
            other 3 attempts at the task (leave-one-trial-out). It is not a trace signal, which is why it can beat the
            first prior.
          </Note>
        </div>
      </Section>
    </Page>
  );
}
