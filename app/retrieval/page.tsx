import { RetrievalPanel } from "@/components/study/retrieval-panel";
import { Note, Page, Section } from "@/components/ui/page";
import { study } from "@/lib/study";

export default function RetrievalPage() {
  return (
    <Page
      title="Retrieval within a task"
      description="Within the same task and model, runs that got all required policy documents into the agent's context failed less often."
    >
      <Section
        title="Passed vs. failed runs of the same task"
        description="One bar per task with mixed outcomes. Height: share of passed runs with all required documents in context, minus the same share for failed runs (averaged over models). Hover a bar for the task."
      >
        <RetrievalPanel tasks={study.retrieval_tasks} rows={study.tables.retrieval} />
      </Section>

      <Section title="How to read this">
        <div className="space-y-2 text-[12.5px] leading-relaxed text-text-secondary">
          <p>
            A raw comparison across tasks misleads: tasks that need many documents are both harder and less often fully
            retrieved. Comparing runs of the same task and model removes that. Tied tasks are those where passed and
            failed runs had the documents equally often.
          </p>
          <p>
            The risk change is in failure probability, all documents in context minus not. A negative number means fewer
            failures when the documents were there. &quot;In context&quot; means the document body entered the agent&apos;s
            context; &quot;opened&quot; is stricter and means the agent deliberately read it. The stricter measure points
            the same way but is weaker.
          </p>
          <p>
            This is an association. Runs that retrieve well may also be runs where the agent understood the request; only
            supplying the correct documents in an experiment would show cause.
          </p>
        </div>
      </Section>
      <Note>Strata are model and task pairs. Brackets: 95% intervals, bootstrap over tasks.</Note>
    </Page>
  );
}
