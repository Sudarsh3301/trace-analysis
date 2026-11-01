import { JudgePanel } from "@/components/study/judge-panel";
import { PushbackPanel } from "@/components/study/pushback-panel";
import { SilentFailures } from "@/components/study/silent-failures";
import { Page, Section } from "@/components/ui/page";
import { study } from "@/lib/study";

export default function PushbackPage() {
  const { tables, validation, models, tasks } = study;
  const keyword = tables.validation.find((r) => r.method === "keyword baseline");

  return (
    <Page
      title="Pushback is not an early warning"
      description="Monitors that watch for user dissatisfaction assume failing conversations show it before the mistake. On these conversations, they do not."
    >
      <Section
        title="Pushback before the decisive action"
        description="Jev checked every natural user message for a correction, a repeated request, a complaint or a request for a human."
      >
        <PushbackPanel silent={tables.silent} window={tables.window} />
      </Section>

      <Section
        title="How good is the pushback check?"
        description="Move the threshold to see precision and recall on the validation sample. The finding above uses 0.5, which was fixed before the analysis."
      >
        <JudgePanel pairs={validation} keyword={keyword} />
      </Section>

      <Section
        title="Silent failures"
        description="Failed conversations with no pushback before the decisive action. Open any one to read it."
      >
        <SilentFailures models={models} tasks={tasks.map((t) => t.id)} />
      </Section>
    </Page>
  );
}
