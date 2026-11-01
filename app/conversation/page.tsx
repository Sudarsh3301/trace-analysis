import { Suspense } from "react";

import { ConversationFromQuery } from "@/components/study/conversation-from-query";
import { study } from "@/lib/study";

export default function ConversationPage() {
  const labels = Object.fromEntries(study.models.map((m) => [m.key, m.label]));
  return (
    <Suspense>
      <ConversationFromQuery labels={labels} />
    </Suspense>
  );
}
