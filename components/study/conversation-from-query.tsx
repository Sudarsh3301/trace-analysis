"use client";

import { useSearchParams } from "next/navigation";

import { PageHeader } from "@/components/ui/primitives";
import { isRunId } from "@/lib/study";

import { ConversationViewer } from "./conversation-viewer";

/** One static page serves every conversation: the run id comes from `?run=`. */
export function ConversationFromQuery({ labels }: { labels: Record<string, string> }) {
  const run = useSearchParams().get("run") ?? "";
  if (!isRunId(run)) {
    return (
      <PageHeader
        title="Conversation"
        description="No valid run id in the link. Open a conversation from the heatmap or the silent-failure list."
      />
    );
  }
  return <ConversationViewer key={run} runId={run} labels={labels} />;
}
