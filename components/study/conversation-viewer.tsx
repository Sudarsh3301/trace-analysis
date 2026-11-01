"use client";

import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { ProbabilityBar } from "./probability-bar";
import { Badge, PageHeader, RawText, Stat } from "@/components/ui/primitives";
import { DIMENSION_PRESENTATION } from "@/lib/jev/presentation";
import { NOUL_DIMENSIONS } from "@/lib/jev/types";
import { fetchConversation, PRESET_THRESHOLD, type Conversation, type ConversationMessage } from "@/lib/study";
import { cn, fmtProbability } from "@/lib/ui";

const DECISIVE: Record<Conversation["decisive"]["kind"], string> = {
  unexpected_write: "The first write action that the task did not expect, by the agent or the customer.",
  wrong_write: "The first expected write action made with wrong arguments, by the agent or the customer.",
  end_of_run: "No extra or wrong write happened, so the decisive point is the end of the run.",
};
const PUSHBACK = [
  ["corrects", "Corrects the agent"],
  ["repeats", "Repeats a request"],
  ["complains", "Complains"],
  ["asksHuman", "Asks for a human"],
  ["saysDone", "Says it is done"],
] as const;
const THRESHOLD = PRESET_THRESHOLD;

export function ConversationViewer({ runId, labels }: { runId: string; labels: Record<string, string> }) {
  const [conv, setConv] = useState<Conversation | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchConversation(runId).then(setConv, (e: Error) => setError(e.message));
  }, [runId]);

  // a message with several tool calls has one row per call
  const groups = useMemo(() => {
    const out: ConversationMessage[][] = [];
    for (const m of conv?.messages ?? []) {
      const last = out.at(-1);
      if (last && last[0].idx === m.idx) last.push(m);
      else out.push([m]);
    }
    return out;
  }, [conv]);

  const back = (
    <Link href="/" className="inline-flex items-center gap-1 text-[12px] text-text-secondary hover:text-text-primary">
      <ArrowLeft size={13} /> Overview
    </Link>
  );

  if (error || !conv) {
    return (
      <div>
        <PageHeader title="Conversation" actions={back} />
        <p className={cn("px-6 py-5 text-[12.5px]", error ? "text-[var(--color-status-critical)]" : "text-text-muted")}>
          {error ? `Could not load this conversation: ${error}` : "Loading the conversation from the dataset…"}
        </p>
      </div>
    );
  }

  const failed = conv.reward === 0;
  const jev = conv.judgments !== undefined;
  const before = Object.entries(conv.pushback ?? {}).filter(([i]) => Number(i) < conv.decisive.idx);
  const maxBefore = Math.max(0, ...before.map(([, p]) => p.pushback));
  const atDecisive = conv.messages.filter((m) => m.idx === conv.decisive.idx && m.role !== "tool");
  const byCustomer = conv.decisive.kind !== "end_of_run" && atDecisive.every((m) => m.role === "user");

  return (
    <div>
      <PageHeader
        title={`${labels[conv.model] ?? conv.model} · ${conv.task_id} · trial ${conv.trial + 1}`}
        description={
          <span className="space-x-1.5">
            <Badge tone={failed ? "critical" : "good"}>{failed ? "failed" : "passed"}</Badge>
            {jev && failed && maxBefore < THRESHOLD && <Badge tone="warning">silent failure</Badge>}
            {byCustomer && <Badge>decided by a customer action</Badge>}
            <span>{DECISIVE[conv.decisive.kind]}</span>
          </span>
        }
        actions={back}
      />

      <div className="mx-auto max-w-[1100px] space-y-4 px-4 py-5 md:px-6">
        <section className="grid grid-cols-2 gap-2 lg:grid-cols-4">
          <Stat label="Messages" value={groups.length} />
          <Stat label="Decisive point" value={`#${conv.decisive.idx}`} hint={conv.decisive.kind.replaceAll("_", " ")} />
          <Stat
            label="Max pushback before it"
            value={jev ? fmtProbability(maxBefore) : "—"}
            hint={jev ? `${before.length} user message${before.length === 1 ? "" : "s"} checked` : "No Jev checks for this model"}
            tone={jev ? "default" : "muted"}
          />
          <Stat
            label="Jev step judgments"
            value={jev ? Object.keys(conv.judgments ?? {}).length : "—"}
            hint={jev ? "First 6 agent steps after the greeting" : "No Jev checks for this model"}
            tone={jev ? "default" : "muted"}
          />
        </section>

        <ol className="space-y-2">
          {groups.map((g) => (
            <li key={g[0].idx}>
              {g[0].idx === conv.decisive.idx && (
                <div className="my-3 flex items-center gap-2 text-[11px] font-medium text-[var(--color-status-critical)]">
                  <span className="h-px flex-1 bg-[var(--color-status-critical)]" />
                  Decisive point
                  <span className="h-px flex-1 bg-[var(--color-status-critical)]" />
                </div>
              )}
              <Message
                rows={g}
                judgment={conv.judgments?.[g[0].idx]}
                pushback={conv.pushback?.[g[0].idx]}
              />
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

function Message({
  rows,
  judgment,
  pushback,
}: {
  rows: ConversationMessage[];
  judgment?: Record<string, number | string>;
  pushback?: Record<string, number>;
}) {
  const m = rows[0];
  const calls = rows.filter((r) => r.tool && r.role !== "tool");
  const who =
    m.role === "tool" ? `Tool result · ${m.tool ?? "?"}` : m.role === "assistant" ? "Agent" : calls.length ? "Customer action" : "Customer";

  return (
    <div className={cn("panel px-3.5 py-2.5", m.role === "user" && "bg-surface-2")}>
      <div className="flex items-center gap-2 text-[11px] text-text-muted">
        <span className="num">#{m.idx}</span>
        <span className="font-medium text-text-secondary">{who}</span>
        {pushback && (
          <Badge tone={pushback.pushback >= THRESHOLD ? "warning" : "neutral"} className="ml-auto">
            pushback {fmtProbability(pushback.pushback)}
          </Badge>
        )}
      </div>

      {m.role === "tool" ? (
        <details className="mt-1">
          <summary className="cursor-pointer text-[11.5px] text-text-muted">
            Show result{(m.text ?? "").includes("characters omitted]") ? " (cut to 1,500 characters)" : ""}
          </summary>
          <RawText value={m.text ?? ""} className="mt-1.5" />
        </details>
      ) : (
        m.text && <p className="mt-1 text-[12.5px] leading-relaxed whitespace-pre-wrap">{m.text}</p>
      )}

      {calls.map((c, i) => (
        <div key={i} className="mt-1.5 rounded-md bg-surface-3 px-2.5 py-1.5">
          <div className="num text-[11.5px] font-medium">{c.tool}</div>
          {c.args && <RawText value={c.args} />}
        </div>
      ))}

      {pushback && (
        <details className="mt-1.5">
          <summary className="cursor-pointer text-[11px] text-text-muted">Jev pushback check</summary>
          <div className="mt-1.5 space-y-1">
            {PUSHBACK.map(([k, label]) => (
              <ProbabilityBar key={k} label={label} value={pushback[k]} color="var(--color-accent)" width={120} />
            ))}
          </div>
        </details>
      )}

      {judgment && (
        <details className="mt-1.5">
          <summary className="cursor-pointer text-[11px] text-text-muted">
            Jev step judgment · phase {String(judgment.phase)} · stuck {fmtProbability(Number(judgment.stuck))}
          </summary>
          <div className="mt-1.5 grid gap-1 sm:grid-cols-2">
            {NOUL_DIMENSIONS.map((d) => (
              <ProbabilityBar
                key={d}
                label={DIMENSION_PRESENTATION[d].short}
                value={Number(judgment[d])}
                color={DIMENSION_PRESENTATION[d].colorVar}
              />
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
