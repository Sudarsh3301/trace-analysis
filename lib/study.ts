/**
 * The τ-bench study data, exported by `python -m analysis.export_site` into public/data/.
 * Every number on the site comes from here; nothing is typed in by hand.
 */
import raw from "@/public/data/study.json";

/** URL prefix when the site is served from a sub-path (GitHub Pages: /trace-analysis). Empty locally. */
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
export const REPO_URL = "https://github.com/Sudarsh3301/trace-analysis";
/** Per-run conversation files, published as a dataset (local/site-data/runs/ in the repo). */
export const DATASET_URL = "https://huggingface.co/datasets/JoyboyXoXo/agentic-failure-attribution-traces";
/** Pushback threshold fixed before the analysis; 0.8 was chosen afterwards on the validation sample. */
export const PRESET_THRESHOLD = 0.5;

export interface StudyModel {
  key: string;
  label: string;
  pass_rate: number;
  /** True for the 4 models that have Jev step judgments and pushback checks. */
  jev: boolean;
}

export interface AurocRow {
  section?: string;
  model: string;
  /** "2", "4" or "all" user turns; the Jev table stores it as a number. */
  k: string | number;
  feature_set: string;
  auroc: number;
  ci_low: number;
  ci_high: number;
}

export interface SilentRow {
  threshold: number;
  model: string;
  group: string;
  n_runs: number;
  share_no_pushback_before_decisive: number;
  ci_low_before_decisive: number;
  ci_high_before_decisive: number;
  share_no_pushback_whole_run: number;
  ci_low_whole_run: number;
  ci_high_whole_run: number;
}

export interface WindowRow {
  threshold: number;
  measure: string;
  passed: number;
  failed: number;
  failed_minus_passed: number;
  ci_low: number;
  ci_high: number;
}

export interface PooledRow {
  section: string;
  quantity: string;
  n: number;
  estimate: number;
  ci_low: number | null;
  ci_high: number | null;
  note: string | null;
}

export interface ValidationPair {
  run_id: string;
  idx: number;
  model: string;
  /** Jev pushback score: the max of the 4 pushback questions. */
  score: number;
  /** 1 = pushback, as labelled by a separate LLM that did not see the scores. */
  label: number;
}

export interface Study {
  models: StudyModel[];
  /** Sorted hardest first. */
  tasks: { id: string; pass_rate: number }[];
  /** passes[model][task]: passing runs; trials[model][task]: runs. */
  passes: number[][];
  trials: number[][];
  tables: {
    auroc_prior: AurocRow[];
    auroc_jev: AurocRow[];
    silent: SilentRow[];
    window: WindowRow[];
    retrieval: PooledRow[];
    validation: { method: string; n: number; precision: number; recall: number }[];
    jev_usage: { part: string; calls: number; cost_usd: number }[];
  };
  prior_check: {
    always_fail_tasks: number;
    auroc_all: number;
    n_all: number;
    auroc_without_always_fail: number;
    n_without: number;
  };
  /** Per task with mixed outcomes: passed minus failed share of runs with the exposure. */
  retrieval_tasks: Record<"all_in_context" | "all_opened", { task: string; diff: number; units: number }[]>;
  validation: ValidationPair[];
}

export const study = raw as unknown as Study;

export const kOf = (row: AurocRow) => String(row.k).replace(/\.0$/, "");
export const MEAN = "MEAN OF MODELS";

export const auroc = (table: AurocRow[], k: string, set: string, model = MEAN) =>
  table.find((r) => r.model === model && kOf(r) === k && r.feature_set === set && r.section !== "delta");

export const pooled = (section: string, quantity: string, note?: string) =>
  study.tables.retrieval.find((r) => r.section === section && r.quantity === quantity && (note === undefined || r.note === note));

/** For next/link, which adds BASE_PATH itself; plain <a> tags must prefix it. */
export const conversationHref = (runId: string) => `/conversation?run=${runId}`;

/** Precision and recall of "score >= threshold" against the labels. */
export function precisionRecall(pairs: { score: number; label: number }[], threshold: number) {
  const predicted = pairs.filter((p) => p.score >= threshold);
  const truePos = predicted.filter((p) => p.label === 1).length;
  const actual = pairs.filter((p) => p.label === 1).length;
  return {
    precision: predicted.length ? truePos / predicted.length : NaN,
    recall: actual ? truePos / actual : NaN,
    flagged: predicted.length,
  };
}

// ---- runs.json ----

export type RunRow = Record<string, string | number | boolean | null>;

export async function fetchRuns(): Promise<RunRow[]> {
  const res = await fetch(`${BASE_PATH}/data/runs.json`);
  if (!res.ok) throw new Error(`runs.json: HTTP ${res.status}`);
  const { columns, rows } = (await res.json()) as { columns: string[]; rows: unknown[][] };
  return rows.map((r) => Object.fromEntries(columns.map((c, i) => [c, r[i]])) as RunRow);
}

/** Failed runs of the Jev models with no pushback at or above the threshold before the decisive point. */
export function silentFailures(runs: RunRow[], threshold = PRESET_THRESHOLD) {
  return runs.filter(
    (r) => r.reward === 0 && r.pushback_max_before !== null && Number(r.pushback_max_before) < threshold,
  );
}

// ---- conversations (Hugging Face dataset) ----

export interface ConversationMessage {
  idx: number;
  role: "user" | "assistant" | "tool";
  turn: number;
  text: string | null;
  tool: string | null;
  args: string | null;
}

export interface Conversation {
  run_id: string;
  model: string;
  task_id: string;
  trial: number;
  reward: number;
  decisive: { idx: number; kind: "unexpected_write" | "wrong_write" | "end_of_run" };
  /** One row per tool call: a message with several calls repeats its idx. */
  messages: ConversationMessage[];
  /** Jev models only, keyed by message idx. */
  judgments?: Record<string, Record<string, number | string>>;
  pushback?: Record<string, Record<string, number>>;
}

export const isRunId = (id: string) => /^[0-9a-f-]{36}$/.test(id);

export async function fetchConversation(runId: string): Promise<Conversation> {
  if (!isRunId(runId)) throw new Error("not a run id");
  const res = await fetch(`${DATASET_URL}/resolve/main/runs/${runId}.json`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
