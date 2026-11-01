/**
 * Types for the semantic evaluation layer.
 *
 * Vocabulary, kept deliberately separate (see README "Scientific constraints"):
 *   - prediction:   a probability Jev returned. Never a label, never ground truth.
 *   - confidence:   how concentrated a Choice distribution is. Not correctness.
 *   - label:        an externally supplied expectation. Only these justify metrics.
 *   - derived:      anything the app computed from the above (drift, F1, ...).
 */

export const NOUL_DIMENSIONS = [
  "taskAlignment",
  "progress",
  "repetition",
  "stuck",
  "needsVerification",
  "prematureCompletion",
  "unexpectedDirection",
] as const;

export type NoulDimension = (typeof NOUL_DIMENSIONS)[number];

export const PHASES = [
  "exploring",
  "implementing",
  "verifying",
  "recovering",
  "finished",
  "unclear",
] as const;

export type Phase = (typeof PHASES)[number];

/**
 * Whether a high probability on a dimension reads as encouraging or concerning.
 * Used only for presentation; it never alters a value. High `stuck` is not
 * "wrong", it is a signal worth looking at.
 */
export const DIMENSION_VALENCE: Record<NoulDimension, "positive" | "concerning" | "neutral"> = {
  taskAlignment: "positive",
  progress: "positive",
  repetition: "concerning",
  stuck: "concerning",
  needsVerification: "neutral",
  prematureCompletion: "concerning",
  unexpectedDirection: "concerning",
};

export const DIMENSION_LABELS: Record<NoulDimension, string> = {
  taskAlignment: "Task alignment",
  progress: "Progress",
  repetition: "Repetition",
  stuck: "Stuck",
  needsVerification: "Needs verification",
  prematureCompletion: "Premature completion",
  unexpectedDirection: "Unexpected direction",
};

/** The state object handed to Jev. Deterministic given (run, step, history, options). */
export interface JevState {
  task: string;
  systemContext?: string;
  currentStep: {
    stepNumber: number;
    type: string;
    content: string;
    tool?: string;
    arguments?: unknown;
    result?: unknown;
  };
  recentHistory: Array<{
    stepNumber: number;
    type: string;
    content: string;
    tool?: string;
    result?: unknown;
  }>;
  /** Present only when history was dropped, so the model knows the window is partial. */
  historyNote?: string;
  /** Perturbations may add fields; they are always visible in the raw inspector. */
  [key: string]: unknown;
}

/** A parsed set of judgments over one state. */
export interface SemanticJudgment {
  nouls: Record<NoulDimension, number>;
  phase: Phase;
  phaseProbabilities: Record<Phase, number>;
  phaseConfidence: number;
  latencyMs: number;
  model: string;
  source: "jev" | "mock";
  requestId?: string;
  /** The unmodified response body, stored for reproducibility. */
  raw: unknown;
}

export interface EvaluationFailure {
  ok: false;
  error: string;
  latencyMs: number;
  model: string;
}

export type EvaluationOutcome = ({ ok: true } & SemanticJudgment) | EvaluationFailure;
