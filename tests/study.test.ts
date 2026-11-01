import { describe, expect, it } from "vitest";

import { precisionRecall, silentFailures, type RunRow } from "@/lib/study";

describe("precisionRecall", () => {
  const pairs = [
    { score: 0.9, label: 1 },
    { score: 0.6, label: 0 },
    { score: 0.4, label: 1 },
    { score: 0.1, label: 0 },
  ];

  it("counts scores at the threshold as flagged", () => {
    expect(precisionRecall(pairs, 0.6)).toEqual({ precision: 0.5, recall: 0.5, flagged: 2 });
  });

  it("returns NaN precision when nothing is flagged", () => {
    expect(precisionRecall(pairs, 0.95).precision).toBeNaN();
  });
});

describe("silentFailures", () => {
  const run = (o: Partial<RunRow>): RunRow => ({ reward: 0, pushback_max_before: 0.1, ...o });

  it("keeps failed Jev runs below the threshold only", () => {
    const runs = [
      run({ run_id: "a" }),
      run({ run_id: "b", reward: 1 }), // passed
      run({ run_id: "c", pushback_max_before: 0.5 }), // pushback at the threshold
      run({ run_id: "d", pushback_max_before: null }), // model without Jev checks
    ];
    expect(silentFailures(runs).map((r) => r.run_id)).toEqual(["a"]);
  });
});
