"""Label check: file pass rate vs leaderboard, before and after re-scoring; picks the analysis sets.

Usage: python -m analysis.data_check
Reads local/data/tau/{kept,skipped}.json, local/work/<name> (original labels) and, where present,
local/data/tau-rescored/<name>/updated_*.json (re-scored with tau2; see results/data_check.csv).
Writes results/data_check.csv, results/missing_runs.csv, local/data/tau/kept_A.json (set A, re-scored
labels) and local/data/tau/kept_B.json (set B, original labels).

Rules: label ok = re-scored pass rate within 1 point of the leaderboard pass_1; missing runs are
(task, trial) pairs without a scored run (results/missing_runs.csv lists those of the original files); a file is kept if < 5% are missing and the 5 most-missing
tasks hold at most half of them. Set B = original pass rate within 1 point and >= 386 runs.
"""
import json
from pathlib import Path

import pandas as pd

from . import load

DATA = Path("local/data/tau")
RESCORED = Path("local/data/tau-rescored")
WORK = Path("local/work")
WORK_RS = Path("local/work-rescored")
EXPECTED = 388


def stats(runs):
    return len(runs), 100 * (runs.reward == 1).mean()


def missing(runs, tasks):
    have = set(zip(runs.task_id, runs.trial))
    trials = range(EXPECTED // len(tasks))
    return [(t, k) for t in tasks for k in trials if (t, k) not in have]


def main():
    kept = json.loads((DATA / "kept.json").read_text(encoding="utf-8"))
    skipped = json.loads((DATA / "skipped.json").read_text(encoding="utf-8"))
    rows, miss_rows, set_a, set_b = [], [], {}, {}
    for name in sorted({*kept, *skipped}):
        if name not in kept:
            rows.append({"submission": name, "status": "skipped", "reason": skipped[name]})
            continue
        info = kept[name]
        runs = pd.read_parquet(WORK / name / "runs.parquet")
        n, rate = stats(runs)
        r = {"submission": name, "file": Path(info["file"]).name, "status": "kept",
             "n_runs": n, "n_expected": EXPECTED, "pass_rate_file": round(rate, 2),
             "leaderboard_pass_1": round(info["pass_1"], 2), "gap": round(rate - info["pass_1"], 2)}
        files = list((RESCORED / name).glob("updated_*.json"))
        reasons = []
        if not files:
            r["rescored"] = "not loadable by tau2 v1.0.1 (legacy message schema)"
            reasons.append("cannot be re-scored")
        else:
            rs = load.load(files[0], WORK_RS / name)
            rr = pd.read_parquet(rs / "runs.parquet")
            rn, rrate = stats(rr)
            tasks = sorted(rr.task_id.unique())
            gone = missing(rr, tasks)
            before = missing(runs, tasks)
            miss_rows += [{"submission": name, "task_id": t, "trial": k, "recovered_by_rescoring": (t, k) not in gone}
                          for t, k in before]
            top5 = pd.Series([t for t, _ in gone], dtype=str).value_counts().head(5).sum() if gone else 0
            r.update(rescored="yes", rescored_n_runs=rn, rescored_pass_rate=round(rrate, 2),
                     rescored_gap=round(rrate - info["pass_1"], 2), n_missing_before=len(before), n_missing=len(gone),
                     changed_runs=int((rr.set_index("run_id").reward != runs.set_index("run_id").reward.reindex(rr.run_id)).sum()))
            if abs(rrate - info["pass_1"]) > 1:
                reasons.append(f"re-scored rate off leaderboard by {rrate - info['pass_1']:+.1f}")
            if len(gone) / EXPECTED >= 0.05:
                reasons.append(f"{len(gone)} runs missing (>= 5%)")
            elif gone and top5 > len(gone) / 2:
                reasons.append(f"missing runs concentrated: {top5} of {len(gone)} in 5 tasks")
        r["set_A"] = not reasons
        r["reason"] = "; ".join(reasons)
        r["set_B"] = abs(rate - info["pass_1"]) <= 1 and n >= 386
        if r["set_A"]:
            set_a[name] = {**info, "file": str(files[0])}
        if r["set_B"]:
            set_b[name] = info
        rows.append(r)
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/data_check.csv", index=False)
    pd.DataFrame(miss_rows, columns=["submission", "task_id", "trial", "recovered_by_rescoring"]).to_csv("results/missing_runs.csv", index=False)
    (DATA / "kept_A.json").write_text(json.dumps(set_a, indent=1), encoding="utf-8")
    (DATA / "kept_B.json").write_text(json.dumps(set_b, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).drop(columns=["file"]).to_string(index=False))
    print(f"set A: {len(set_a)} files, set B: {len(set_b)} files")


if __name__ == "__main__":
    main()
