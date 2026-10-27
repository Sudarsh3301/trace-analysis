"""Window-length check for the pushback result (cache only; no Jev calls).

Failed runs reach their decisive write earlier than passed runs, so "no pushback before the
decisive point" is biased toward failed runs. Two length-controlled comparisons, per threshold:
  - first_k: no pushback within the first k user messages, among runs with >= k user messages
    before their decisive point (k = 1, 2, 3);
  - per_message: pushback rate per user message before the decisive point (ratio of sums).
Failed minus passed uses the same task resamples. Writes results/c1_window_check.csv.
Usage: python -m analysis.c1_window
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .boot import ci, task_counts, task_sums
from .jev import Jev
from .jev_silent import THRESHOLDS, decisive_idx, load_set, score_table, share_draws


def ratio_draws(num, den, tasks, keys):
    s_n, s_d = task_sums(tasks, num, den, keys=keys)
    _, _, counts = task_counts(tasks, keys=keys)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num.sum() / den.sum(), (counts @ s_n) / (counts @ s_d)


def per_run(data, sc):
    rows = []
    for m, t in data.items():
        by = {rid: g for rid, g in t["messages"].groupby("run_id")}
        ck = {rid: g for rid, g in t["action_checks"].groupby("run_id")}
        empty = t["action_checks"].iloc[:0]
        ft = t["failure_types"].set_index("run_id")
        s = sc[sc.model == m]
        for r in t["runs"].itertuples():
            if r.run_id not in by:
                continue
            d, _ = decisive_idx(by[r.run_id], ck.get(r.run_id, empty), ft.loc[r.run_id].rename_axis("run_id"))
            u = s[(s.run_id == r.run_id) & (s.idx < d)].sort_values("idx")
            rows.append({"model": m, "run_id": r.run_id, "task_id": r.task_id, "reward": r.reward,
                         "scores": u.pushback.to_numpy()})
    return pd.DataFrame(rows)


def main():
    data = load_set()
    sc = score_table(data, Jev(dry_run=True))  # peek reads the cache only
    runs = per_run(data, sc)
    keys = np.unique(np.concatenate([t["runs"].task_id.values for t in data.values()]))
    out = []
    for th in THRESHOLDS:
        for k in (1, 2, 3):
            sub = runs[runs.scores.map(len) >= k]
            silent = sub.scores.map(lambda x: float(not (x[:k] >= th).any())).to_numpy()
            res = {}
            for rew in (1, 0):
                msk = (sub.reward == rew).to_numpy()
                res[rew] = share_draws(silent[msk], sub.task_id.to_numpy()[msk], keys)
            lo, hi = ci(res[0][1] - res[1][1])
            out.append({"threshold": th, "measure": f"no pushback in first {k} user messages",
                        "passed": round(res[1][0], 3), "failed": round(res[0][0], 3),
                        "n_passed": int((sub.reward == 1).sum()), "n_failed": int((sub.reward == 0).sum()),
                        "failed_minus_passed": round(res[0][0] - res[1][0], 3),
                        "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
        res = {}
        for rew in (1, 0):
            sub = runs[runs.reward == rew]
            num = sub.scores.map(lambda x: float((x >= th).sum())).to_numpy()
            den = sub.scores.map(lambda x: float(len(x))).to_numpy()
            res[rew] = (ratio_draws(num, den, sub.task_id.to_numpy(), keys), int(len(sub)), int(den.sum()))
        (pf, df_), nf, mf = res[0]
        (pp, dp), np_, mp = res[1]
        lo, hi = ci(df_ - dp)
        out.append({"threshold": th, "measure": "pushback per user message before the decisive point",
                    "passed": round(pp, 3), "failed": round(pf, 3), "n_passed": np_, "n_failed": nf,
                    "failed_minus_passed": round(pf - pp, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                    "messages_passed": mp, "messages_failed": mf})
    res = pd.DataFrame(out)
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/c1_window_check.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
