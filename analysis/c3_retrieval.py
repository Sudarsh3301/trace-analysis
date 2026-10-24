"""Retrieval claim: do runs fail more when a required document never entered the context?

Usage: python -m analysis.c3_retrieval [work_dir]   -> results/c3_retrieval.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .docs import doc_events
from .facts import work_dir

BOOT, SEED = 2000, 0
BINS = [("1-3", 1, 3), ("4-8", 4, 8), ("9+", 9, 10**6)]
rows = []


def add(section, exposure, stratum, quantity, n, est, lo=np.nan, hi=np.nan, note=""):
    rows.append({"section": section, "exposure": exposure, "stratum": stratum, "quantity": quantity,
                 "n": n, "estimate": round(float(est), 3), "ci_low": round(float(lo), 3),
                 "ci_high": round(float(hi), 3), "note": note})


def clustered(df, stat):
    """stat(df) -> tuple of numbers; returns estimates and 95% task-clustered bootstrap CIs."""
    rng = np.random.default_rng(SEED)
    groups = {t: g for t, g in df.groupby("task_id")}
    keys = list(groups)
    draws = []
    for _ in range(BOOT):
        d = pd.concat([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        draws.append(stat(d))
    lo, hi = np.nanpercentile(np.array(draws, dtype=float), [2.5, 97.5], axis=0)
    return stat(df), lo, hi


def split_stat(exposure):
    """P(fail | exposed=not complete), P(fail | complete), difference."""
    def stat(d):
        a, b = d[~d[exposure]].fail.mean(), d[d[exposure]].fail.mean()
        return a, b, a - b
    return stat


def contrast(df, exposure, stratum):
    est, lo, hi = clustered(df, split_stat(exposure))
    n_bad, n_ok = int((~df[exposure]).sum()), int(df[exposure].sum())
    small = " (n<20)" if min(n_bad, n_ok) < 20 else ""
    for i, q in enumerate(["p_fail_incomplete", "p_fail_complete", "difference"]):
        add("raw" if stratum == "all" else "stratified", exposure, stratum, q,
            n_bad if i == 0 else n_ok if i == 1 else len(df), est[i], lo[i], hi[i],
            f"n_incomplete={n_bad}, n_complete={n_ok}{small}")


def paired(df, exposure):
    """Tasks with mixed outcomes: compare pass vs fail trials on the share with the exposure."""
    task_diff, pair_pf, pair_fp = [], 0, 0  # pair counts: discordant (pass has it, fail lacks) / reverse
    for _, g in df.groupby("task_id"):
        if g.fail.nunique() < 2:
            continue
        ps, fs = g[g.fail == 0][exposure], g[g.fail == 1][exposure]
        task_diff.append(ps.mean() - fs.mean())
        pair_pf += int(ps.sum() * (len(fs) - fs.sum()))
        pair_fp += int((len(ps) - ps.sum()) * fs.sum())
    pos, neg = sum(d > 0 for d in task_diff), sum(d < 0 for d in task_diff)
    add("paired", exposure, "mixed tasks", "n_tasks", len(task_diff), len(task_diff))
    add("paired", exposure, "mixed tasks", "mean_share_diff_pass_minus_fail", len(task_diff),
        np.mean(task_diff))
    p_task = binomtest(pos, pos + neg).pvalue if pos + neg else np.nan
    add("paired", exposure, "mixed tasks", "tasks_pass_higher", pos + neg, pos,
        note=f"tasks_fail_higher={neg}, ties={len(task_diff) - pos - neg}; exact sign test p={p_task:.4f} (task level)")
    p_pair = binomtest(pair_pf, pair_pf + pair_fp).pvalue if pair_pf + pair_fp else np.nan
    add("paired", exposure, "mixed tasks", "discordant_pairs_pass_has", pair_pf + pair_fp, pair_pf,
        note=f"reverse={pair_fp}; exact sign test p={p_pair:.4f} (pairs share runs, so not independent)")


def timing(df, rows_by_run):
    """Failed runs with a missing/unexpected write: was a required doc missing from context when
    the first unexpected write happened, and by the end of the run?"""
    sel = df[(df.fail == 1) & (df.expected_write_never_attempted | df.unexpected_write)]
    at_write = at_end = both = n_w = 0
    for r in sel.itertuples():
        ev = doc_events(rows_by_run[r.run_id])
        first = {d: (ev.get(d, {}).get("in_context")) for d in r.required_docs}
        end_missing = any(v is None for v in first.values())
        at_end += end_missing
        if r.unexpected_write:
            n_w += 1
            t = r.first_unexpected_write_idx
            miss = any(v is None or v > t for v in first.values())
            at_write += miss
            both += miss and not end_missing
    add("timing", "in_context", "failed runs with flagged write", "n_runs", len(sel), len(sel))
    add("timing", "in_context", "failed runs with unexpected write", "missing_at_first_unexpected_write", n_w,
        at_write / n_w, note=f"{at_write} of {n_w}; {both} of them got the doc into context only later")
    add("timing", "in_context", "failed runs with flagged write", "missing_at_end_of_run", len(sel),
        at_end / len(sel), note=f"{at_end} of {len(sel)}")


def main(work):
    runs, facts, ft, msgs = (pd.read_parquet(work / f"{n}.parquet")
                             for n in ("runs", "facts", "failure_types", "messages"))
    f = facts[facts.k == "all"].set_index("run_id")
    df = runs.set_index("run_id").join(ft.drop(columns="reward").set_index("run_id"))
    df["fail"] = (df.reward == 0).astype(int)
    df["all_in_context"] = f.required_in_context == f.required_docs_total
    df["all_opened"] = f.required_opened == f.required_docs_total
    df["n_req"] = df.required_docs.map(len)
    df = df.reset_index()
    rows_by_run = {r: g.sort_values("idx", kind="stable").to_dict("records") for r, g in msgs.groupby("run_id")}
    for exp in ("all_in_context", "all_opened"):
        contrast(df, exp, "all")
        for name, lo, hi in BINS:
            contrast(df[df.n_req.between(lo, hi)], exp, f"n_required {name}")
        paired(df, exp)
    timing(df, rows_by_run)
    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/c3_retrieval.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(work_dir(sys.argv[1] if len(sys.argv) > 1 else None))
