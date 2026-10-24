"""Retrieval claim: do runs fail more when a required document never entered the context?

Usage: python -m analysis.c3_retrieval --submission <name> | --all
Writes results/c3_retrieval.csv (--all: local/work/<name>/c3_retrieval.csv for each).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .boot import ci, task_counts, task_sums
from .docs import doc_events
from .facts import out_csv, work_dirs

BINS = [("1-3", 1, 3), ("4-8", 4, 8), ("9+", 9, 10**6)]
EXPOSURES = ("all_in_context", "all_opened")


def row(section, exposure, stratum, quantity, n, est, lo=np.nan, hi=np.nan, note=""):
    return {"section": section, "exposure": exposure, "stratum": stratum, "quantity": quantity,
            "n": n, "estimate": round(float(est), 3), "ci_low": round(float(lo), 3),
            "ci_high": round(float(hi), 3), "note": note}


def contrast(df, exposure, stratum):
    """P(fail | exposure false), P(fail | exposure true) and their difference, with CIs."""
    e, f = df[exposure].values, df.fail.values
    n_bad, f_bad, n_ok, f_ok = task_sums(df.task_id.values, ~e, ~e & (f == 1), e, e & (f == 1))
    _, _, counts = task_counts(df.task_id.values)

    def stats(c):
        with np.errstate(invalid="ignore", divide="ignore"):
            a, b = (c @ f_bad) / (c @ n_bad), (c @ f_ok) / (c @ n_ok)
        return a, b, a - b
    est = [float(x[0]) for x in stats(np.ones((1, len(n_bad))))]
    draws = stats(counts)
    nb, no = int((~e).sum()), int(e.sum())
    note = f"n_incomplete={nb}, n_complete={no}" + (" (n<20)" if min(nb, no) < 20 else "")
    sec = "raw" if stratum == "all" else "stratified"
    return [row(sec, exposure, stratum, q, n, est[i], *ci(draws[i]), note)
            for i, (q, n) in enumerate([("p_fail_incomplete", nb), ("p_fail_complete", no),
                                        ("difference", len(df))])]


def paired(df, exposure):
    """Tasks with mixed outcomes: compare pass vs fail trials on the share with the exposure."""
    task_diff, pair_pf, pair_fp = [], 0, 0  # pairs: (pass has it, fail lacks) / reverse
    for _, g in df.groupby("task_id"):
        if g.fail.nunique() < 2:
            continue
        ps, fs = g[g.fail == 0][exposure], g[g.fail == 1][exposure]
        task_diff.append(ps.mean() - fs.mean())
        pair_pf += int(ps.sum() * (len(fs) - fs.sum()))
        pair_fp += int((len(ps) - ps.sum()) * fs.sum())
    pos, neg = sum(d > 0 for d in task_diff), sum(d < 0 for d in task_diff)
    p_task = binomtest(pos, pos + neg).pvalue if pos + neg else np.nan
    p_pair = binomtest(pair_pf, pair_pf + pair_fp).pvalue if pair_pf + pair_fp else np.nan
    s = "mixed tasks"
    return [row("paired", exposure, s, "n_tasks", len(task_diff), len(task_diff)),
            row("paired", exposure, s, "mean_share_diff_pass_minus_fail", len(task_diff),
                np.mean(task_diff) if task_diff else np.nan),
            row("paired", exposure, s, "tasks_pass_higher", pos + neg, pos,
                note=f"tasks_fail_higher={neg}, ties={len(task_diff) - pos - neg}; "
                     f"exact sign test p={p_task:.4f} (task level)"),
            row("paired", exposure, s, "discordant_pairs_pass_has", pair_pf + pair_fp, pair_pf,
                note=f"reverse={pair_fp}; exact sign test p={p_pair:.4f} (pairs share runs, so not independent)")]


def retrieval_rows(df):
    """Raw, stratified and within-task paired results; df needs task_id, fail, n_req, exposures."""
    out = []
    for exp in EXPOSURES:
        out += contrast(df, exp, "all")
        for name, lo, hi in BINS:
            out += contrast(df[df.n_req.between(lo, hi)], exp, f"n_required {name}")
        out += paired(df, exp)
    return out


def exposure_frame(runs, facts, ft):
    """One row per run with fail, exposures, failure flags (all joined on run_id)."""
    f = facts[facts.k == "all"].set_index("run_id")
    df = runs.set_index("run_id").join(ft.drop(columns="reward").set_index("run_id"))
    df["fail"] = (df.reward == 0).astype(int)
    df["all_in_context"] = f.required_in_context == f.required_docs_total
    df["all_opened"] = f.required_opened == f.required_docs_total
    df["n_req"] = df.required_docs.map(len)
    return df.reset_index()


def timing(df, rows_by_run):
    """Failed runs with a missing/unexpected write: was a required doc missing from context when
    the first unexpected write happened, and by the end of the run?"""
    sel = df[(df.fail == 1) & (df.expected_write_never_attempted | df.unexpected_write)]
    at_write = at_end = both = n_w = 0
    for r in sel.itertuples():
        ev = doc_events(rows_by_run[r.run_id])
        first = {d: ev.get(d, {}).get("in_context") for d in r.required_docs}
        end_missing = any(v is None for v in first.values())
        at_end += end_missing
        if r.unexpected_write:
            n_w += 1
            t = r.first_unexpected_write_idx
            miss = any(v is None or v > t for v in first.values())
            at_write += miss
            both += miss and not end_missing
    return [row("timing", "in_context", "failed runs with flagged write", "n_runs", len(sel), len(sel)),
            row("timing", "in_context", "failed runs with unexpected write", "missing_at_first_unexpected_write",
                n_w, at_write / n_w, note=f"{at_write} of {n_w}; {both} of them got the doc into context only later"),
            row("timing", "in_context", "failed runs with flagged write", "missing_at_end_of_run",
                len(sel), at_end / len(sel), note=f"{at_end} of {len(sel)}")]


def main(work, many=False):
    runs, facts, ft, msgs = (pd.read_parquet(work / f"{n}.parquet")
                             for n in ("runs", "facts", "failure_types", "messages"))
    df = exposure_frame(runs, facts, ft)
    rows_by_run = {r: g.sort_values("idx", kind="stable").to_dict("records") for r, g in msgs.groupby("run_id")}
    out = pd.DataFrame(retrieval_rows(df) + timing(df, rows_by_run))
    Path("results").mkdir(exist_ok=True)
    out.to_csv(out_csv("c3_retrieval", work, many), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    for d in work_dirs("c3_retrieval"):
        main(d, sys.argv[1] == "--all")
