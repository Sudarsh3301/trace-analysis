"""Run the pipeline on every kept submission and compare models.

Usage: python -m analysis.multi_model --set A|B [--analyze-only]
Set A / B = local/data/tau/kept_A.json / kept_B.json (from analysis.data_check): A uses the re-scored
labels (work dirs local/work-rescored/), B the original labels (local/work/). Writes per-model tables
to the work dirs, pooled tables (with a `model` column) to <work>/_all/, and
results/multi_model_{per_model,r3,pooled}_<set>.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import facts as facts_mod
from . import failure_type, load
from .boot import auc_point_and_draws, ci, task_counts, task_sums
from .c3_retrieval import EXPOSURES, exposure_frame, retrieval_rows, row
from .predict import NON_FEATURES, baseline_rows, build as build_model, oof_r1, oof_r2, task_prior

WORK = ALL = None  # set in main() for the chosen set
BASE_SETS = {("R1", "T"), ("R1", "T+F"), ("R1", "F"), ("R2", "P"), ("R2", "P+F"), ("R2", "F")}


def build_all(kept):
    """load -> facts -> failure_type per submission; write pooled runs/facts/failure_types."""
    pooled = {"runs": [], "facts": [], "failure_types": []}
    for name, info in kept.items():
        d = load.load(info["file"], WORK / name)
        runs, msgs, checks = (pd.read_parquet(d / f"{n}.parquet") for n in ("runs", "messages", "action_checks"))
        facts = facts_mod.compute(runs, msgs)
        ft = failure_type.compute(runs, msgs, checks)
        facts.to_parquet(d / "facts.parquet", index=False)
        ft.to_parquet(d / "failure_types.parquet", index=False)
        for key, df in (("runs", runs), ("facts", facts), ("failure_types", ft)):
            pooled[key].append(df.assign(model=name))
    ALL.mkdir(parents=True, exist_ok=True)
    for key, dfs in pooled.items():
        pd.concat(dfs).to_parquet(ALL / f"{key}.parquet", index=False)


def per_model_rows(kept):
    out = []
    for name, info in kept.items():
        d = WORK / name
        runs, T, facts = build_model(d)
        ft = pd.read_parquet(d / "failure_types.parquet")
        rate = 100 * (runs.reward == 1).mean()
        ok = abs(rate - info["pass_1"]) <= 1
        meta = json.loads((d / "meta.json").read_text())
        total = meta["n_simulations"]
        rate_all = 100 * (runs.reward == 1).sum() / total  # dropped runs counted as failures
        ok = ok or abs(rate_all - info["pass_1"]) <= 1
        rows = [row("runs", "", "", "n_runs", len(runs), len(runs)),
                row("runs", "", "", "n_dropped_no_reward", total, meta["dropped_no_reward"]),
                row("runs", "", "", "pass_rate_pct", len(runs), rate),
                row("runs", "", "", "pass_rate_pct_dropped_as_fail", total, rate_all),
                row("runs", "", "", "leaderboard_pass_1_pct", len(runs), info["pass_1"],
                    note="matches within 1 point" if ok else "MISMATCH")]
        for r in baseline_rows(runs, T, facts, ks=("2", "all"), only=BASE_SETS):
            rows.append(row("baseline", "", f"{r['regime']} k={r['k']}", r["feature_set"], r["n_runs"],
                            r["auroc"], r["ci_low"], r["ci_high"]))
        rows += retrieval_rows(exposure_frame(runs.drop(columns="y"), facts, ft))
        out += [{"model": name, **r} for r in rows]
    return pd.DataFrame(out)


def cross_model_prior(runs):
    """For each run: mean over the OTHER models of that model's pass rate on the run's task."""
    rate = runs.groupby(["model", "task_id"]).reward.mean().unstack("model")
    out = np.zeros(len(runs))
    for m, idx in runs.groupby("model").indices.items():
        others = rate.drop(columns=m).mean(axis=1)
        out[idx] = others.reindex(runs.task_id.values[idx]).values
    return out


def r3_rows(kept):
    """Cross-model prior C, with/without own other-trial prior P and trace facts F."""
    all_runs = pd.concat([pd.read_parquet(WORK / n / "runs.parquet").assign(model=n) for n in kept]).reset_index(drop=True)
    all_runs["C"] = cross_model_prior(all_runs)
    keys = np.unique(all_runs.task_id)
    out = []
    for k in ("2", "all"):
        res = {}  # (model, set) -> (point, draws)
        for name in kept:
            runs, _, facts = build_model(WORK / name)
            f = facts[facts.k == k].set_index("run_id").loc[runs.run_id]
            F = f[[c for c in f.columns if c not in NON_FEATURES]].fillna(0).values.astype(float)
            C = all_runs[all_runs.model == name].set_index("run_id").loc[runs.run_id].C.values[:, None]
            P = task_prior(runs)[:, None]
            y, tasks = runs.y.values, runs.task_id.values
            preds = {"C": oof_r1(C, y, tasks), "C+F": oof_r1(np.hstack([C, F]), y, tasks),
                     "C+P": oof_r2(np.hstack([P, C]), y, runs, True),
                     "C+P+F": oof_r2(np.hstack([P, C, F]), y, runs, True)}
            for s, p in preds.items():
                res[(name, s)] = auc_point_and_draws(y, p, tasks, keys=keys)
        for s in ("C", "C+F", "C+P", "C+P+F"):
            for name in [*kept, "MEAN OF MODELS"]:
                pts = [res[(m, s)] for m in (kept if name == "MEAN OF MODELS" else [name])]
                pt, dr = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts], axis=0)
                out.append({"model": name, "k": k, "feature_set": s, "auroc": pt, "ci_low": ci(dr)[0], "ci_high": ci(dr)[1]})
        for name in [*kept, "MEAN OF MODELS"]:
            for base, plus in (("C", "C+F"), ("C+P", "C+P+F")):
                pts = [(res[(m, plus)], res[(m, base)]) for m in (kept if name == "MEAN OF MODELS" else [name])]
                pt = np.mean([a[0] - b[0] for a, b in pts])
                dr = np.mean([a[1] - b[1] for a, b in pts], axis=0)
                out.append({"model": name, "k": k, "feature_set": f"delta {plus} - {base}", "auroc": pt,
                            "ci_low": ci(dr)[0], "ci_high": ci(dr)[1]})
    return pd.DataFrame(out).round(3)


def mantel_haenszel(strata, tasks):
    """RD of failure, exposed minus unexposed, over strata (model, task); task-clustered CI."""
    n1, n0 = strata.n_exposed, strata.n - strata.n_exposed
    a, c = strata.fail_exposed, strata.fail - strata.fail_exposed
    num = ((a * n0 - c * n1) / strata.n).values
    den = (n1 * n0 / strata.n).values
    num_t, den_t = task_sums(tasks, num, den)
    _, _, counts = task_counts(tasks)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (counts @ num_t) / (counts @ den_t)
    return num.sum() / den.sum(), *ci(draws), len(strata)


def pooled_rows(kept):
    runs, facts, ft = (pd.read_parquet(ALL / f"{n}.parquet") for n in ("runs", "facts", "failure_types"))
    frames = []
    for m in kept:
        frames.append(exposure_frame(runs[runs.model == m].drop(columns="model"),
                                     facts[facts.model == m].drop(columns="model"),
                                     ft[ft.model == m].drop(columns="model")).assign(model=m))
    df = pd.concat(frames).reset_index(drop=True)
    out = []

    def prow(section, quantity, n, est, lo=np.nan, hi=np.nan, note=""):
        out.append({"section": section, "quantity": quantity, "n": n, "estimate": round(float(est), 3),
                    "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3), "note": note})

    # 3.3 pooled within-task retrieval test
    for exp in EXPOSURES:
        diffs = {}
        strata = []
        for (m, t), g in df.groupby(["model", "task_id"]):
            strata.append({"model": m, "task_id": t, "n": len(g), "fail": g.fail.sum(),
                           "n_exposed": g[exp].sum(), "fail_exposed": (g[exp] & (g.fail == 1)).sum()})
            if g.fail.nunique() == 2:
                diffs.setdefault(t, []).append(g[g.fail == 0][exp].mean() - g[g.fail == 1][exp].mean())
        per_task = {t: np.mean(v) for t, v in diffs.items()}
        pos, neg = sum(v > 0 for v in per_task.values()), sum(v < 0 for v in per_task.values())
        p = binomtest(pos, pos + neg).pvalue
        prow(f"3.3 {exp}", "mixed (model,task) units", sum(len(v) for v in diffs.values()),
             sum(len(v) for v in diffs.values()))
        prow(f"3.3 {exp}", "tasks_with_mixed_units", len(per_task), len(per_task))
        prow(f"3.3 {exp}", "mean_task_diff_pass_minus_fail", len(per_task), np.mean(list(per_task.values())))
        prow(f"3.3 {exp}", "tasks_pass_higher", pos + neg, pos,
             note=f"fail_higher={neg}, ties={len(per_task) - pos - neg}; exact sign test p={p:.4f}")
        st = pd.DataFrame(strata)
        for label, sel in (("all strata", st.n > 0), ("mixed strata only", (st.fail > 0) & (st.fail < st.n))):
            s = st[sel]
            est, lo, hi, n = mantel_haenszel(s, s.task_id.values)
            prow(f"3.3 {exp}", "MH_risk_diff_fail_complete_minus_incomplete", n, est, lo, hi, note=label)

    # 3.4 how much is the task?
    cell = df.groupby(["task_id"]).reward.agg(["mean", "size"])
    n_tasks = len(cell)
    prow("3.4 task share", "tasks_failing_in_every_run_of_every_model", n_tasks, (cell["mean"] == 0).sum() / n_tasks,
         note=f"{(cell['mean'] == 0).sum()} of {n_tasks}")
    prow("3.4 task share", "tasks_passing_in_every_run_of_every_model", n_tasks, (cell["mean"] == 1).sum() / n_tasks,
         note=f"{(cell['mean'] == 1).sum()} of {n_tasks}")
    mt = df.groupby(["model", "task_id"]).reward.mean()
    prow("3.4 task share", "model_task_cells_failing_all_trials", len(mt), (mt == 0).mean())
    prow("3.4 task share", "model_task_cells_passing_all_trials", len(mt), (mt == 1).mean())
    prow("3.4 task share", "model_task_cells_mixed", len(mt), ((mt > 0) & (mt < 1)).mean())

    rr = pd.concat([runs[runs.model == m].assign(y=lambda d: (d.reward == 0).astype(int)) for m in kept]).reset_index(drop=True)
    rr["C"] = cross_model_prior(rr)  # task fail-prior from other models: pass rate -> use 1 - C as fail score
    rr["M"] = 1 - rr.groupby("model").reward.transform("mean")  # model fail rate
    rr["Cf"] = 1 - rr.C
    y, tasks = rr.y.values, rr.task_id.values
    Z = StandardScaler().fit_transform(rr[["Cf", "M"]])
    rr["B"] = LogisticRegression(max_iter=1000).fit(Z, y).predict_proba(Z)[:, 1]
    for name, col in (("task_only (other models' fail rate)", "Cf"), ("model_only (model fail rate)", "M"),
                      ("task+model (2-feature logistic, in-sample)", "B")):
        pt, dr = auc_point_and_draws(y, rr[col].values, tasks)
        prow("3.4 AUROC of priors", name, len(rr), pt, *ci(dr))
    return pd.DataFrame(out)


def main(which, analyze_only):
    global WORK, ALL
    WORK = Path("local/work-rescored" if which == "A" else "local/work")
    ALL = WORK / f"_all_{which}"
    kept = json.loads(Path(f"local/data/tau/kept_{which}.json").read_text(encoding="utf-8"))
    if not analyze_only:
        build_all(kept)
    Path("results").mkdir(exist_ok=True)
    pm = per_model_rows(kept)
    pm.to_csv(f"results/multi_model_per_model_{which}.csv", index=False)
    r3 = r3_rows(kept)
    r3.to_csv(f"results/multi_model_r3_{which}.csv", index=False)
    pooled = pooled_rows(kept)
    pooled.to_csv(f"results/multi_model_pooled_{which}.csv", index=False)
    print(pm[pm.section == "runs"].pivot(index="model", columns="quantity", values="estimate").to_string())
    print(r3[r3.model == "MEAN OF MODELS"].to_string(index=False))
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:2] != ["--set"] or sys.argv[2:3] not in (["A"], ["B"]):
        sys.exit("usage: python -m analysis.multi_model --set A|B [--analyze-only]")
    main(sys.argv[2], "--analyze-only" in sys.argv)
