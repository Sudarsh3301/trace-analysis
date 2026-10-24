"""Baselines: predict run failure from task, prior and trace features (no model calls).

R1 = new tasks (GroupKFold by task). R2 = new trial of a seen task (leave-one-trial-out, prior =
mean reward of the task's other trials). AUROC CI: task-clustered bootstrap.

Usage: python -m analysis.predict [work_dir]   -> results/baselines.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .boot import auc_point_and_draws, ci
from .facts import work_dir

AREAS = ["bank_accounts", "business_checking_accounts", "business_credit_cards",
         "business_savings_accounts", "checking_accounts", "credit_cards", "customer_support",
         "savings_accounts"]
NON_FEATURES = {"run_id", "k", "reward", "task_id", "trial"}


def area(doc):
    return next(a for a in AREAS if doc.startswith("doc_" + a + "_"))


def task_features(runs, checks):
    """One row per run: n_required_docs, n_expected_writes, share of required docs per area."""
    w = checks[checks.tool_type == "write"].groupby("run_id").size()
    t = pd.DataFrame({"run_id": runs.run_id, "n_required_docs": runs.required_docs.map(len),
                      "n_expected_writes": runs.run_id.map(w).fillna(0).values})
    for a in AREAS:
        t["area_" + a] = [np.mean([area(d) == a for d in docs]) for docs in runs.required_docs]
    return t


def r1_folds(groups, n_splits=5):
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def oof_r1(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in r1_folds(groups):
        pred[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return pred


def task_prior(df, exclude_trial=None):
    """Mean reward of each run's task's other trials (also excluding `exclude_trial` if given)."""
    out = np.zeros(len(df))
    for i, (tid, tr) in enumerate(zip(df.task_id, df.trial)):
        o = df[(df.task_id == tid) & (df.trial != tr) & (df.trial != exclude_trial)]
        out[i] = o.reward.mean() if len(o) else df.reward.mean()  # no other trial: overall pass rate
    return out


def oof_r2(X, y, df, has_prior):
    """Leave-one-trial-out: fold j holds out trial j of every task. The prior feature (last column
    of X when present) is recomputed per fold without the held-out trial for training rows."""
    pred = np.zeros(len(y))
    for j in sorted(df.trial.unique()):
        te = (df.trial == j).values
        tr = ~te
        Xf = X.copy()
        if has_prior:
            Xf[:, 0] = np.where(te, task_prior(df), task_prior(df, exclude_trial=j))
        pred[te] = model().fit(Xf[tr], y[tr]).predict_proba(Xf[te])[:, 1]
    return pred


def boot_auc(y, pred, tasks):
    """AUROC and 95% CI over task-clustered bootstrap resamples."""
    auc, draws = auc_point_and_draws(y, pred, tasks)
    return auc, *ci(draws)


def build(work):
    runs, checks = (pd.read_parquet(work / f"{n}.parquet") for n in ("runs", "action_checks"))
    facts = pd.read_parquet(work / "facts.parquet")
    runs = runs.assign(y=(runs.reward == 0).astype(int)).reset_index(drop=True)
    return runs, task_features(runs, checks), facts


def baseline_rows(runs, T, facts, ks=("2", "4", "6", "all"), only=None):
    """AUROC rows for every regime/feature set; `only` = {(regime, set), ...} limits the work."""
    y, tasks = runs.y.values, runs.task_id.values
    T_cols = [c for c in T.columns if c != "run_id"]
    P = task_prior(runs)[:, None]
    Tm = T.set_index("run_id").loc[runs.run_id][T_cols].values.astype(float)
    rows = []
    for k in ks:
        f = facts[facts.k == k].set_index("run_id").loc[runs.run_id]
        F = f[[c for c in f.columns if c not in NON_FEATURES]].fillna(0).values.astype(float)
        L = f[["n_messages", "n_tool_calls"]].values.astype(float)
        sets = {"R1": {"T": Tm, "L": L, "F": F, "T+L": np.hstack([Tm, L]), "T+F": np.hstack([Tm, F])},
                "R2": {"P": P, "L": L, "F": F, "P+L": np.hstack([P, L]), "P+F": np.hstack([P, F])}}
        for regime, fs in sets.items():
            for name, X in fs.items():
                if only and (regime, name) not in only:
                    continue
                pred = oof_r1(X, y, tasks) if regime == "R1" else oof_r2(X, y, runs, name.startswith("P"))
                auc, lo, hi = boot_auc(y, pred, tasks)
                rows.append({"regime": regime, "k": k, "feature_set": name, "auroc": round(auc, 3),
                             "ci_low": round(lo, 3), "ci_high": round(hi, 3), "n_runs": len(y)})
    return rows


def main(work):
    runs, T, facts = build(work)
    out = pd.DataFrame(baseline_rows(runs, T, facts))
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/baselines.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(work_dir(sys.argv[1] if len(sys.argv) > 1 else None))
