"""Export the study data for the web app. Makes no Jev call: Jev answers come from the cache.

Usage: python -m analysis.export_site
Writes, from the set A tables (local/work-rescored/) and results/:
  public/data/study.json      models, tasks, pass counts per model and task, and the result tables
  public/data/runs.json       one row per run: outcome, trace facts, failure flags, Jev summaries
  local/site-data/runs/<run_id>.json   conversation text, decisive point, Jev answers per message
The first two are derived numbers and are committed. The per-run files hold benchmark text and
are published as a dataset instead (local/ is not committed).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from . import jev_judgments as jj
from . import jev_silent as js
from .c3_retrieval import EXPOSURES, exposure_frame
from .failure_type import FLAGS
from .jev import Jev, clamp
from .multi_model import cross_model_prior

WORK = Path("local/work-rescored")
PUBLIC = Path("public/data")
CONV = Path("local/site-data/runs")
TOOL_TEXT = 1500  # characters kept per tool result and tool arguments; policy documents are long
FACTS = ["n_tool_calls", "search_calls", "write_calls", "required_docs_total", "required_surfaced",
         "required_in_context", "required_opened", "repeated_identical_calls", "human_transfer_called"]
TABLES = {"auroc_prior": "multi_model_r3_A", "auroc_jev": "c2_judgments", "silent": "c1_silent",
          "window": "c1_window_check", "retrieval": "multi_model_pooled_A", "validation": "c1_validation",
          "jev_usage": "jev_usage"}


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def records(df):
    return json.loads(df.to_json(orient="records"))  # NaN -> null


def prior_check(runs):
    """Pooled AUROC of the cross-model task prior, with and without the tasks every run fails."""
    r = runs.reset_index(drop=True).copy()
    r["C"], r["y"] = cross_model_prior(r), (r.reward == 0).astype(int)
    always = r.groupby("task_id").reward.max().eq(0)
    keep = ~r.task_id.map(always)
    out = {"always_fail_tasks": int(always.sum()),
           "auroc_all": round(roc_auc_score(r.y, 1 - r.C), 3), "n_all": len(r),
           "auroc_without_always_fail": round(roc_auc_score(r.y[keep], 1 - r.C[keep]), 3), "n_without": int(keep.sum())}
    assert out["auroc_all"] == 0.838, out  # the pooled number in the README
    return out


def retrieval_tasks(runs, facts, ft, models, pooled):
    """Per task with mixed outcomes: mean over (model, task) units of passed-minus-failed share with the
    exposure. Same computation as multi_model.pooled_rows; checked against its published counts."""
    df = pd.concat([exposure_frame(runs[runs.model == m].drop(columns="model"), facts[facts.model == m].drop(columns="model"),
                                   ft[ft.model == m].drop(columns="model")).assign(model=m) for m in models])
    out = {}
    for exp in EXPOSURES:
        diffs = {}
        for (_, t), g in df.groupby(["model", "task_id"]):
            if g.fail.nunique() == 2:
                diffs.setdefault(t, []).append(g[g.fail == 0][exp].mean() - g[g.fail == 1][exp].mean())
        rows = [{"task": t, "diff": round(float(np.mean(v)), 4), "units": len(v)} for t, v in sorted(diffs.items())]
        pub = pooled[(pooled.section == f"3.3 {exp}") & (pooled.quantity == "tasks_pass_higher")].iloc[0]
        pos, neg = sum(r["diff"] > 0 for r in rows), sum(r["diff"] < 0 for r in rows)
        assert (pos, neg) == (int(pub.estimate), int(pub.note.split("fail_higher=")[1].split(",")[0])), (exp, pos, neg)
        out[exp] = rows
    return out


def validation_pairs():
    """The 60 validation messages: Jev pushback score and the separate LLM's label. No message text."""
    lab = pd.read_csv("local/work/pushback_handlabel.csv")[["sample_id", "pushback"]].rename(columns={"pushback": "label"})
    sc = pd.read_csv("local/work/pushback_handlabel_scores.csv")[["sample_id", "model", "run_id", "idx", "pushback"]]
    df = sc.merge(lab, on="sample_id").dropna(subset=["label"])
    return [{"run_id": r.run_id, "idx": int(r.idx), "model": r.model, "score": round(float(r.pushback), 4),
             "label": int(r.label)} for r in df.itertuples()]


def conversation(run, msgs, checks, ft_row):
    """One run -> dict with clamped message rows and the decisive point (no Jev fields)."""
    msgs = msgs.sort_values("idx", kind="stable")
    d, kind = js.decisive_idx(msgs, checks, ft_row)
    rows = []
    for r in msgs.itertuples():
        long = r.role == "tool"
        rows.append({"idx": int(r.idx), "role": r.role, "turn": int(r.user_turn),
                     "text": clamp(r.text, TOOL_TEXT) if long else r.text,
                     "tool": r.tool_name, "args": clamp(r.tool_args, TOOL_TEXT) if r.tool_args else None})
    return {"run_id": run.run_id, "task_id": run.task_id, "trial": int(run.trial), "reward": int(run.reward),
            "decisive": {"idx": d, "kind": kind}, "messages": rows}


def step_answers(msgs, j):
    """Jev step judgments of one run, keyed by message idx. Same states as jev_judgments.build."""
    rows = msgs.sort_values("idx", kind="stable").to_dict("records")
    ev, out = jj.events_of(rows), {}
    for _, st in jj.run_steps(rows):
        a = j.peek(st, jj.QUESTIONS)
        if a:
            out[int(ev[st["currentStep"]["stepNumber"] - 1]["idx"])] = {
                **{k: round(v, 4) for k, v in a["nouls"].items()},
                "phase": a["choices"]["phase"]["choice"]}
    return out


def main():
    kept = json.loads(Path("local/data/tau/kept_A.json").read_text(encoding="utf-8"))
    runs = pd.read_parquet(WORK / "_all_A" / "runs.parquet")
    facts_all = pd.read_parquet(WORK / "_all_A" / "facts.parquet")
    facts = facts_all[facts_all.k == "all"].set_index("run_id")[FACTS]
    ft_all = pd.read_parquet(WORK / "_all_A" / "failure_types.parquet")
    ft = ft_all.set_index("run_id")

    models = list(kept)
    tasks = runs.groupby("task_id").reward.mean().sort_values()  # hardest first
    passes = runs.pivot_table(index="model", columns="task_id", values="reward", aggfunc="sum").reindex(index=models, columns=tasks.index)
    trials = runs.pivot_table(index="model", columns="task_id", values="reward", aggfunc="size").reindex(index=models, columns=tasks.index)
    study = {
        "models": [{"key": m, "label": kept[m]["model"], "pass_rate": round(runs[runs.model == m].reward.mean(), 4),
                    "jev": m in js.MODELS} for m in models],
        "tasks": [{"id": t, "pass_rate": round(p, 4)} for t, p in tasks.items()],
        "passes": passes.fillna(0).astype(int).values.tolist(),
        "trials": trials.fillna(0).astype(int).values.tolist(),
        "tables": {k: records(pd.read_csv(f"results/{f}.csv")) for k, f in TABLES.items()},
        "prior_check": prior_check(runs),
        "retrieval_tasks": retrieval_tasks(runs, facts_all, ft_all, models, pd.read_csv("results/multi_model_pooled_A.csv")),
        "validation": validation_pairs(),
    }
    print("prior check:", study["prior_check"], "| validation messages:", len(study["validation"]))
    dump(study, PUBLIC / "study.json")

    # Jev: pushback per user message and step judgments, 4 models, cache only
    j = Jev()
    data = js.load_set()
    pb = js.score_table(data, j)
    pb_by_run = {rid: g for rid, g in pb.groupby("run_id")}
    jev_cols = {}  # run_id -> summary columns
    actor = {}  # run_id -> who made the decisive write: agent, customer, or none (end of run)
    kinds = {}  # run_id -> decisive point kind
    n_steps = 0
    for m in models:
        d = WORK / m
        msgs = pd.read_parquet(d / "messages.parquet")
        checks = pd.read_parquet(d / "action_checks.parquet")
        by_msgs = {rid: g for rid, g in msgs.groupby("run_id")}
        by_checks = {rid: g for rid, g in checks.groupby("run_id")}
        for r in runs[runs.model == m].itertuples():
            if r.run_id not in by_msgs:
                continue  # no messages: nothing to show
            conv = conversation(r, by_msgs[r.run_id], by_checks.get(r.run_id, checks.iloc[:0]),
                                ft.loc[r.run_id].rename_axis("run_id"))
            conv["model"] = m
            dec = conv["decisive"]
            roles = {x["role"] for x in conv["messages"] if x["idx"] == dec["idx"] and x["role"] != "tool"}
            actor[r.run_id] = "none" if dec["kind"] == "end_of_run" else "customer" if roles == {"user"} else "agent"
            kinds[r.run_id] = dec["kind"]
            if m in js.MODELS:
                steps = step_answers(by_msgs[r.run_id], j)
                n_steps += len(steps)
                u = pb_by_run.get(r.run_id, pb.iloc[:0])
                conv["judgments"] = {str(k): v for k, v in steps.items()}
                conv["pushback"] = {str(int(x.idx)): {**{k: round(getattr(x, k), 4) for k in js.QUESTIONS},
                                                      "pushback": round(x.pushback, 4)} for x in u.itertuples()}
                before = u[u.idx < conv["decisive"]["idx"]]
                jev_cols[r.run_id] = {
                    "stuck_max": max((s["stuck"] for s in steps.values()), default=None),
                    "premature_max": max((s["prematureCompletion"] for s in steps.values()), default=None),
                    "pushback_max_before": round(before.pushback.max(), 4) if len(before) else 0.0}
            dump(conv, CONV / f"{r.run_id}.json")
        print(f"{m}: done")

    model_i = {m: i for i, m in enumerate(models)}
    task_i = {t: i for i, t in enumerate(tasks.index)}
    table = runs[["run_id", "model", "task_id", "trial", "reward", "n_messages", "n_user_msgs"]].copy()
    table["model"], table["task_id"] = table.model.map(model_i), table.task_id.map(task_i)
    table = table.join(facts, on="run_id").join(ft[FLAGS].astype(int), on="run_id")
    table = table.join(pd.DataFrame.from_dict(jev_cols, orient="index"), on="run_id")
    table["decisive_kind"], table["decisive_actor"] = table.run_id.map(kinds), table.run_id.map(actor)
    split = json.loads(table.to_json(orient="split", index=False))  # NaN -> null, numpy -> plain
    dump({"columns": split["columns"], "rows": split["data"]}, PUBLIC / "runs.json")
    print(f"{len(table)} runs, {len(jev_cols)} with Jev, {n_steps} judged steps, {len(pb)} scored user messages")
    for p in (PUBLIC / "study.json", PUBLIC / "runs.json"):
        print(f"{p}: {p.stat().st_size / 1e6:.2f} MB")
    sizes = [p.stat().st_size for p in CONV.glob("*.json")]
    print(f"{CONV}: {len(sizes)} files, {sum(sizes) / 1e6:.1f} MB, largest {max(sizes) / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
