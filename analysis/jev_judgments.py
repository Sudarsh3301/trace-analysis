"""Part B: do Jev step judgments add to the cross-model task prior C? (claim C2)

Usage: python -m analysis.jev_judgments --dry-run | --live --max-calls N [--sample N] | --analyze
Runs: kimi-k3 and claude-opus-5, 60 each (30 pass, 30 fail, one run per task first), first 6 agent
steps after the opening greeting (user turn >= 1), one request per step with the 8 JevScope
questions (lib/jev/questions.ts, MIT). State follows lib/jev/build-state.ts: task, currentStep,
recentHistory (8 events). `task` = the customer's first message; no ground truth is sent.
Repeat check: 20 steps x 3 requests (rep 0 is the main request). Set A labels.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import multi_model
from .boot import auc_point_and_draws, ci
from .jev import Jev, choice, clamp, noul, sample, summary
from .predict import NON_FEATURES, oof_r1

MODELS = ["kimi-k3_sierra_2026-08-04", "claude-opus-5_sierra_2026-08-04"]
WORK = Path("local/work-rescored")
N_STEPS, HISTORY, KS = 6, 8, ("2", "4")
SEED = 0

PHASES = {
    "exploring": "The agent is gathering information or building an understanding of the problem: reading files, inspecting code, searching, or asking questions. It is not yet changing anything.",
    "implementing": "The agent is making changes intended to accomplish the stated task: writing or editing files, adding code, or applying a fix.",
    "verifying": "The agent is testing or checking whether the work is correct: running tests, re-reading its own changes, or comparing output against what the task requires.",
    "recovering": "The agent is responding to an error, a failed attempt, or an unexpected result: reverting, retrying, diagnosing a failure, or changing approach after something did not work.",
    "finished": "The evidence in `currentStep` and `recentHistory` indicates the requested task has been completed AND checked, not merely asserted to be complete.",
    "unclear": "The current behavior does not clearly fit any of the other phases, or there is not enough evidence in the supplied state to tell.",
}
QUESTIONS = {
    "taskAlignment": noul(
        "An AI agent is working on the task in `task`. Considering the action in `currentStep`, does that action directly contribute to accomplishing the stated task?",
        "The action is a plausible and relevant part of doing the stated task, including necessary setup, inspection, or verification work that the task requires.",
        "The action is unrelated to the stated task, addresses a different problem, or is busywork that does not move the stated task forward."),
    "progress": noul(
        "An AI agent is working on the task in `task`. Does the action in `currentStep` represent meaningful progress toward completing that task, compared with the situation shown in `recentHistory`?",
        "The step advances the work: it produces new information the agent did not have, changes the system toward the goal, or confirms something that was previously unknown.",
        "The step leaves the agent in substantially the same position as before: it produces nothing new, or it undoes progress without replacing it."),
    "repetition": noul(
        "Compare the action in `currentStep` against the actions in `recentHistory`. Is the agent substantially repeating work it has already performed, without evidence in the state that the repetition is necessary?",
        "The same or an essentially equivalent action has already been performed in `recentHistory`, and nothing in the state explains why doing it again is warranted.",
        "The action is new, or it repeats earlier work for a reason the state supports, such as re-running a test after a change that could plausibly affect the result."),
    "stuck": noul(
        "Look at the sequence of actions in `recentHistory` together with `currentStep`. Does this sequence indicate that the agent is stuck, that is, cycling or repeatedly failing without making meaningful progress toward `task`?",
        "The recent sequence shows repeated attempts, reverts, or failures that keep returning to the same unresolved problem, with no sign of the agent's understanding or the system state improving.",
        "The recent sequence shows the agent moving forward: resolving problems, learning from results, or working through distinct parts of the task, even if individual steps failed."),
    "needsVerification": noul(
        "Looking at `currentStep` and `recentHistory`, is there a change the agent has made whose effect has not yet been observed? That is, is work sitting unverified at this moment?",
        "The most recent substantive change has not been followed by any observation of its effect: no test was run, no output was inspected, and no result was read back since it was made.",
        "No unverified change is outstanding. Either the most recent change was followed by an observed result, such as a test run or an inspected output, or the agent has not changed anything."),
    "prematureCompletion": noul(
        "In `currentStep`, is the agent asserting or signalling that `task` is finished, when the evidence in the supplied state does not establish that it is?",
        "The current step claims, states, or clearly implies that the work is done, and the supplied state does not support that: checks last seen were failing, required work is untouched, or no result confirms the outcome.",
        "The current step makes no claim that the work is finished, or it does claim so and the supplied evidence does support it."),
    "unexpectedDirection": noul(
        "An AI agent was given the task in `task`. Has it moved onto work that is not required for that task?",
        "The agent is doing work outside the scope of the stated task: refactoring unrelated code, fixing unrelated problems, or pursuing a goal the task did not ask for.",
        "The agent's work is within the scope of the stated task, including necessary supporting work the task implies."),
    "phase": choice(
        "An AI agent is working on the task in `task`. Considering `currentStep` in light of `recentHistory`, which phase best describes the agent's current behavior?",
        PHASES),
}
NOULS = [k for k in QUESTIONS if k != "phase"]


def select_runs(runs, n_per_class=30, seed=SEED):
    """30 pass + 30 fail; one run per task before any task repeats."""
    rng = random.Random(seed)
    keep = []
    for reward in (1, 0):
        by_task = {t: g.run_id.tolist() for t, g in runs[runs.reward == reward].groupby("task_id")}
        for ids in by_task.values():
            rng.shuffle(ids)
        tasks, picked, rnd = sorted(by_task), [], 0
        while len(picked) < n_per_class:
            rng.shuffle(tasks)
            new = [by_task[t][rnd] for t in tasks if len(by_task[t]) > rnd]
            if not new:
                break
            picked += new[:n_per_class - len(picked)]
            rnd += 1
        keep += picked
    return keep


def events_of(rows):
    """Message rows of one run -> events. Tool results attach to the latest event."""
    ev = []
    for r in rows:
        if r["role"] == "tool":
            if ev:
                ev[-1]["result"] = (ev[-1].get("result", "") + f"\n[{r['tool_name']}] {r['text']}").strip()
        elif r["role"] == "user" and not r["tool_name"]:
            ev.append({"type": "user_message", "content": r["text"] or "", "turn": r["user_turn"]})
        else:
            agent = r["role"] == "assistant"
            last = ev[-1] if ev else None
            if last and last.get("idx") == r["idx"]:  # parallel calls of one message
                last["tool"] += f", {r['tool_name']}"
                last["arguments"] += f"\n{r['tool_name']}: {r['tool_args']}"
                continue
            e = {"idx": r["idx"], "turn": r["user_turn"], "agent": agent,
                 "type": ("tool_call" if r["tool_name"] else "message") if agent else "user_tool_call",
                 "content": r["text"] or ""}
            if r["tool_name"]:
                e.update(tool=r["tool_name"], arguments=f"{r['tool_name']}: {r['tool_args']}")
            ev.append(e)
    return ev


def summarize(e, n):
    out = {"stepNumber": n, "type": e["type"], "content": clamp(e["content"] or e.get("tool", ""))}
    if e.get("tool"):
        out["tool"] = e["tool"]
    if e.get("result"):
        out["result"] = clamp(e["result"])
    return out


def run_steps(rows):
    """-> [(user_turn, state)] for the first N_STEPS agent steps after the greeting."""
    ev = events_of(rows)
    task = next((e["content"] for e in ev if e["type"] == "user_message"), "")
    steps = []
    for p, e in enumerate(ev):
        if not e.get("agent") or e["turn"] < 1:
            continue
        hist = [x for x in ev[:p] if (x["content"] or x.get("tool"))][-HISTORY:]
        dropped = len([x for x in ev[:p] if (x["content"] or x.get("tool"))]) - len(hist)
        cur = summarize(e, p + 1)
        if e.get("arguments"):
            cur["arguments"] = clamp(e["arguments"])
        st = {"task": clamp(task), "currentStep": cur,
              "recentHistory": [summarize(x, p - len(hist) + i + 1) for i, x in enumerate(hist)]}
        if dropped:
            st["historyNote"] = f"{dropped} earlier event(s) occurred before this window and are not shown."
        steps.append((e["turn"], st))
        if len(steps) == N_STEPS:
            break
    return steps


def build(models=MODELS):
    """-> (sel, items): sel[model] = (runs, facts, {run_id: [(turn, state)]})."""
    sel = {}
    for m in models:
        d = WORK / m
        runs, msgs = pd.read_parquet(d / "runs.parquet"), pd.read_parquet(d / "messages.parquet")
        ids = select_runs(runs)
        g = {rid: gr.sort_values("idx", kind="stable").to_dict("records") for rid, gr in msgs[msgs.run_id.isin(ids)].groupby("run_id")}
        sel[m] = (runs[runs.run_id.isin(ids)].reset_index(drop=True), pd.read_parquet(d / "facts.parquet"),
                  {rid: run_steps(g[rid]) for rid in ids})
    return sel


def main_items(sel):
    return [(st, QUESTIONS, f"B:{m}:{rid}:{i}", 0) for m, (_, _, steps) in sel.items()
            for rid, ss in steps.items() for i, (_, st) in enumerate(ss)]


def repeat_items(sel, n=20):
    base = main_items(sel)
    return [(st, q, tag, rep) for st, q, tag, _ in random.Random(SEED).sample(base, n) for rep in (1, 2)]


def step_features(sel, j):
    rows = []
    for m, (_, _, steps) in sel.items():
        for rid, ss in steps.items():
            for i, (turn, st) in enumerate(ss):
                a = j.peek(st, QUESTIONS)
                if a:
                    rows.append({"model": m, "run_id": rid, "step": i, "turn": turn, **a["nouls"],
                                 "p_recovering": a["choices"]["phase"]["probabilities"]["recovering"]})
    return pd.DataFrame(rows)


COLS = [*NOULS, "p_recovering"]


def j_matrix(steps_df, run_ids, k):
    seen = steps_df[steps_df.turn <= int(k)].groupby("run_id")[COLS].agg(["mean", "max"])
    seen.columns = [f"{a}_{b}" for a, b in seen.columns]
    return seen.reindex(run_ids).fillna(0).values.astype(float)


def flip_rows(sel, j):
    """Per question: share of repeated steps whose 3 answers do not agree (Noul side of 0.5; phase argmax)."""
    base = main_items(sel)
    steps = [it for it in random.Random(SEED).sample(base, 20)]
    out = {q: [] for q in QUESTIONS}
    for st, _, _, _ in steps:
        ans = [j.peek(st, QUESTIONS, rep) for rep in (0, 1, 2)]
        if not all(ans):
            continue
        for q in NOULS:
            v = [a["nouls"][q] for a in ans]
            out[q].append(len({x >= 0.5 for x in v}) > 1)
        out["phase"].append(len({max(a["choices"]["phase"]["probabilities"].items(), key=lambda kv: kv[1])[0] for a in ans}) > 1)
    return [{"section": "flip_rate", "model": "both", "k": "", "feature_set": q, "auroc": round(float(np.mean(v)), 3) if v else np.nan,
             "ci_low": np.nan, "ci_high": np.nan, "n_steps": len(v)} for q, v in out.items()]


def analyze(sel, j):
    allruns = pd.read_parquet(WORK / "_all_A" / "runs.parquet").reset_index(drop=True)
    allruns["C"] = multi_model.cross_model_prior(allruns)
    keys = np.unique(allruns.task_id)
    steps_df = step_features(sel, j)
    res, rows = {}, []
    for m, (runs, facts, _) in sel.items():
        y, tasks = (runs.reward == 0).astype(int).values, runs.task_id.values
        C = allruns[allruns.model == m].set_index("run_id").loc[runs.run_id].C.values[:, None]
        for k in KS:
            f = facts[facts.k == k].set_index("run_id").loc[runs.run_id]
            F = f[[c for c in f.columns if c not in NON_FEATURES]].fillna(0).values.astype(float)
            J = j_matrix(steps_df[steps_df.model == m], runs.run_id.values, k)
            for s, X in {"C": C, "C+F": np.hstack([C, F]), "C+J": np.hstack([C, J]), "C+F+J": np.hstack([C, F, J])}.items():
                res[(m, k, s)] = auc_point_and_draws(y, oof_r1(X, y, tasks), tasks, keys=keys)
    names = [*MODELS, "MEAN OF MODELS"]
    for k in KS:
        for name in names:
            ms = MODELS if name == "MEAN OF MODELS" else [name]
            for s in ("C", "C+F", "C+J", "C+F+J"):
                pt, dr = np.mean([res[(m, k, s)][0] for m in ms]), np.mean([res[(m, k, s)][1] for m in ms], axis=0)
                rows.append({"section": "auroc", "model": name, "k": k, "feature_set": s, "auroc": pt, "ci_low": ci(dr)[0], "ci_high": ci(dr)[1]})
            for base, plus in (("C", "C+J"), ("C+F", "C+F+J"), ("C", "C+F")):
                pt = np.mean([res[(m, k, plus)][0] - res[(m, k, base)][0] for m in ms])
                dr = np.mean([res[(m, k, plus)][1] - res[(m, k, base)][1] for m in ms], axis=0)
                rows.append({"section": "delta", "model": name, "k": k, "feature_set": f"{plus} - {base}", "auroc": pt,
                             "ci_low": ci(dr)[0], "ci_high": ci(dr)[1]})
    out = pd.DataFrame(rows + flip_rows(sel, j))
    out["auroc"] = out.auroc.astype(float).round(3)
    Path("results").mkdir(exist_ok=True)
    out.round(3).to_csv("results/c2_judgments.csv", index=False)
    print(f"{len(steps_df)} scored steps\n{out.to_string(index=False)}")


def main(argv):
    args = dict(zip(argv, argv[1:]))
    mode = next((a for a in ("--dry-run", "--live", "--analyze") if a in argv), None)
    if mode is None:
        sys.exit(__doc__)
    sel = build()
    j = Jev(dry_run=mode == "--dry-run", max_calls=int(args.get("--max-calls", 0)))
    if mode == "--analyze":
        return analyze(sel, j)
    main_, rep = main_items(sel), repeat_items(sel)
    uniq = {}
    for it in main_ + rep:
        uniq.setdefault((json.dumps(it[0], sort_keys=True), it[3]), it)
    print(f"{len(main_)} steps, {len(rep)} repeat requests, {len(uniq)} unique payloads")
    items = sample(main_ if "--sample" in args else list(uniq.values()), int(args["--sample"]) if "--sample" in args else None)
    _, stopped = j.run_all(items, int(args.get("--workers", 4)))
    print(summary(j, len(items)) + (f"; {stopped} not run (budget)" if stopped else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
