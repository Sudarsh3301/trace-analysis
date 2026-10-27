"""Part A: silent failures. One Jev request per user message asks whether the user pushes back.

Usage: python -m analysis.jev_silent --dry-run | --live --max-calls N [--sample N] | --analyze
                                     | --handlabel | --validate
State per request: that user message and the preceding agent message with text. Set A labels.
--analyze writes results/c1_silent.csv; --handlabel writes local/work/pushback_handlabel.csv (for
hand labels, column `pushback` = 1/0) and a scores file; --validate writes results/c1_validation.csv.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .boot import ci, task_counts, task_sums
from .failure_type import FLAGS
from .jev import Jev, clamp, noul, sample, summary

MODELS = ["kimi-k3_sierra_2026-08-04", "claude-opus-5_sierra_2026-08-04",
          "gpt-5-5_sierra_2026-05-05", "qwen3-8-max_sierra_2026-08-04"]
WORK = Path("local/work-rescored")
# 0.8 is a post-hoc sensitivity chosen on the validation sample, not a pre-set threshold
THRESHOLDS = (0.5, 0.8)
PUSHBACK = ["corrects", "repeats", "complains", "asksHuman"]
KEYWORDS = re.compile(r"\bno\b|that'?s not|i said|human|agent", re.I)

QUESTIONS = {
    "corrects": noul(
        "In `userMessage`, does the user say that something the agent said or did in `agentMessage` is wrong, or ask the agent to change or redo it?",
        "The user disagrees with, corrects, or rejects what the agent said or did, or asks for it to be changed or done again.",
        "The user accepts what the agent said, answers a question, or moves on without disagreeing."),
    "repeats": noul(
        "Does `userMessage` repeat or restate a request already made in `previousUserMessage`, because the agent did not satisfy it in `agentMessage`?",
        "The user asks again for something they asked in `previousUserMessage`, because the agent ignored it or did not do it.",
        "The user makes a new request, answers the agent, or the message gives no sign of a repeated request."),
    "complains": noul(
        "Does the user in `userMessage` complain, or express frustration or dissatisfaction?",
        "The message shows annoyance, impatience, anger, or dissatisfaction with the agent or the service.",
        "The message is neutral, polite, or positive."),
    "asksHuman": noul(
        "Does the user in `userMessage` ask to speak to a human, a person, or a supervisor instead of the agent?",
        "The user asks for a human, a live person, a supervisor, or a manager.",
        "The user does not ask for a human."),
    "saysDone": noul(
        "Does the user in `userMessage` say that their request is done, or that they need nothing more?",
        "The user says the matter is finished, thanks the agent and ends the conversation, or says nothing more is needed.",
        "The user is still asking for something or giving information."),
}


def load_set(models=MODELS):
    out = {}
    for m in models:
        d = WORK / m
        out[m] = {n: pd.read_parquet(d / f"{n}.parquet") for n in ("runs", "messages", "action_checks", "failure_types")}
    return out


def user_texts(msgs):
    """-> [(run_id, idx, user_text, preceding agent text, previous user text)] for natural user messages
    (no ###CONTROL### tokens)."""
    out = []
    for rid, g in msgs.groupby("run_id"):
        agent = prev = ""
        for r in g.sort_values("idx", kind="stable").itertuples():
            if r.role == "assistant" and r.text:
                agent = r.text
            elif r.role == "user" and r.tool_name is None and r.text and not r.text.startswith("###"):
                out.append((rid, int(r.idx), r.text, agent, prev))
                prev = r.text
    return out


def state_for(user, agent, prev):
    return {"previousUserMessage": clamp(prev), "agentMessage": clamp(agent), "userMessage": clamp(user)}


def requests(data):
    """-> list of (model, run_id, idx, state)."""
    return [(m, rid, idx, state_for(u, a, p)) for m, t in data.items() for rid, idx, u, a, p in user_texts(t["messages"])]


def decisive_idx(msgs, checks, ft):
    """Message index of the first extra or wrong-argument write; the run's last index if none
    (also when the expected write was never attempted). -> (idx, kind)."""
    calls = msgs[(msgs.role != "tool") & msgs.tool_name.notna()]
    wrong = {(c.requestor, c["name"], c["sub"]) for _, c in checks[(checks.tool_type == "write") & ~checks.action_match].iterrows()}
    from .load import WRITE_TOOLS, sub_name
    cand = []
    if pd.notna(ft.first_unexpected_write_idx):
        cand.append((int(ft.first_unexpected_write_idx), "unexpected_write"))
    for r in calls[calls.tool_name.isin(WRITE_TOOLS)].itertuples():
        if (r.role, r.tool_name, sub_name(json.loads(r.tool_args))) in wrong:
            cand.append((int(r.idx), "wrong_write"))
            break
    return min(cand) if cand else (int(msgs.idx.max()), "end_of_run")


def score_table(data, j):
    """One row per user message with Jev's answers (rows without a cached answer are dropped)."""
    rows = []
    for m, rid, idx, state in requests(data):
        a = j.peek(state, QUESTIONS)
        if a:
            rows.append({"model": m, "run_id": rid, "idx": idx, **a["nouls"], "agent": state["agentMessage"],
                         "user": state["userMessage"]})
    df = pd.DataFrame(rows)
    df["pushback"] = df[PUSHBACK].max(axis=1)
    return df


def share_draws(silent, tasks, keys):
    n = np.ones(len(silent))
    s_t, n_t = task_sums(tasks, silent, n, keys=keys)
    _, _, counts = task_counts(tasks, keys=keys)
    with np.errstate(invalid="ignore", divide="ignore"):
        return silent.mean(), (counts @ s_t) / (counts @ n_t)


def share_ci(silent, tasks, keys):
    est, draws = share_draws(silent, tasks, keys)
    return est, *ci(draws)


def analyze(data, j, thresholds=THRESHOLDS):
    sc = score_table(data, j)
    keys = np.unique(np.concatenate([t["runs"].task_id.values for t in data.values()]))
    runs = []
    for m, t in data.items():
        by = {rid: g for rid, g in t["messages"].groupby("run_id")}
        ck = {rid: g for rid, g in t["action_checks"].groupby("run_id")}
        empty = t["action_checks"].iloc[:0]
        ft = t["failure_types"].set_index("run_id")
        s = sc[sc.model == m]
        for r in t["runs"].itertuples():
            if r.run_id not in by:  # a run with no messages at all has nothing to observe
                print(f"skipped {m} run {r.run_id}: no messages (reward {r.reward})")
                continue
            d, kind = decisive_idx(by[r.run_id], ck.get(r.run_id, empty), ft.loc[r.run_id].rename_axis("run_id"))
            u = s[s.run_id == r.run_id]
            f = ft.loc[r.run_id]
            runs.append({"model": m, "run_id": r.run_id, "task_id": r.task_id, "reward": r.reward, "kind": kind,
                         "n_user_scored": len(u),
                         **{f"{col}_{t}": int(not (x.pushback >= t).any()) for t in thresholds
                            for col, x in (("silent", u[u.idx < d]), ("silent_whole", u))},
                         **{c: bool(f[c]) for c in FLAGS}})
    df = pd.DataFrame(runs)
    groups = [("passed", lambda d: d.reward == 1), ("failed", lambda d: d.reward == 0)]
    groups += [(f"failed & {c}", lambda d, c=c: (d.reward == 0) & d[c]) for c in FLAGS]
    groups += [(f"failed & decisive={k}", lambda d, k=k: (d.reward == 0) & (d.kind == k))
               for k in ("unexpected_write", "wrong_write", "end_of_run")]
    out = []
    for t in thresholds:
        for name in ["ALL", *data]:
            d0 = df if name == "ALL" else df[df.model == name]
            for g, f in groups:
                d = d0[f(d0)]
                if len(d):
                    row = {"threshold": t, "model": name, "group": g, "n_runs": len(d)}
                    for col, label in ((f"silent_{t}", "before_decisive"), (f"silent_whole_{t}", "whole_run")):
                        est, lo, hi = share_ci(d[col].values.astype(float), d.task_id.values, keys)
                        row |= {f"share_no_pushback_{label}": round(est, 3), f"ci_low_{label}": round(lo, 3),
                                f"ci_high_{label}": round(hi, 3)}
                    out.append(row)
            # equally silent? failed minus passed, same task resamples
            row = {"threshold": t, "model": name, "group": "failed minus passed", "n_runs": len(d0)}
            for col, label in ((f"silent_{t}", "before_decisive"), (f"silent_whole_{t}", "whole_run")):
                (pf, df_), (pp, dp) = (share_draws(d0[d0.reward == r][col].values.astype(float),
                                                   d0[d0.reward == r].task_id.values, keys) for r in (0, 1))
                lo, hi = ci(df_ - dp)
                row |= {f"share_no_pushback_{label}": round(pf - pp, 3), f"ci_low_{label}": round(lo, 3),
                        f"ci_high_{label}": round(hi, 3)}
            out.append(row)
    res = pd.DataFrame(out)
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/c1_silent.csv", index=False)
    print(res.to_string(index=False))


def handlabel(data, j):
    sc = score_table(data, j)
    sc["any_noul"] = sc[list(QUESTIONS)].max(axis=1)
    hi, lo = sc[sc.any_noul >= 0.5], sc[sc.any_noul < 0.5]
    pick = pd.concat([hi.sample(min(30, len(hi)), random_state=0), lo.sample(min(30, len(lo)), random_state=0)])
    pick = pick.sample(frac=1, random_state=1).reset_index(drop=True)
    pick.insert(0, "sample_id", range(len(pick)))
    out = Path("local/work")
    pick[["sample_id", "agent", "user"]].assign(pushback="").to_csv(out / "pushback_handlabel.csv", index=False)
    pick.drop(columns=["agent", "user"]).to_csv(out / "pushback_handlabel_scores.csv", index=False)
    print(f"{len(pick)} messages ({min(30, len(hi))} with a Noul >= 0.5) -> {out / 'pushback_handlabel.csv'}")


def prf(pred, truth):
    tp = int((pred & truth).sum())
    return tp / max(pred.sum(), 1), tp / max(truth.sum(), 1)


def validate():
    lab = pd.read_csv("local/work/pushback_handlabel.csv").merge(
        pd.read_csv("local/work/pushback_handlabel_scores.csv").drop(columns=["pushback"], errors="ignore"),
        on="sample_id")
    lab = lab[lab.pushback.notna()]
    truth = lab.pushback.astype(int).values == 1
    rows = []
    for name, pred in (("jev pushback (max of 4 Nouls >= 0.5)", lab[PUSHBACK].max(axis=1) >= 0.5),
                       ("keyword baseline", lab.user.str.contains(KEYWORDS))):
        p, r = prf(pred.values, truth)
        rows.append({"method": name, "n": len(lab), "n_pushback": int(truth.sum()), "precision": round(p, 3), "recall": round(r, 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv("results/c1_validation.csv", index=False)


def main(argv):
    args = dict(zip(argv, argv[1:]))
    mode = next((a for a in ("--dry-run", "--live", "--analyze", "--handlabel", "--validate") if a in argv), None)
    if mode == "--validate":
        return validate()
    data = load_set()
    j = Jev(dry_run=mode == "--dry-run", max_calls=int(args.get("--max-calls", 0)))
    if mode in ("--analyze", "--handlabel"):
        return (analyze if mode == "--analyze" else handlabel)(data, j)
    if mode is None:
        sys.exit(__doc__)
    reqs = requests(data)
    uniq = {}
    for m, rid, idx, st in reqs:
        uniq.setdefault(json.dumps(st, sort_keys=True), (st, QUESTIONS, f"A:{m}:{rid}:{idx}", 0))
    items = sample(list(uniq.values()), int(args["--sample"]) if "--sample" in args else None)
    print(f"{len(reqs)} user messages, {len(uniq)} unique payloads; this run covers {len(items)}")
    for m in data:
        print(f"  {m}: {sum(1 for r in reqs if r[0] == m)} messages")
    _, stopped = j.run_all(items, int(args.get("--workers", 4)))
    print(summary(j, len(items)) + (f"; {stopped} not run (budget)" if stopped else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
