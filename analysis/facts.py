"""Per-run trace facts using only the first k user turns (k = 2, 4, 6, all).

Usage: python -m analysis.facts [work_dir]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .docs import doc_events
from .load import WRITE_TOOLS, sub_name

KS = [2, 4, 6, None]


def work_dirs(module):
    """Work dirs picked by --submission <name> or --all; otherwise print usage and exit 0."""
    a = sys.argv[1:]
    if a == ["--all"]:
        return sorted(d for d in Path("local/work").iterdir() if d.is_dir() and not d.name.startswith("_"))
    if len(a) == 2 and a[0] == "--submission":
        return [Path("local/work") / a[1]]
    print(f"usage: python -m analysis.{module} --submission <name> | --all")
    sys.exit(0)


def out_csv(stem, d, many):
    """results/<stem>.csv for one submission; <work dir>/<stem>.csv under --all."""
    return (d if many else Path("results")) / f"{stem}.csv"


def work_dir(arg=None):
    if arg:
        return Path(arg)
    dirs = [d for d in Path("local/work").iterdir() if d.is_dir()]
    if len(dirs) != 1:
        sys.exit("pass the work dir: several or no dirs under local/work")
    return dirs[0]


def is_transfer(r):
    return r["tool_name"] == "transfer_to_human_agents" or "transfer_to_human" in (
        sub_name(json.loads(r["tool_args"])) or "")


def run_facts(rows, required, k):
    sub = [r for r in rows if k is None or r["user_turn"] <= k]
    calls = [r for r in sub if r["role"] != "tool" and r["tool_name"]]
    by_name = Counter(r["tool_name"] for r in calls)
    ev = doc_events(sub)
    surf = sum(1 for d in required if ev.get(d, {}).get("surfaced") is not None)
    ctx = sum(1 for d in required if ev.get(d, {}).get("in_context") is not None)
    opened = sum(1 for d in required if ev.get(d, {}).get("opened") is not None)
    n = len(required)
    lens = [len(r["text"]) for r in sub if r["role"] == "assistant" and r["text"]]
    return {
        "k": "all" if k is None else str(k),
        "n_messages": len({r["idx"] for r in sub}),
        "n_agent_msgs": len({r["idx"] for r in sub if r["role"] == "assistant"}),
        "n_tool_calls": len(calls),
        "n_shell": by_name["shell"], "n_kb_bm25": by_name["KB_search_bm25"],
        "n_kb_dense": by_name["KB_search_dense"],
        "search_calls": by_name["shell"] + by_name["KB_search_bm25"] + by_name["KB_search_dense"],
        "write_calls": sum(v for t, v in by_name.items() if t in WRITE_TOOLS),
        "required_docs_total": n, "required_surfaced": surf, "required_in_context": ctx,
        "required_opened": opened,
        "required_surfaced_frac": surf / n if n else np.nan,
        "required_in_context_frac": ctx / n if n else np.nan,
        "required_opened_frac": opened / n if n else np.nan,
        "repeated_identical_calls": sum(
            c - 1 for c in Counter((r["tool_name"], r["tool_args"]) for r in calls).values()),
        "human_transfer_called": any(is_transfer(r) for r in calls if r["role"] == "assistant"),
        "agent_msg_len_mean": float(np.mean(lens)) if lens else 0.0,
        "agent_msg_len_max": max(lens, default=0),
        "agent_msg_len_slope": float(np.polyfit(range(len(lens)), lens, 1)[0]) if len(lens) > 1 else 0.0,
    }


def compute(runs, msgs):
    grouped = {rid: g.sort_values("idx", kind="stable").to_dict("records") for rid, g in msgs.groupby("run_id")}
    out = []
    for run in runs.itertuples():
        rows = grouped.get(run.run_id, [])
        for k in KS:
            out.append({"run_id": run.run_id, **run_facts(rows, list(run.required_docs), k)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    d = work_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    df = compute(pd.read_parquet(d / "runs.parquet"), pd.read_parquet(d / "messages.parquet"))
    df.to_parquet(d / "facts.parquet", index=False)
    print(f"{len(df)} fact rows -> {d / 'facts.parquet'}")
