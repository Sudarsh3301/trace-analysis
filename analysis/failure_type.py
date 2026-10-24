"""Failure flags for every run (pass and fail), plus pass/fail base rates per flag.

Flags come from reward_info.action_checks. They do not decide the reward, so a flag only
describes failures if its rate is clearly higher in failed runs (see results/failure_flag_rates.csv).

Usage: python -m analysis.failure_type --submission <name> | --all
Writes failure_types.parquet in the work dir; results/failure_flag_rates.csv (--all: <work dir>/failure_flag_rates.csv).
"""
import json
import sys
from pathlib import Path

import pandas as pd

from .docs import doc_events
from .facts import out_csv, work_dirs
from .load import WRITE_TOOLS, sub_name

FLAGS = ["expected_write_never_attempted", "nonmatching_expected_actions", "unexpected_write",
         "required_doc_not_opened", "required_doc_not_in_context"]


def run_flags(rows, required, checks):
    """checks: action-check dicts of this run."""
    wanted = [c for c in checks if c["tool_type"] == "write"]
    first_call = {}  # (requestor, name, sub) -> first message idx
    for r in rows:
        if r["role"] != "tool" and r["tool_name"] in WRITE_TOOLS:
            k = (r["role"], r["tool_name"], sub_name(json.loads(r["tool_args"])))
            first_call.setdefault(k, r["idx"])
    keys = {(c["requestor"], c["name"], c["sub"]) for c in wanted}
    unexpected = [i for k, i in first_call.items() if k not in keys]
    ev = doc_events(rows)
    return {
        "expected_write_never_attempted": any(k not in first_call for k in keys),
        # diagnostic; also common in passing runs
        "nonmatching_expected_actions": sum(not c["action_match"] for c in checks),
        "unexpected_write": bool(unexpected),
        "required_doc_not_opened": any(ev.get(d, {}).get("opened") is None for d in required),
        "required_doc_not_in_context": any(ev.get(d, {}).get("in_context") is None for d in required),
        "first_unexpected_write_idx": min(unexpected) if unexpected else None,
    }


def compute(runs, msgs, checks):
    grouped = {rid: g.sort_values("idx", kind="stable").to_dict("records") for rid, g in msgs.groupby("run_id")}
    cks = {rid: g.to_dict("records") for rid, g in checks.groupby("run_id")}
    out = [{"run_id": r.run_id, "reward": r.reward,
            **run_flags(grouped.get(r.run_id, []), list(r.required_docs), cks.get(r.run_id, []))}
           for r in runs.itertuples()]
    return pd.DataFrame(out)


def flag_rates(df):
    """Rate of each flag among passing and failed runs (count flags are read as > 0)."""
    p, f = df[df.reward == 1], df[df.reward == 0]
    rows = [{"flag": c, "rate_pass": (p[c] > 0).mean(), "rate_fail": (f[c] > 0).mean(),
             "n_pass": len(p), "n_fail": len(f)} for c in FLAGS]
    out = pd.DataFrame(rows)
    out.insert(3, "difference", out.rate_fail - out.rate_pass)
    return out


if __name__ == "__main__":
    for d in work_dirs("failure_type"):
        df = compute(*(pd.read_parquet(d / f"{n}.parquet") for n in ("runs", "messages", "action_checks")))
        df.to_parquet(d / "failure_types.parquet", index=False)
        Path("results").mkdir(exist_ok=True)
        rates = flag_rates(df)
        rates.round(3).to_csv(out_csv("failure_flag_rates", d, sys.argv[1] == "--all"), index=False)
        print(f"{len(df)} runs -> {d / 'failure_types.parquet'}\n{rates.round(3).to_string(index=False)}")
