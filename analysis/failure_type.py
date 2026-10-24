"""Failure flags for failed runs (reward 0), from reward_info.action_checks.

Usage: python -m analysis.failure_type [work_dir]
"""
import json
import sys

import pandas as pd

from .docs import doc_events
from .facts import work_dir
from .load import WRITE_TOOLS, sub_name


def run_flags(rows, required, checks):
    """checks: action-check dicts of this run; only the write ones count."""
    wanted = [c for c in checks if c["tool_type"] == "write"]
    first_call = {}  # (requestor, name, sub) -> first message idx
    for r in rows:
        if r["role"] != "tool" and r["tool_name"] in WRITE_TOOLS:
            k = (r["role"], r["tool_name"], sub_name(json.loads(r["tool_args"])))
            first_call.setdefault(k, r["idx"])
    keys = [(c["requestor"], c["name"], c["sub"]) for c in wanted]
    missing = any(k not in first_call for k in keys)
    wrong = [first_call[k] for k, c in zip(keys, wanted) if k in first_call and not c["action_match"]]
    extra = [i for k, i in first_call.items() if k not in keys]
    ev = doc_events(rows)
    bad = wrong + extra
    return {"missing_write": missing, "wrong_args": bool(wrong), "extra_write": bool(extra),
            "required_doc_never_read": any(ev.get(d, {}).get("read") is None for d in required),
            "first_bad_write_idx": min(bad) if bad else None}


def compute(runs, msgs, checks):
    grouped = {rid: g.sort_values("idx", kind="stable").to_dict("records") for rid, g in msgs.groupby("run_id")}
    cks = {rid: g.to_dict("records") for rid, g in checks.groupby("run_id")}
    out = [{"run_id": r.run_id,
            **run_flags(grouped.get(r.run_id, []), list(r.required_docs), cks.get(r.run_id, []))}
           for r in runs[runs.reward == 0].itertuples()]
    return pd.DataFrame(out)


if __name__ == "__main__":
    d = work_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    df = compute(*(pd.read_parquet(d / f"{n}.parquet") for n in ("runs", "messages", "action_checks")))
    df.to_parquet(d / "failure_types.parquet", index=False)
    print(f"{len(df)} failed runs -> {d / 'failure_types.parquet'}")
