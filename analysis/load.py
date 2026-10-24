"""Load a tau-bench results JSON into parquet tables.

Usage: python -m analysis.load <results.json> [out_dir]
out_dir defaults to local/work/<name of the input's folder>.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from .docs import canon

WRITE_TOOLS = {
    "call_discoverable_agent_tool", "log_verification", "change_user_email",
    "call_discoverable_user_tool", "apply_for_credit_card",
    "request_human_agent_transfer", "submit_referral", "submit_transaction",
}


def sub_name(args):
    """Wrapper tools hide the real tool name in their arguments."""
    a = args if isinstance(args, dict) else {}
    return a.get("agent_tool_name") or a.get("discoverable_tool_name")


def load_sim(sim, required):
    """-> (run row, message rows, action-check rows). Tool-call rows repeat the message idx;
    text is set on the first row of a message only. Tool-result rows carry the answered tool's name."""
    names, msgs, user_turn = {}, [], 0
    for idx, m in enumerate(sim["messages"]):
        if m["role"] == "user":
            user_turn += 1
        base = {"run_id": sim["id"], "idx": idx, "role": m["role"], "user_turn": user_turn}
        text = m.get("content") or ""
        calls = m.get("tool_calls") or []
        if m["role"] == "tool":
            msgs.append({**base, "text": text, "tool_name": names.get(m.get("id")), "tool_args": None})
        elif not calls:
            msgs.append({**base, "text": text, "tool_name": None, "tool_args": None})
        for i, c in enumerate(calls):
            names[c["id"]] = c["name"]
            msgs.append({**base, "text": text if i == 0 else None, "tool_name": c["name"],
                         "tool_args": json.dumps(c["arguments"], sort_keys=True)})
    ri = sim["reward_info"]
    checks = []
    for i, a in enumerate(ri.get("action_checks") or []):
        act = a["action"]
        checks.append({"run_id": sim["id"], "action_idx": i, "requestor": act["requestor"],
                       "name": act["name"], "sub": sub_name(act["arguments"]),
                       "action_match": bool(a["action_match"]), "tool_type": a["tool_type"]})
    run = {"run_id": sim["id"], "task_id": sim["task_id"], "trial": sim["trial"],
           "reward": int(round(ri["reward"])), "n_messages": len(sim["messages"]),
           "n_user_msgs": user_turn,
           "n_tool_calls": sum(1 for r in msgs if r["role"] != "tool" and r["tool_name"]),
           "required_docs": [canon(d) for d in required]}
    return run, msgs, checks


def load(path, out_dir=None):
    path = Path(path)
    out = Path(out_dir) if out_dir else Path("local/work") / path.parent.name
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    req = {t["id"]: t.get("required_documents") or [] for t in data["tasks"]}
    runs, msgs, checks = [], [], []
    # runs without a reward (infrastructure errors) cannot be scored: drop them, and say so
    scored = [s for s in data["simulations"] if (s.get("reward_info") or {}).get("reward") is not None]
    dropped = len(data["simulations"]) - len(scored)
    (out / "meta.json").write_text(json.dumps({"n_simulations": len(data["simulations"]), "dropped_no_reward": dropped}))
    for sim in scored:
        r, m, c = load_sim(sim, req.get(sim["task_id"], []))
        runs.append(r), msgs.extend(m), checks.extend(c)
    for name, rows in (("runs", runs), ("messages", msgs), ("action_checks", checks)):
        pd.DataFrame(rows).to_parquet(out / f"{name}.parquet", index=False)
    print(f"{len(runs)} runs ({dropped} dropped: no reward), {len(msgs)} message rows -> {out}")
    return out


if __name__ == "__main__":
    load(*sys.argv[1:3])
