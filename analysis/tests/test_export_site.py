from pathlib import Path

import pandas as pd

from analysis import failure_type, load
from analysis.export_site import TOOL_TEXT, conversation

FIX = Path(__file__).parent / "fixtures"


def test_conversation_keeps_every_message_and_clamps_tool_text(tmp_path):
    out = load.load(FIX / "fail.json", tmp_path / "fail")
    runs, msgs, checks = (pd.read_parquet(out / f"{n}.parquet") for n in ("runs", "messages", "action_checks"))
    ft = failure_type.compute(runs, msgs, checks).set_index("run_id")
    run = next(runs.itertuples())
    conv = conversation(run, msgs, checks, ft.loc[run.run_id].rename_axis("run_id"))
    assert len(conv["messages"]) == len(msgs)
    assert conv["reward"] == 0 and conv["decisive"]["idx"] <= msgs.idx.max()
    assert all(len(m["text"] or "") <= TOOL_TEXT for m in conv["messages"] if m["role"] == "tool")
