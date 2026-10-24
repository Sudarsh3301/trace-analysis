from pathlib import Path

import pandas as pd
import pytest

from analysis import facts, failure_type, load
from analysis.docs import canon

FIX = Path(__file__).parent / "fixtures"


def tables(name, tmp_path):
    out = load.load(FIX / name, tmp_path / name)
    return [pd.read_parquet(out / f"{n}.parquet") for n in ("runs", "messages", "action_checks")]


def fact(df, k):
    return df[df.k == k].iloc[0]


def test_canon_both_styles():
    a = canon("doc_bank_accounts_bank_accounts_(general)_032")
    assert a == canon("./doc_bank_accounts_bank_accounts__general__032.md")
    assert canon("DOC_X_1.json") == "doc_x_1"


def test_loader_runs(tmp_path):
    runs, msgs, _ = tables("fail.json", tmp_path)
    r = runs.iloc[0]
    assert (r.reward, r.n_user_msgs, r.n_tool_calls) == (0, 3, 6)
    assert list(r.required_docs)[0] == "doc_bank_accounts_bank_accounts__general__032"
    # two calls in one message -> two rows, text only on the first
    two = msgs[(msgs.idx == 2) & (msgs.role == "assistant")]
    assert list(two.tool_name) == ["KB_search_bm25", "shell"] and two.text.notna().tolist() == [True, False]
    # user turn counts user messages up to and including the row
    assert msgs[msgs.idx == 7].user_turn.iloc[0] == 2


def test_facts_k_truncation(tmp_path):
    runs, msgs, _ = tables("fail.json", tmp_path)
    f = facts.compute(runs, msgs)
    k2, allk = fact(f, "2"), fact(f, "all")
    # doc 032 is surfaced at idx 4 and opened only after user turn 2; the gold doc came in a
    # KB result the agent never cites: in context, not opened
    assert (k2.required_surfaced, k2.required_in_context, k2.required_opened) == (2, 1, 0)
    assert (allk.required_surfaced, allk.required_in_context, allk.required_opened) == (2, 2, 1)
    assert k2.repeated_identical_calls == 1 and k2.write_calls == 2 and k2.search_calls == 3
    assert allk.n_tool_calls == 6 and not allk.human_transfer_called


def test_facts_no_leakage(tmp_path):
    runs, msgs, _ = tables("fail.json", tmp_path)
    before = fact(facts.compute(runs, msgs), "2")
    after = fact(facts.compute(runs, msgs[msgs.user_turn <= 2]), "2")
    pd.testing.assert_series_equal(before, after)


def test_failure_flags(tmp_path):
    runs, msgs, checks = tables("fail.json", tmp_path)
    f = failure_type.compute(runs, msgs, checks).iloc[0]
    assert f.expected_write_never_attempted and f.unexpected_write
    assert f.required_doc_not_opened and f.required_doc_not_in_context
    assert f.nonmatching_expected_actions == 2
    assert f.first_unexpected_write_idx == 10  # the change_user_email call


def test_passing_run_with_nonmatching_action(tmp_path):
    runs, msgs, checks = tables("pass.json", tmp_path)
    f = failure_type.compute(runs, msgs, checks).iloc[0]
    assert f.reward == 1 and f.nonmatching_expected_actions == 1
    assert not f.expected_write_never_attempted and not f.unexpected_write
    a = fact(facts.compute(runs, msgs), "all")
    assert (a.required_in_context, a.required_opened, a.write_calls, a.n_tool_calls) == (1, 0, 1, 2)


def test_flag_rates(tmp_path):
    ft = pd.concat([failure_type.compute(*tables(n, tmp_path)) for n in ("pass.json", "fail.json")])
    r = failure_type.flag_rates(ft).set_index("flag")
    assert r.loc["unexpected_write", "rate_fail"] == 1 and r.loc["unexpected_write", "rate_pass"] == 0
    assert r.loc["nonmatching_expected_actions", "rate_pass"] == 1


def test_group_kfold_never_splits_a_task():
    from analysis import predict
    groups = [f"t{i // 4}" for i in range(40)]  # 10 tasks x 4 trials
    for tr, te in predict.r1_folds(groups):
        assert not {groups[i] for i in tr} & {groups[i] for i in te}
