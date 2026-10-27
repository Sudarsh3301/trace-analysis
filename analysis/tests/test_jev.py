from types import SimpleNamespace as NS

import pytest

from analysis.jev import BudgetStop, Jev, noul, payload_hash

Q = {"q": noul("Is it?", "yes", "no")}


def fake_client(counter):
    def make():
        def system_one(state, questions):
            counter.append(state)
            return NS(answers={"q": NS(type="noul", noul=0.7)}, usage=NS(input_tokens=100, output_tokens=2))
        return NS(system_one=system_one)
    return make


def test_cache_makes_reruns_free(tmp_path):
    calls = []
    j = Jev(max_calls=5, work=tmp_path, client_factory=fake_client(calls))
    assert j.ask({"a": 1}, Q)["nouls"] == {"q": 0.7}
    j2 = Jev(max_calls=0, work=tmp_path, client_factory=fake_client(calls))  # no budget, still served
    assert j2.ask({"a": 1}, Q)["nouls"] == {"q": 0.7}
    assert (len(calls), j.live, j2.live, j2.hits) == (1, 1, 0, 1)


def test_budget_stops_live_calls(tmp_path):
    calls = []
    j = Jev(max_calls=2, work=tmp_path, client_factory=fake_client(calls))
    res, stopped = j.run_all([({"i": i}, Q, "t", 0) for i in range(5)], workers=1)
    assert (len(calls), stopped, sum(r is not None for r in res)) == (2, 3, 2)
    with pytest.raises(BudgetStop):
        j.ask({"i": 99}, Q)


def test_dry_run_calls_nothing_and_writes_payload(tmp_path):
    calls = []
    j = Jev(dry_run=True, work=tmp_path, client_factory=fake_client(calls))
    assert j.ask({"a": 1}, Q) is None
    assert not calls and j.would == 1
    assert (tmp_path / "jev-dry" / f"{payload_hash({'a': 1}, Q)}.json").exists()


def test_log_has_tokens_and_cost(tmp_path):
    Jev(max_calls=1, work=tmp_path, client_factory=fake_client([])).ask({"a": 1}, Q, "tag")
    row = (tmp_path / "jev-log.csv").read_text().splitlines()[1].split(",")
    assert row[1:4:2] == ["tag", "ok"] and row[5] == "100" and float(row[7]) == pytest.approx(4.2e-6)


def test_rep_changes_hash():
    assert payload_hash({"a": 1}, Q, 0) != payload_hash({"a": 1}, Q, 1)
