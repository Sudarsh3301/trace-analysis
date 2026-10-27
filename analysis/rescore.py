"""Re-score one trajectory file with tau2 v1.0.1 grading code, fresh tasks, BM25-only retrieval
variant (the official CLI builds the OpenAI-embedding 'alltools' env and needs an API key).

Run with the tau2 environment (tau2-bench v1.0.1, Python 3.12), not the analysis environment:
    local/tau2-env/Scripts/python -m analysis.rescore <trajectory.json> <out_dir>
Writes <out_dir>/updated_<name>.json, read by `python -m analysis.data_check`. Makes no LLM call:
grading is database/action based; the one LLM-judged check (task_102) reuses the judgment recorded
in the file. Check: on the 8 submissions whose stored rewards already match the leaderboard
exactly, re-scoring changed no run (results/data_check.csv, column changed_runs)."""
import sys, time
from pathlib import Path
from loguru import logger
logger.remove()
import tau2.scripts.evaluate_trajectories as ev
from tau2.data_model.simulation import Results
from tau2.runner.build import _derive_read_log_allowlist

def kw(domain, task):
    return {"read_log_allowlist": _derive_read_log_allowlist(task), "retrieval_variant": "bm25"}
ev._build_eval_env_kwargs = kw

# NL assertions need an LLM judge: reuse the judgment recorded in the file, make no call.
from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import RewardType
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator as NLEvaluator
cur, reused = {}, [0]
_eval = ev.evaluate_simulation
def eval_sim(**k):
    cur["sim"] = k["simulation"]
    return _eval(**k)
ev.evaluate_simulation = eval_sim
def nl_reuse(cls, task, full_trajectory):
    chk = cur["sim"].reward_info.nl_assertions
    assert chk, "no recorded NL judgment"
    reused[0] += 1
    rw = 1.0 if all(c.met for c in chk) else 0.0
    return RewardInfo(reward=rw, nl_assertions=chk, reward_breakdown={RewardType.NL_ASSERTION: rw})
NLEvaluator.calculate_reward = classmethod(nl_reuse)
src, out = Path(sys.argv[1]), Path(sys.argv[2])
t = time.time()
r = ev.compute_simulation_rewards(Results.load(src), fresh_tasks=True)
out.mkdir(parents=True, exist_ok=True)
r.save(out / f"updated_{src.name}")
print(src.name, len(r.simulations), f"nl_reused={reused[0]}", f"{time.time()-t:.0f}s")
