# Trace analysis

Loads a tau-bench results JSON and derives per-run facts. No API keys or model calls needed.

```
pip install -r requirements.txt
python -m analysis.load <results.json>      # -> local/work/<input folder name>/
python -m analysis.facts                    # -> facts.parquet (k = 2, 4, 6, all user turns)
python -m analysis.failure_type             # -> failure_types.parquet (failed runs only)
pytest analysis/tests
```

`facts` and `failure_type` take an optional work dir; without it they use the only dir under `local/work`.

Tables (in the work dir):

- `runs.parquet`: one row per run, with canonical `required_docs`.
- `messages.parquet`: one row per tool call, or per message without calls. `text` is set on the
  first row of a message only. Tool-result rows (`role == "tool"`) carry the answered tool's name.
- `action_checks.parquet`: expected actions from `reward_info`, used by `failure_type`.

A document is *surfaced* when its ID appears in a tool result, and *read* when a `cat`/`head`/`sed`/`less`
shell call names it or a KB search result returns it (KB results hold the full body).
