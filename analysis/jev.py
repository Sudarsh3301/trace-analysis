"""Cached, budgeted, logged Jev (TypeSafe System One) calls.

A question is a plain dict: {"kind": "noul", "instructions", "criteria": {"true", "false"}} or
{"kind": "choice", "instructions", "criteria": {name: description}}. A call is keyed by the hash of
(model, state, questions, rep); a cached answer costs nothing. --dry-run writes payloads and calls
nothing; --max-calls is a hard stop on live calls. The key is read from .env and never printed.
"""
import csv
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "jev-1.14.0"  # pinned; "jev-latest" currently points here
USD_PER_INPUT_TOKEN = 0.042 / 1e6  # output tokens are free
WORK = Path("local/work")
LOG_FIELDS = ["time", "tag", "hash", "status", "latency_s", "input_tokens", "output_tokens", "cost_usd"]


class BudgetStop(Exception):
    pass


def noul(instructions, true, false):
    return {"kind": "noul", "instructions": instructions, "criteria": {"true": true, "false": false}}


def choice(instructions, criteria):
    return {"kind": "choice", "instructions": instructions, "criteria": criteria}


def load_key(env=Path(".env")):
    if not os.environ.get("TYPESAFE_API_KEY") and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "TYPESAFE_API_KEY":
                os.environ[k.strip()] = v.strip().strip("\"'")


def payload_hash(state, questions, rep=0, model=MODEL):
    blob = json.dumps({"model": model, "state": state, "questions": questions, "rep": rep}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def to_sdk(questions):
    from typesafe_sdk import Choice, Noul
    return {k: Noul(instructions=q["instructions"], criteria=q["criteria"]) if q["kind"] == "noul"
            else Choice(instructions=q["instructions"], criteria=q["criteria"]) for k, q in questions.items()}


def parse(resp):
    out = {"nouls": {}, "choices": {}}
    for k, a in resp.answers.items():
        if a.type == "noul":
            out["nouls"][k] = a.noul
        elif a.type == "choice":
            out["choices"][k] = {"choice": a.choice, "probabilities": a.probabilities}
    return out, resp.usage.input_tokens or 0, resp.usage.output_tokens or 0


class Jev:
    """ask() -> answers dict, or None in a dry run. Safe to call from several threads."""

    def __init__(self, dry_run=False, max_calls=0, work=WORK, client_factory=None):
        self.dry_run, self.max_calls = dry_run, max_calls
        self.cache, self.dry, self.log = work / "jev-cache", work / "jev-dry", work / "jev-log.csv"
        self.lock = threading.Lock()
        self.live = self.hits = self.would = 0
        self.local = threading.local()
        self.client_factory = client_factory or self._default_client
        for d in (self.cache, self.dry):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_client():
        from typesafe_sdk import TypeSafeClient
        load_key()
        return TypeSafeClient(model=MODEL)

    def peek(self, state, questions, rep=0):
        """Cached answer or None; never calls."""
        path = self.cache / f"{payload_hash(state, questions, rep)}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def ask(self, state, questions, tag="", rep=0):
        h = payload_hash(state, questions, rep)
        path = self.cache / f"{h}.json"
        if path.exists():
            with self.lock:
                self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        if self.dry_run:
            (self.dry / f"{h}.json").write_text(json.dumps(
                {"tag": tag, "rep": rep, "model": MODEL, "state": state, "questions": questions}), encoding="utf-8")
            with self.lock:
                self.would += 1
            return None
        with self.lock:
            if self.live >= self.max_calls:
                raise BudgetStop(f"--max-calls {self.max_calls} reached")
            self.live += 1
        if not hasattr(self.local, "client"):
            self.local.client = self.client_factory()
        t0, status, ans, tin, tout = time.time(), "ok", None, 0, 0
        try:
            ans, tin, tout = parse(self.local.client.system_one(state, to_sdk(questions)))
            path.write_text(json.dumps(ans), encoding="utf-8")
        except Exception as e:  # logged, not cached; the run goes on
            status = f"error:{type(e).__name__}:{getattr(e, 'status', '')}"
        with self.lock:
            new = not self.log.exists()
            with self.log.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, LOG_FIELDS)
                if new:
                    w.writeheader()
                w.writerow({"time": round(t0, 3), "tag": tag, "hash": h, "status": status,
                            "latency_s": round(time.time() - t0, 3), "input_tokens": tin,
                            "output_tokens": tout, "cost_usd": round(tin * USD_PER_INPUT_TOKEN, 8)})
        return ans

    def run_all(self, items, workers=4):
        """items: (state, questions, tag, rep). Results in order; stops cleanly on BudgetStop."""
        def one(it):
            try:
                return self.ask(*it)
            except BudgetStop:
                return "STOP"
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(one, items))
        stopped = sum(r == "STOP" for r in res)
        return [None if r == "STOP" else r for r in res], stopped


def sample(items, n, seed=0):
    """Seeded subset (pilot); n=None keeps everything."""
    import random
    if n is None or n >= len(items):
        return list(items)
    return random.Random(seed).sample(list(items), n)


def clamp(text, n=1500):
    text = text or ""
    if len(text) <= n:
        return text
    head = int((n - 20) * 0.6)
    return f"{text[:head]}\n...[{len(text) - n + 20} characters omitted]...\n{text[-(n - 20 - head):]}"


def summary(j, n_unique):
    return (f"{n_unique} unique requests: {j.hits} cached, "
            + (f"{j.would} to call (dry run)" if j.dry_run else f"{j.live} live calls this run"))
