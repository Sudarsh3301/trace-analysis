"""Download the banking_knowledge trajectories of qualifying tau-bench submissions.

Keeps a submission only if it has banking_knowledge results, tau2_bench_version >= 1.0.1, text
(or unstated) modality, submission_type "standard" and user_simulator "gpt-5.2". Writes the files to
local/data/tau/<submission>/ (resumable) and the kept list to local/data/tau/kept.json.

Usage: python -m analysis.fetch_tau [--dry]     (--dry: list and filter only, no downloads)
"""
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://sierra-tau-bench-public.s3.amazonaws.com/"
OUT = Path("local/data/tau")


def get(url):
    return urllib.request.urlopen(url, timeout=60).read()


def list_submissions():
    names, token = [], None
    while True:
        q = {"list-type": "2", "prefix": "submissions/", "delimiter": "/"}
        if token:
            q["continuation-token"] = token
        xml = get(BUCKET + "?" + urllib.parse.urlencode(q)).decode()
        names += [p.split("/")[1] for p in re.findall(r"<CommonPrefixes><Prefix>([^<]+)", xml)]
        token = re.search(r"<NextContinuationToken>([^<]+)", xml)
        if not token:
            return names
        token = token.group(1)


def version(v):
    return tuple(int(x) for x in re.findall(r"\d+", v or "0"))


def reason(sub):
    """None if the submission qualifies, else why not."""
    m = sub.get("methodology") or {}
    if not (sub.get("results") or {}).get("banking_knowledge"):
        return "no banking_knowledge results"
    if version(m.get("tau2_bench_version")) < (1, 0, 1):
        return f"tau2_bench_version {m.get('tau2_bench_version')}"
    if sub.get("modality") not in (None, "text"):
        return f"modality {sub.get('modality')}"
    if sub.get("submission_type") != "standard":
        return f"submission_type {sub.get('submission_type')}"
    if m.get("user_simulator") != "gpt-5.2":
        return f"user_simulator {m.get('user_simulator')}"
    if not (sub.get("trajectory_files") or {}).get("banking_knowledge"):
        return "no trajectory file"
    return None


def head_size(url):
    return int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])


def download(url, dest, size):
    """Resumable: skip a complete file, otherwise fetch to .part and rename."""
    if dest.exists() and dest.stat().st_size == size:
        return size
    part = dest.with_suffix(".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.rename(dest)
    return size


def schema_error(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    if not {"tasks", "simulations"} <= set(d):
        return "missing tasks/simulations"
    if not d["simulations"] or any(not {"reward_info", "messages", "task_id", "id"} <= set(s)
                                   for s in d["simulations"]):
        return "simulation missing reward_info/messages/task_id/id"
    return None


def main(dry):
    kept, table = {}, []
    for name in list_submissions():
        try:
            sub = json.loads(get(f"{BUCKET}submissions/{name}/submission.json"))
        except Exception as e:  # example folders etc. have no usable submission.json
            table.append((name, "skipped", f"no submission.json ({type(e).__name__})", ""))
            continue
        why = reason(sub)
        if why:
            table.append((name, "skipped", why, ""))
            continue
        fname = sub["trajectory_files"]["banking_knowledge"]
        url = f"{BUCKET}submissions/{name}/trajectories/{fname}"
        try:
            size = head_size(url)
        except Exception as e:
            table.append((name, "skipped", f"trajectory file {fname} not found ({type(e).__name__})", ""))
            continue
        if dry:
            table.append((name, "kept (dry)", "", f"{size / 1e6:.0f} MB"))
            continue
        dest = OUT / name / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        download(url, dest, size)
        bad = schema_error(dest)
        if bad:
            table.append((name, "skipped", f"schema: {bad}", f"{size / 1e6:.0f} MB"))
            continue
        table.append((name, "kept", "", f"{size / 1e6:.0f} MB"))
        kept[name] = {"model": sub["model_name"], "file": str(dest), "pass_1": sub["results"]["banking_knowledge"]["pass_1"]}
    for row in table:
        print("{:<62} {:<11} {:<40} {}".format(*row))
    if not dry:
        (OUT / "kept.json").write_text(json.dumps(kept, indent=1), encoding="utf-8")
    print(f"{sum(r[1].startswith('kept') for r in table)} kept, {sum(r[1] == 'skipped' for r in table)} skipped")


if __name__ == "__main__":
    main("--dry" in sys.argv)
