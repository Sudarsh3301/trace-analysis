"""Match knowledge-base documents against a run: surfaced vs read, with first message index."""
import json
import re

DOC_TOKEN = re.compile(r"doc_[A-Za-z0-9_()\-]+")
READ_CMD = re.compile(r"\b(cat|head|sed|less)\b")


def canon(doc_id):
    s = doc_id.strip().lower().replace("(", "_").replace(")", "_")
    if s.startswith("./"):
        s = s[2:]
    for ext in (".md", ".json"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    return s.rstrip("_")


def _ids(text):
    return {canon(t) for t in DOC_TOKEN.findall(text or "")}


def doc_events(rows):
    """rows: dicts with idx, role, text, tool_name, tool_args, in message order.
    Returns {canonical_id: {"surfaced": first_idx|None, "read": first_idx|None}}."""
    out = {}

    def mark(doc, level, idx):
        e = out.setdefault(doc, {"surfaced": None, "read": None})
        if e[level] is None:
            e[level] = idx

    for r in rows:
        text, name = r["text"] or "", r["tool_name"] or ""
        if r["role"] == "tool":
            for d in _ids(text):
                mark(d, "surfaced", r["idx"])
            if name.startswith("KB_search"):  # KB results carry the full body
                for d in re.findall(r"ID: (\S+)", text):
                    mark(canon(d), "read", r["idx"])
        elif name == "shell":
            cmd = json.loads(r["tool_args"] or "{}").get("command", "")
            if READ_CMD.search(cmd):
                for d in _ids(cmd):
                    mark(d, "read", r["idx"])
    return out
