#!/usr/bin/env python3
"""Peek at a session's recent meaningful turns + any goal/loop directive."""
import os, sys, json, glob, re

USER = os.environ.get("FLEET_USER_HOME", os.path.expanduser("~"))

def find(sid):
    for p in glob.glob(os.path.join(USER, ".claude*", "projects", "*", sid + ".jsonl")):
        return p
    return None

def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    out.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    inp = b.get("input", {})
                    arg = inp.get("command") or inp.get("prompt") or inp.get("file_path") or ""
                    out.append(f"[TOOL {b.get('name')}: {str(arg)[:120]}]")
                elif b.get("type") == "tool_result":
                    c = b.get("content")
                    out.append("[result] " + (c if isinstance(c, str) else text_of(c))[:160])
        return "\n".join(x for x in out if x)
    return ""

sid = sys.argv[1]
p = find(sid)
if not p:
    print("NOT FOUND", sid); sys.exit(1)
objs = []
with open(p, encoding="utf-8", errors="replace") as f:
    for ln in f:
        ln = ln.strip()
        if ln:
            try: objs.append(json.loads(ln))
            except: pass
# first user prompt (the original directive)
goal = None
for o in objs:
    if o.get("type") == "user":
        t = text_of((o.get("message") or {}).get("content"))
        if re.search(r"/goal|/loop|/dispatch|/next-up|<command-name>", t):
            goal = t[:400]; break
        if goal is None and t.strip():
            goal = t[:400]
print(f"=== {sid} ===  ({len(objs)} records)  file={os.path.basename(os.path.dirname(p))}")
print("ORIGINAL DIRECTIVE:", (goal or "(none)").replace("\n"," ")[:380])
print("--- last meaningful turns ---")
msgs = [o for o in objs if o.get("type") in ("user", "assistant")]
for o in msgs[-6:]:
    role = (o.get("message") or {}).get("role", o.get("type"))
    syn = "(synthetic)" if (o.get("message") or {}).get("model") == "<synthetic>" else ""
    t = text_of((o.get("message") or {}).get("content")).replace("\n", " ")
    print(f"  [{role}{syn}] {t[:240]}")
