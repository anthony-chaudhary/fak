"""Guard kernel-control pilot driver (#2509 CPU-scale pilot).

One identical workload, run once per arm, against an OpenAI-compatible
/v1/chat/completions endpoint served by fak's own in-kernel engine:

  arm=direct  BASE_URL = the fak serve address (guard-off baseline)
  arm=guard   BASE_URL = the guard gateway address injected into the child env
              (this script IS the guarded child: fak guard --gguf ... -- python driver.py)

Measures, per request: TTFB (wall to first response byte, stream=true) and
total wall. Then drives a 6-turn shared-prefix conversation so the kernel's
radix KV-prefix reuse has something to bite on, and (guard arm) snapshots
/metrics + the env-var NAMES visible to the child (never values).

stdlib only; timings are WITNESSED-derived wall-clock readings on this box.
"""
import json
import os
import sys
import time
import urllib.request

ARM = sys.argv[1] if len(sys.argv) > 1 else "direct"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
BASE = (
    os.environ.get("FAK_PILOT_BASE")
    or os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or ""
).rstrip("/")
if BASE.endswith("/v1"):
    BASE = BASE[: -len("/v1")]
if not BASE:
    sys.exit("no base url: set FAK_PILOT_BASE or run under fak guard (OPENAI_BASE_URL)")
MODEL = os.environ.get("FAK_PILOT_MODEL", "qwen2.5-1.5b")

SYSTEM = (
    "You are a terse benchmark assistant for the fak guard kernel-control pilot. "
    "Answer in one short sentence."
)
PROMPTS = [
    "Name one prime number.",
    "What color is the sky at noon?",
    "How many legs does a spider have?",
    "Name a fruit that is yellow.",
    "What is 7 plus 5?",
    "Name one planet in the solar system.",
    "What language is spoken in Paris?",
    "How many days are in a week?",
    "Name a mammal that can fly.",
    "What is the opposite of cold?",
]
WARMUP = 2
N = 20  # measured single-turn requests (PROMPTS cycled)


def chat(messages, max_tokens=8):
    """POST a streamed completion; return (ttfb_s, total_s, status)."""
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        resp.read(1)
        ttfb = time.perf_counter() - t0
        resp.read()
        total = time.perf_counter() - t0
        return ttfb, total, resp.status


rows = []
for i in range(WARMUP):
    chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPTS[i]}])

for i in range(N):
    p = PROMPTS[i % len(PROMPTS)]
    ttfb, total, status = chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
    )
    rows.append(
        {"arm": ARM, "kind": "single", "i": i, "ttfb_s": round(ttfb, 4), "total_s": round(total, 4), "status": status}
    )
    print(f"{ARM} single {i} ttfb={ttfb:.3f}s total={total:.3f}s", flush=True)

# shared-prefix multi-turn: same system + growing history -> radix reuse target
history = [{"role": "system", "content": SYSTEM}]
for turn in range(6):
    history.append({"role": "user", "content": PROMPTS[turn]})
    ttfb, total, status = chat(history, max_tokens=8)
    history.append({"role": "assistant", "content": "ok."})
    rows.append(
        {"arm": ARM, "kind": "multiturn", "turn": turn, "ttfb_s": round(ttfb, 4), "total_s": round(total, 4), "status": status}
    )
    print(f"{ARM} multiturn {turn} ttfb={ttfb:.3f}s total={total:.3f}s", flush=True)

with open(os.path.join(OUT, f"timings-{ARM}.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

# metrics snapshot (the kv_prefix family) + child env NAMES (never values)
try:
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as resp:
        open(os.path.join(OUT, f"metrics-{ARM}.txt"), "wb").write(resp.read())
except Exception as e:  # /metrics may be off the child-visible base in guard mode
    open(os.path.join(OUT, f"metrics-{ARM}.err"), "w").write(str(e))

env_names = sorted(os.environ.keys())
cred_like = sorted(
    n
    for n in env_names
    if any(t in n.upper() for t in ("API_KEY", "TOKEN", "SECRET", "CREDENTIAL", "PASSWORD"))
)
with open(os.path.join(OUT, f"childenv-{ARM}.json"), "w") as f:
    json.dump({"arm": ARM, "cred_like_names": cred_like, "base_url_var_present": bool(os.environ.get("OPENAI_BASE_URL"))}, f, indent=1)
print(f"{ARM} done: {len(rows)} rows; cred-like env names: {cred_like}", flush=True)
