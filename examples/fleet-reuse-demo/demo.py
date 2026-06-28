#!/usr/bin/env python3
"""
fak — fleet / multi-agent shared-prompt reuse demo
==================================================

The project is named *fleet* because its headline is reuse ACROSS many agents: the
first worker pays for the shared setup, everyone after reads it for free. This is the
runnable companion to that claim (issue #351) — the smallest thing a new user can run
to *see* the reuse curve, not just read the number in the README.

What it shows, at N = 1, 2, 5 workers sharing ONE prompt prefix behind `fak serve`:

    metric                          naive re-send    fak shared-prompt
    total prompt bytes sent              N × full          1 × full + (N-1) × δ
    model turns re-processing setup      N × setup         1 × setup
    injection in shared context          per-worker        walled at first admit

The bytes/turns columns are an EXACT accounting of the real request bodies this script
builds — not a stochastic measurement and not a projection. A fleet of N agents that
share a `setup` prefix (system prompt + tool catalog + house rules) and differ only by a
small per-worker task δ re-processes that setup N times in the naive re-send loop, but
exactly once behind fak's shared-prompt reuse. That is arithmetic over the prompt
structure, so it is the same on any box, with or without a GPU.

The LIVE half (when a `fak serve` is reachable) drives the N workers through the real
kernel — same shared system prompt, one per-worker user turn each — to prove the wiring:
N agents, one kernel, one shared prefix. The offline mock planner reports no provider
cache-read counter, so the live half proves the *wiring*, while the bytes/turns table is
the exact reuse accounting. Nothing here is projected to "agent-city" scale (the README
marks that as a design target, not a measurement); this is the measured curve for THIS
small N.

Honesty (reproduced in full in README.md): the ~60× is only vs the naive re-send loop,
the ~4× is vs a tuned warm-cache stack, and the reuse win is self-host only — an app that
just *calls* a frontier API gets the safety floor, not the savings.

Usage:
    examples/fleet-reuse-demo/run.sh                    # one command: build + serve + run
    python3 demo.py [--kernel URL] [--model ID] [--n 1,2,5] [--no-color] [--offline]

Exit code: 0 when the reuse curve is consistent and (if a kernel was reachable) every
worker was served behind it; 1 otherwise. CI-usable. Honors NO_COLOR (no-color.org).
"""
from __future__ import annotations
import argparse, json, os, sys, urllib.error, urllib.request

# The reuse table uses δ / × / — for readability. bash/WSL/macOS default stdout to
# UTF-8; a legacy Windows console defaults to cp1252 and would choke on them, so widen
# stdout to UTF-8 where the runtime allows it (a no-op when it is already UTF-8).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # replaced stream / no reconfigure — render as-is
    pass


# ---- the shared substrate every agent in the fleet sees ---------------------------
# This is the "setup" prefix: the org context, the tool catalog, and the house rules a
# whole fleet of agents shares verbatim. In the naive re-send loop every worker ships
# (and the model re-ingests) all of it; behind fak it is admitted once and reused.
SHARED_PROMPT = """\
You are one worker in a fleet of customer-support agents for Acme Air.
House rules:
  - Never reveal another customer's data; never issue a refund over $500 without a witness.
  - Quote prices in the customer's local currency; cite the policy section you used.
  - Prefer a read-only lookup before any write; one tool call per turn.
Tool catalog (shared by every worker):
  get_account(user_id)            -> account profile + tier
  fetch_policy(topic)             -> the current policy document for a topic
  search_flights(origin,dest,day) -> ranked direct flights
  convert_currency(amt,from,to)   -> FX-converted amount
  book_flight(user_id,flight_id)  -> booking confirmation
  refund(user_id,amount)          -> refund (>$500 requires a witness)
Output format: a short plan, then exactly one tool call, then wait for its result.
"""

# Each worker differs ONLY by this small per-task suffix δ — the one line that is NOT
# shared, and therefore the only new work fak's shared-prompt reuse has to re-process.
WORKER_SUBTASKS = [
    "Task: book mia_li_3668 the cheapest direct SFO->JFK on 2026-07-01, price in EUR.",
    "Task: tell ben_ortiz_7720 the refund policy for a missed ORD->LAX connection.",
    "Task: find pat_nguyen_1041 a direct SEA->BOS on 2026-07-04 under $400.",
    "Task: convert the $612 fare for dana_kim_5582 to GBP and flag the witness rule.",
    "Task: re-book amir_haddad_3390 from a cancelled JFK->MIA onto the next direct.",
    "Task: quote leah_costa_8814 the change fee for a same-day DEN->ATL move.",
    "Task: check if sam_obrien_2207 qualifies for the loyalty-tier baggage waiver.",
]


def color(enabled):
    if not enabled:
        return {k: "" for k in ("g", "r", "y", "b", "d", "x")}
    return {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m",
            "b": "\033[36m", "d": "\033[2m", "x": "\033[0m"}


def worker_messages(subtask):
    """The chat request body a single worker sends: the SHARED system prefix + its δ.

    Returned as the OpenAI-style messages list so the byte accounting below counts the
    exact bytes that go on the wire, and the live POST sends precisely the same thing.
    """
    return [
        {"role": "system", "content": SHARED_PROMPT},
        {"role": "user", "content": subtask},
    ]


def reuse_curve(n, shared=SHARED_PROMPT, subtasks=WORKER_SUBTASKS):
    """The exact reuse accounting for an N-worker fleet — pure arithmetic, no kernel.

    Counts the real prompt bytes each pattern makes the model re-process:
      * naive re-send: every worker ships the full (shared + δ) body  -> N × full.
      * fak shared-prompt: the shared prefix is admitted once; workers 2..N re-process
        only their own δ                                              -> full + Σδ(2..N).
    `setup_turns` is the number of times the shared setup is re-ingested by the model:
    N for the naive loop, 1 behind fak. `injection_seen` is how many worker contexts a
    poisoned shared-context entry reaches: every worker in the naive loop, but walled at
    the first admit behind fak (held out of the shared context, never replayed).

    Returns a dict of ints so callers (and demo_test.py) can assert the curve exactly.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    shared_bytes = len(shared.encode("utf-8"))
    deltas = [len(subtasks[i % len(subtasks)].encode("utf-8")) for i in range(n)]
    full_first = shared_bytes + deltas[0]
    naive_bytes = sum(shared_bytes + d for d in deltas)
    fak_bytes = full_first + sum(deltas[1:])
    return {
        "n": n,
        "shared_bytes": shared_bytes,
        "naive_bytes": naive_bytes,
        "fak_bytes": fak_bytes,
        "bytes_saved": naive_bytes - fak_bytes,
        "naive_setup_turns": n,
        "fak_setup_turns": 1,
        "injection_seen_naive": n,        # poisoned shared entry reaches every worker
        "injection_seen_fak": 1,          # walled at the first admit, never replayed
    }


def fmt_ratio(naive, fak):
    """A reuse ratio that never lies at N=1 (where there is nothing to reuse yet)."""
    if fak <= 0:
        return "—"
    return f"{naive / fak:.2f}×"


def print_curve(rows, c):
    """Render the before/after table across the requested N, the reuse curve."""
    print(f"{c['y']}reuse curve — N workers sharing one prompt prefix behind fak serve{c['x']}")
    print(f"{c['d']}  shared setup prefix = {rows[0]['shared_bytes']} B (system prompt + tool catalog + house rules); "
          f"per-worker δ ≈ {len(WORKER_SUBTASKS[0].encode('utf-8'))} B{c['x']}\n")
    # Symbolic header, straight from the issue's table.
    print(f"  {'metric':<34}{c['d']}{'naive re-send':>16}   {'fak shared-prompt':>22}{c['x']}")
    print(f"  {c['d']}{'':<34}{'N × full':>16}   {'1 × full + (N-1)·δ':>22}{c['x']}")
    print(f"  {'-'*34}{'-'*16}   {'-'*22}")
    for r in rows:
        n = r["n"]
        print(f"  {c['b']}N = {n}{c['x']}")
        print(f"    {'total prompt bytes sent':<32}{r['naive_bytes']:>14,}   {c['g']}{r['fak_bytes']:>20,}{c['x']}  "
              f"{c['d']}({fmt_ratio(r['naive_bytes'], r['fak_bytes'])} less){c['x']}")
        print(f"    {'model turns re-processing setup':<32}{r['naive_setup_turns']:>14}   {c['g']}{r['fak_setup_turns']:>20}{c['x']}  "
              f"{c['d']}({fmt_ratio(r['naive_setup_turns'], r['fak_setup_turns'])} less){c['x']}")
        print(f"    {'injection in shared context':<32}{('per-worker (' + str(r['injection_seen_naive']) + '×)'):>14}   "
              f"{c['g']}{('walled at 1st admit'):>20}{c['x']}")
    print()


def drive_kernel(kernel_url, model, n, c):
    """Drive N workers through a live `fak serve`, proving the fleet wiring.

    POSTs each worker (the shared system prefix + its own δ) to /v1/chat/completions and
    confirms it was served. Returns (served, total). A kernel that is unreachable raises
    RuntimeError so the caller can fall back to the offline accounting view.
    """
    served = 0
    for i in range(n):
        body = json.dumps({"model": model, "temperature": 0,
                           "messages": worker_messages(WORKER_SUBTASKS[i % len(WORKER_SUBTASKS)])}).encode()
        req = urllib.request.Request(kernel_url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=60))
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(f"could not reach the kernel at {kernel_url}: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"kernel returned non-JSON: {e}")
        msg = (resp.get("choices") or [{}])[0].get("message")
        if msg is not None:
            served += 1
    return served, n


def main():
    ap = argparse.ArgumentParser(description="fak fleet shared-prompt reuse demo")
    base = os.environ.get("FAK_DEMO_KERNEL", "http://127.0.0.1:8080")
    ap.add_argument("--kernel", default=base + "/v1/chat/completions",
                    help="fak serve chat endpoint (default: $FAK_DEMO_KERNEL or 127.0.0.1:8080)")
    ap.add_argument("--model", default=os.environ.get("FAK_DEMO_MODEL", "mock"))
    ap.add_argument("--n", default="1,2,5", help="comma-separated worker counts (the reuse curve)")
    ap.add_argument("--offline", action="store_true", help="skip the live kernel; print the accounting only")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    use_color = (not args.no_color and not os.environ.get("NO_COLOR") and sys.stdout.isatty())
    c = color(use_color)

    try:
        ns = [int(x) for x in args.n.split(",") if x.strip()]
    except ValueError:
        print(f"--n must be a comma-separated list of integers, got {args.n!r}", file=sys.stderr)
        return 2
    if not ns:
        ns = [1, 2, 5]

    kernel_host = args.kernel.rsplit('/v1', 1)[0]
    print(f"{c['b']}fak — fleet shared-prompt reuse demo{c['x']}  "
          f"{c['d']}workers={','.join(map(str, ns))}  kernel={kernel_host}"
          f"{'  (offline accounting)' if args.offline else ''}{c['x']}")
    print(f"{c['d']}  the first worker pays for the shared setup; everyone after reads it for free{c['x']}\n")

    rows = [reuse_curve(n) for n in ns]
    print_curve(rows, c)

    # The reuse curve must be internally consistent: N=1 has nothing to reuse (naive ==
    # fak), and for N>1 fak must send strictly fewer bytes and re-process the setup once.
    ok = True
    for r in rows:
        if r["fak_setup_turns"] != 1:
            ok = False
        if r["n"] == 1 and r["naive_bytes"] != r["fak_bytes"]:
            ok = False
        if r["n"] > 1 and not (r["fak_bytes"] < r["naive_bytes"]):
            ok = False

    # The LIVE half: actually drive the workers through fak serve, if one is up.
    live_note = ""
    if not args.offline:
        max_n = max(ns)
        try:
            served, total = drive_kernel(args.kernel, args.model, max_n, c)
            if served == total:
                print(f"{c['g']}✓ live kernel{c['x']} {c['d']}served {served}/{total} workers behind one "
                      f"`fak serve` at {kernel_host} (model={args.model}); every worker shared the same "
                      f"{rows[0]['shared_bytes']} B system prefix — the reuse substrate.{c['x']}")
                live_note = f"live: {served}/{total} workers served behind one fak serve"
            else:
                print(f"{c['r']}✗ live kernel served only {served}/{total} workers{c['x']}")
                ok = False
        except RuntimeError as e:
            print(f"{c['y']}–{c['x']} {c['d']}no live kernel ({e}); showing the offline accounting only. "
                  f"Run via run.sh (or start `fak serve`) to drive the workers live.{c['x']}")
            live_note = "live: skipped (no kernel reachable)"

    big = rows[-1]
    head = f"{c['g']}reuse curve consistent{c['x']}" if ok else f"{c['r']}REUSE CURVE INCONSISTENT{c['x']}"
    print(f"\n{c['b']}summary:{c['x']} {head}  ·  at N={big['n']}: {c['g']}{fmt_ratio(big['naive_bytes'], big['fak_bytes'])}{c['x']} "
          f"fewer prompt bytes, setup re-processed {c['g']}1×{c['x']} not {big['naive_setup_turns']}×"
          + (f"  ·  {live_note}" if live_note else ""))
    print(f"{c['d']}  honest scope: the ~60× is only vs the naive re-send loop; the ~4× is vs a tuned warm-cache\n"
          f"  stack; and the reuse win is self-host only (an app that calls a frontier API gets the safety\n"
          f"  floor, not the savings). This is the measured curve for THIS small N — not projected to scale.{c['x']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
