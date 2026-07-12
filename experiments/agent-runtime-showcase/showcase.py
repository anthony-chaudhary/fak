#!/usr/bin/env python3
# showcase.py — a runnable governed code-review agent, ON the fak kernel (#3291).
#
# WHAT THIS IS
#   A small, cloneable reference app that runs a real multi-step task (a governed
#   *code-review* agent) where EVERY proposed tool call is adjudicated by the real
#   fak kernel via `fak preflight` — the structural admission seam, "ALLOW/DENY by
#   structure, no model in the loop". It shows, in ONE run:
#     * a multi-step task under a least-agency preset (#C4, coding-agent-safe.json),
#     * a live kernel DENY of a dangerous action (two refusal paths: gitgate + monitor),
#     * a cost cap + accounting that halts overspend,
#     * an append-only audit tail of every adjudication.
#
# WHY IT RUNS ON A PLAIN DEV HOST (the honest scope)
#   The kernel's verdict is deterministic and MODEL-INDEPENDENT: the same
#   (policy, proposed call) always yields the same ALLOW/DENY. So the *governance*
#   proof needs no model, no gateway, no GPU — `fak preflight` decides here and now.
#   What a live model WOULD add is the *proposals*: here the multi-step plan is
#   scripted (a realistic review trajectory), and the kernel really adjudicates it.
#   The per-step cost is an ILLUSTRATIVE model-turn estimate (there is no live model
#   to bill), so the cost CAP demonstrates the enforcement *mechanism*; a cap bound to
#   real model tokens is promotion work (needs the runtime seam, #B1). Everything the
#   kernel decides below is real.
#
# RUN
#   python experiments/agent-runtime-showcase/showcase.py            # human transcript + summary
#   python experiments/agent-runtime-showcase/showcase.py --json     # machine audit tail (JSONL) to stdout
#   FAK_BIN=/path/to/fak python .../showcase.py                      # use a prebuilt kernel binary
#
# Requires: a `fak` binary (repo-root ./fak[.exe], $FAK_BIN, or `fak` on PATH) and
# examples/presets/coding-agent-safe.json. No network. Completes in well under a second.
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))   # experiments/agent-runtime-showcase -> repo root
POLICY = os.path.join(ROOT, "examples", "presets", "coding-agent-safe.json")

# The least-agency floor gives a fixed dollars-per-cent; the cap is the governance ceiling.
COST_CAP_CENTS = 50   # halt the run before cumulative model-turn spend exceeds this


def find_fak():
    """Locate a fak kernel binary: $FAK_BIN, then repo-root ./fak[.exe], then PATH."""
    env = os.environ.get("FAK_BIN")
    if env and os.path.exists(env):
        return env
    for name in ("fak.exe", "fak"):
        cand = os.path.join(ROOT, name)
        if os.path.exists(cand):
            return cand
    onpath = shutil.which("fak")
    if onpath:
        return onpath
    return None


# The scripted review trajectory: a governed code-review agent working a proposed patch.
# Each entry is one model turn: (label, tool, args, est_turn_cost_cents, danger?).
# The first four are the legitimate review; the next two are dangerous actions the floor
# must refuse; the last is legitimate but trips the budget ceiling — three governance stops.
STEPS = [
    ("read the file under review",          "Read",  {"file_path": "internal/uiquality/uiquality.go"},              8, False),
    ("locate call-sites before commenting", "Grep",  {"pattern": "func Score", "path": "internal/uiquality"},       6, False),
    ("run the package tests",               "Bash",  {"command": "go test ./internal/uiquality -count=1"},         15, False),
    ("write the review summary",            "Write", {"file_path": "review-notes.md", "content": "LGTM: 1 nit"},    7, False),
    ("auto-publish the approved fix",       "Bash",  {"command": "git push origin main --force"},                   6, True),
    ("purge a flaky test account",          "delete_account", {"id": "acct_7731"},                                   5, True),
    ("continue into the next file",         "Read",  {"file_path": "internal/uiquality/score_table.go"},            8, False),
]


def adjudicate(fak, tool, args):
    """Ask the real kernel for a verdict on one proposed tool call. No model, no execution."""
    proc = subprocess.run(
        [fak, "preflight", "--tool", tool, "--args", json.dumps(args), "--policy", POLICY, "--json"],
        cwd=ROOT, capture_output=True, text=True,
    )
    out = proc.stdout
    brace = out.find("{")
    if brace < 0:
        raise RuntimeError("fak preflight emitted no JSON verdict: %s%s" % (proc.stdout, proc.stderr))
    return json.loads(out[brace:])


def run():
    fak = find_fak()
    if not fak:
        sys.stderr.write("fak binary not found (set FAK_BIN, or build ./cmd/fak); cannot adjudicate.\n")
        return 2, []
    if not os.path.exists(POLICY):
        sys.stderr.write("least-agency preset missing: %s\n" % POLICY)
        return 2, []

    audit = []
    spent = 0
    for seq, (label, tool, args, cost, danger) in enumerate(STEPS, start=1):
        # Governance gate 1 — the cost cap: refuse the turn BEFORE it is proposed if it
        # would breach the ceiling. This is budget governance, independent of the kernel.
        if spent + cost > COST_CAP_CENTS:
            audit.append({
                "seq": seq, "step": label, "tool": tool, "verdict": "DENY",
                "reason": "COST_CAP", "by": "budget", "disposition": "TERMINAL",
                "turn_cost_cents": cost, "cum_cost_cents": spent,
                "detail": "cap %dc would be breached (%d+%d)" % (COST_CAP_CENTS, spent, cost),
            })
            continue

        # Governance gate 2 — the kernel: real structural adjudication of the proposed call.
        v = adjudicate(fak, tool, args)
        spent += cost   # the model turn happened (the proposal was formed), so it is billed
        rec = {
            "seq": seq, "step": label, "tool": tool,
            "args_digest": v.get("args_digest"),
            "verdict": v.get("verdict"),
            "reason": v.get("reason", ""),
            "by": v.get("by"),
            "disposition": v.get("disposition", ""),
            "turn_cost_cents": cost, "cum_cost_cents": spent,
        }
        audit.append(rec)

    return 0, audit


def render(audit):
    allow = sum(1 for a in audit if a["verdict"] == "ALLOW")
    deny = sum(1 for a in audit if a["verdict"] == "DENY")
    spent = max((a["cum_cost_cents"] for a in audit), default=0)
    print("governed code-review agent -- %d steps, least-agency floor: examples/presets/coding-agent-safe.json" % len(audit))
    print("  (every verdict below is the real fak kernel via `fak preflight` -- no model, no execution)\n")
    for a in audit:
        mark = "ALLOW " if a["verdict"] == "ALLOW" else "DENY  "
        tail = "" if a["verdict"] == "ALLOW" else "  <- %s (by=%s)" % (a["reason"], a["by"])
        print("  %d. %-34s %-14s %s%s" % (a["seq"], a["step"], a["tool"], mark, tail))
    print("\nsummary: %d ALLOW, %d DENY | spend %dc / cap %dc" % (allow, deny, spent, COST_CAP_CENTS))
    print("  DENY paths shown: kernel POLICY_BLOCK (gitgate), kernel DEFAULT_DENY (monitor), COST_CAP (budget).")
    print("  audit tail (append-only, one JSON object per adjudication):")
    for a in audit:
        print("    " + json.dumps(a, ensure_ascii=False, sort_keys=True))


def main(argv):
    rc, audit = run()
    if rc != 0:
        return rc
    if "--json" in argv[1:]:
        for a in audit:
            print(json.dumps(a, ensure_ascii=False, sort_keys=True))
    else:
        render(audit)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
