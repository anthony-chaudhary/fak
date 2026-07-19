#!/usr/bin/env python3
# code_review_bot.py — a runnable governed code-review agent, ON the fak kernel (#3291).
#
# WHAT THIS IS
#   The flagship reference app for epic #3256's governed agent runtime: a small,
#   cloneable *governed code-review bot* that works a pull request where EVERY
#   proposed tool call is adjudicated by the real fak kernel via `fak preflight`.
#   Its least-agency floor is the SHIPPED manifest examples/code-review-bot-policy.json
#   (the same file the hermetic Go contract internal/adjudicator/codereview_agent_test.go
#   proves actually governs). One run shows all four acceptance elements:
#     * a multi-step review task under a least-agency floor (#C4),
#     * a live kernel DENY of a dangerous action on two distinct paths
#       (POLICY_BLOCK for a listed danger, DEFAULT_DENY for an ungranted capability),
#     * a cost cap + per-turn accounting that halts overspend,
#     * an append-only audit tail of every adjudication.
#
# WHY IT RUNS ON A PLAIN DEV HOST (the honest scope)
#   The kernel's verdict is deterministic and MODEL-INDEPENDENT: the same
#   (policy, proposed call) always yields the same ALLOW/DENY. So the *governance*
#   proof needs no model, no gateway, no GPU — `fak preflight` decides here and now.
#   What a live model WOULD add is only the *proposals*: here the review trajectory
#   is scripted (a realistic plan a reviewer would follow, plus two dangerous actions
#   an injected/confused model might attempt), and the kernel really adjudicates each.
#   Per-step cost is an ILLUSTRATIVE model-turn estimate (there is no live model to
#   bill), so the cost CAP demonstrates the enforcement *mechanism*; binding the cap
#   to real model tokens is promotion work (needs the runtime seam, #B1). Everything
#   the kernel decides below is real.
#
# RUN
#   python examples/code-review-bot/code_review_bot.py           # human transcript + summary
#   python examples/code-review-bot/code_review_bot.py --json    # machine audit tail (JSONL) to stdout
#   FAK_BIN=/path/to/fak python .../code_review_bot.py           # use a prebuilt kernel binary
#
# Requires: a `fak` binary (repo-root ./fak[.exe], $FAK_BIN, or `fak` on PATH) and
# examples/code-review-bot-policy.json. No network. Completes in well under a second.
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))  # examples/code-review-bot -> repo root
POLICY = os.path.join(ROOT, "examples", "code-review-bot-policy.json")
POLICY_REL = "examples/code-review-bot-policy.json"

# The least-agency floor gives a fixed dollars-per-cent; the cap is the governance ceiling.
COST_CAP_CENTS = 50  # halt the run before cumulative model-turn spend exceeds this

# The scripted review trajectory: a governed code-review bot working pull request #128.
# Each entry is one model turn: (label, tool, args, est_turn_cost_cents).
# Steps 1-4 are the legitimate review (read + comment); step 5 is a listed danger the
# floor must POLICY_BLOCK (a review bot never merges); step 6 is an ungranted capability
# an injected PR body tries to trigger, which fails closed (DEFAULT_DENY); step 7 is
# legitimate work that trips the budget ceiling — three distinct governance stops.
STEPS = [
    ("read the PR diff under review",        "read_diff",           {"pr": 128},                                                      8),
    ("load the PR metadata + author",        "get_pull_request",    {"pr": 128},                                                      7),
    ("search the tree for the changed API",  "search_code",         {"q": "func (s *Server) Handle"},                                12),
    ("post the review comment",              "post_review_comment", {"pr": 128, "body": "nit: guard the nil client before Handle; otherwise LGTM"}, 9),
    ("auto-merge the approved PR",           "merge_pull_request",  {"pr": 128, "method": "squash"},                                  6),
    ("email the diff to the requester",      "send_email",          {"to": "reviewer@example.com", "attach": "pr-128.diff"},           6),
    ("open the next changed file",           "read_file",           {"path": "internal/server/handle.go"},                            9),
]


def find_fak():
    """Locate a fak kernel binary: $FAK_BIN, then repo-root ./fak[.exe], then PATH."""
    env = os.environ.get("FAK_BIN")
    if env and os.path.exists(env):
        return env
    for name in ("fak.exe", "fak"):
        cand = os.path.join(ROOT, name)
        if os.path.exists(cand):
            return cand
    return shutil.which("fak")


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
        sys.stderr.write("least-agency floor missing: %s\n" % POLICY)
        return 2, []

    audit = []
    spent = 0
    for seq, (label, tool, args, cost) in enumerate(STEPS, start=1):
        # Governance gate 1 — the cost cap: refuse the turn BEFORE it is proposed if it
        # would breach the ceiling. Budget governance, independent of the kernel verdict.
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
        spent += cost  # the model turn happened (the proposal was formed), so it is billed
        audit.append({
            "seq": seq, "step": label, "tool": tool,
            "args_digest": v.get("args_digest"),
            "verdict": v.get("verdict"),
            "reason": v.get("reason", ""),
            "by": v.get("by"),
            "disposition": v.get("disposition", ""),
            "turn_cost_cents": cost, "cum_cost_cents": spent,
        })

    return 0, audit


def render(audit):
    allow = sum(1 for a in audit if a["verdict"] == "ALLOW")
    deny = sum(1 for a in audit if a["verdict"] == "DENY")
    spent = max((a["cum_cost_cents"] for a in audit), default=0)
    print("governed code-review bot -- %d steps, least-agency floor: %s" % (len(audit), POLICY_REL))
    print("  (every verdict below is the real fak kernel via `fak preflight` -- no model, no execution)\n")
    for a in audit:
        mark = "ALLOW " if a["verdict"] == "ALLOW" else "DENY  "
        tail = "" if a["verdict"] == "ALLOW" else "  <- %s (by=%s)" % (a["reason"], a["by"])
        print("  %d. %-34s %-20s %s%s" % (a["seq"], a["step"], a["tool"], mark, tail))
    print("\nsummary: %d ALLOW, %d DENY | spend %dc / cap %dc" % (allow, deny, spent, COST_CAP_CENTS))
    print("  DENY paths: kernel POLICY_BLOCK (a listed danger), kernel DEFAULT_DENY (ungranted tool), COST_CAP (budget).")
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
