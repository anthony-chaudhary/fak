#!/usr/bin/env python3
"""
fak + CrewAI manager-worker (hierarchical) crew — a dependency-free runnable proof.

This example demonstrates the two things fak adds to a CrewAI *hierarchical* crew (a
manager agent that delegates subtasks to worker agents):

  1. GOVERNANCE.  Every tool call a worker proposes passes the capability floor in
     `policy.json` before it runs (deny-as-value: a refusal is a verdict, not a crash).
     The verdicts this demo prints are exactly what `fak preflight --policy
     examples/crewai-crew/policy.json --tool <name>` returns — cross-check any line.

  2. COORDINATION-OVERHEAD REDUCTION.  In the manager-worker pattern the manager
     re-sends a shared crew brief (goal + roster + tool schemas + accumulated context)
     on every delegation. Without a cross-agent shared prefix that brief is re-prefilled
     once per delegation; with fak's addressable, bit-exact KV cache it is prefilled
     once and reused. Part 2 computes that token-accounting model deterministically.

It is deliberately dependency-free: it does NOT import `crewai`, does NOT call LiteLLM
or any model, and needs no key, network, or GPU. The verdict logic mirrors the kernel's
allow-list evaluation so the proof is reproducible on any machine. In a real crew you do
not reimplement any of this — you repoint the crew's `LLM(base_url=...)` at a running
`fak serve` (see README.md "Wiring a real CrewAI crew").

Optional live boundary: set FAK_BASE_URL=http://127.0.0.1:8080 to route every
adjudication through a running `fak serve` gateway instead of the local evaluator. The
printed verdicts are identical either way.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POLICY_PATH = Path(__file__).with_name("policy.json")


def cleanup() -> None:
    """No-op teardown marker: the demo is read-only today and future temp state belongs here."""
    return None


atexit.register(cleanup)


# --------------------------------------------------------------------------- #
# Verdict source: a local evaluator grounded in policy.json (default), or a    #
# live `fak serve` gateway when FAK_BASE_URL is set. Both speak one vocabulary.#
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    kind: str          # ALLOW | DENY  (the kinds this allow-list floor produces)
    reason: str        # NONE | POLICY_BLOCK | SECRET_EXFIL | DEFAULT_DENY


def _load_policy() -> dict[str, Any]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def adjudicate_local(policy: dict[str, Any], tool: str) -> Verdict:
    """Mirror the kernel's name-level allow-list decision for one proposed tool.

    Precedence matches the policy engine: explicit deny wins, then exact allow, then
    allow_prefix, else the fail_closed DEFAULT_DENY floor.
    """
    deny = policy.get("deny", {})
    if tool in deny:
        return Verdict("DENY", deny[tool])
    if tool in policy.get("allow", []):
        return Verdict("ALLOW", "NONE")
    if any(tool.startswith(p) for p in policy.get("allow_prefix", [])):
        return Verdict("ALLOW", "NONE")
    return Verdict("DENY", "DEFAULT_DENY")


def adjudicate_gateway(base_url: str, tool: str, arguments: dict[str, Any]) -> Verdict:
    """Adjudicate one call through a live `fak serve` /v1/fak/adjudicate endpoint."""
    body = json.dumps({"tool": tool, "arguments": arguments}).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/fak/adjudicate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (localhost dev)
        payload = json.load(resp)
    v = payload.get("verdict") or {}
    return Verdict(v.get("kind", "DENY"), v.get("reason", "DEFAULT_DENY"))


# --------------------------------------------------------------------------- #
# The crew: a manager delegating subtasks to workers, each proposing tools.    #
# --------------------------------------------------------------------------- #
@dataclass
class Delegation:
    worker: str            # the worker role the manager delegates to
    tool: str              # the tool that worker proposes for the subtask
    expect: str            # the verdict kind we expect (for the self-check)


# A research crew run under CrewAI's hierarchical process. The manager coordinates
# three workers; the last delegation is an over-eager worker that proposes an
# off-floor exfiltration tool — the floor stops it before it runs.
CREW: list[Delegation] = [
    Delegation("Researcher", "web_search", "ALLOW"),
    Delegation("Researcher", "fetch_url", "ALLOW"),
    Delegation("Analyst", "read_dataset", "ALLOW"),   # read_ prefix
    Delegation("Analyst", "summarize", "ALLOW"),
    Delegation("Writer", "send_email", "DENY"),       # explicit deny -> POLICY_BLOCK
    Delegation("Writer", "exfiltrate_notes", "DENY"),  # not on the floor -> DEFAULT_DENY
]

# The manager's own delegation tools (CrewAI's hierarchical machinery) — adjudicated
# the same way every worker tool is. Allow-listed so coordination itself is governed,
# not blocked.
MANAGER_TOOLS = ["delegate_work", "ask_coworker"]


def run_governance(policy: dict[str, Any], base_url: str | None) -> bool:
    print("== Part 1: every crew tool call passes the capability floor ==")
    label = f"live fak serve at {base_url}" if base_url else "local policy.json evaluator"
    print(f"verdict source: {label}\n")

    ok = True

    def show(actor: str, tool: str, expect: str | None = None) -> None:
        nonlocal ok
        if base_url:
            v = adjudicate_gateway(base_url, tool, {})
        else:
            v = adjudicate_local(policy, tool)
        run = "run" if v.kind == "ALLOW" else "BLOCKED"
        print(f"  {actor:<11} {tool:<18} verdict={v.kind:<5} reason={v.reason:<13} -> {run}")
        if expect is not None and v.kind != expect:
            print(f"    !! expected {expect}, got {v.kind}")
            ok = False

    print("  manager coordination tools:")
    for tool in MANAGER_TOOLS:
        show("Manager", tool, "ALLOW")
    print("  worker subtask tools:")
    for d in CREW:
        show(d.worker, d.tool, d.expect)

    print()
    return ok


# --------------------------------------------------------------------------- #
# The coordination-overhead model (deterministic token accounting).           #
# --------------------------------------------------------------------------- #
def run_overhead_model() -> bool:
    print("== Part 2: manager-role coordination-overhead model (modeled, not wall-clock) ==")

    # The shared crew brief the manager re-sends on every delegation: the goal, the
    # worker roster, the tool schemas, and the accumulated task context.
    shared_brief_tokens = 2000
    # Worker-specific instruction unique to each delegation.
    per_delegation_unique_tokens = 300
    # Total manager -> worker delegations across the crew run (3 workers, ~4 rounds).
    delegations = 12

    # Naive baseline: no cross-agent shared prefix, so the shared brief is re-prefilled
    # on every delegation.
    naive = delegations * (shared_brief_tokens + per_delegation_unique_tokens)
    # fak: the shared brief is prefilled once (a kernel KV-cache object) and reused
    # across every delegation; only the per-delegation unique tokens are new prefill.
    fak = shared_brief_tokens + delegations * per_delegation_unique_tokens
    factor = naive / fak

    print(f"  shared crew brief (re-sent per delegation) : {shared_brief_tokens} tok")
    print(f"  per-delegation unique instruction          : {per_delegation_unique_tokens} tok")
    print(f"  manager -> worker delegations              : {delegations}")
    print(f"  naive prefill (brief re-sent each time)    : {naive} tok")
    print(f"  fak  prefill (brief prefilled once, reused): {fak} tok")
    print(f"  modeled prefill reduction                  : {factor:.2f}x")
    print()
    print("  This is deterministic token accounting (the manager-worker specialization of")
    print("  fak's fan-out geometry), NOT a measured wall-clock. The authoritative numbers")
    print("  live in BENCHMARK-AUTHORITY.md; the host-runnable deterministic ladder is")
    print("  `go run ./cmd/fanbench --grid canonical`. The live-model CrewAI-vs-native")
    print("  wall-clock comparison is the deferred bench-node run tracked by D-001 (#255).")
    print()

    # The model is only meaningful if reuse actually saves prefill at this geometry.
    return factor > 1.0


def main() -> int:
    policy = _load_policy()
    base_url = os.environ.get("FAK_BASE_URL") or None

    gov_ok = run_governance(policy, base_url)
    model_ok = run_overhead_model()

    passed = gov_ok and model_ok
    print(f"summary: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
