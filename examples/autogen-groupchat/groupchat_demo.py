#!/usr/bin/env python3
"""
fak + AutoGen multi-agent group chat — a dependency-free runnable proof.

This example demonstrates the two things fak adds to an AutoGen *group chat* (several
agents — here a Planner, a Coder, and a Critic — taking turns in one shared
conversation, with a UserProxy executing the tools they propose):

  1. GOVERNANCE / TOOL-CALL ROUTING.  Every tool call any agent in the group chat
     proposes passes the capability floor in `policy.json` before it runs, and so does
     every speaker hand-off (the `transfer_to_*` coordination calls). A refusal is a
     verdict, not a crash (deny-as-value). The verdicts this demo prints are exactly
     what `fak preflight --policy examples/autogen-groupchat/policy.json --tool <name>`
     returns — cross-check any line.

  2. CONVERSATION-STATE PRESERVATION.  An AutoGen group chat's shared state IS the
     growing conversation transcript: every agent's turn re-sends the full accumulated
     message history to the model. Without a cross-turn shared prefix that transcript is
     re-prefilled from scratch on every turn (the cost grows with the square of the turn
     count); with fak's addressable, bit-exact KV cache the common conversation prefix is
     prefilled once and reused, so only each turn's new tokens are fresh prefill. Part 2
     computes that token-accounting model deterministically.

It is deliberately dependency-free: it does NOT import `autogen` / `autogen_agentchat`,
does NOT call any model, and needs no key, network, or GPU. The verdict logic mirrors the
kernel's allow-list evaluation so the proof is reproducible on any machine. In a real
group chat you do not reimplement any of this — you repoint the agents'
`OpenAIChatCompletionClient(base_url=...)` at a running `fak serve` (see README.md
"Wiring a real AutoGen group chat").

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POLICY_PATH = Path(__file__).with_name("policy.json")


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
# The group chat: agents taking turns in a shared conversation, each proposing #
# tools; a UserProxy executes the admitted ones.                              #
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    agent: str             # the agent whose turn it is
    tool: str              # the tool that agent proposes this turn
    expect: str            # the verdict kind we expect (for the self-check)


# A round-robin group chat working a coding task. The Planner scopes and hands off, the
# Coder edits and tests, the Critic reviews; the last two turns are over-eager calls
# (an irreversible repo delete and an exfiltration) that the floor stops before they run.
GROUP_CHAT: list[Turn] = [
    Turn("Planner", "web_search", "ALLOW"),
    Turn("Planner", "list_dir", "ALLOW"),         # list_ prefix
    Turn("Coder", "read_file", "ALLOW"),
    Turn("Coder", "run_tests", "ALLOW"),
    Turn("Critic", "get_diff", "ALLOW"),          # get_ prefix
    Turn("Coder", "delete_repo", "DENY"),         # explicit deny -> POLICY_BLOCK
    Turn("Critic", "post_to_pastebin", "DENY"),    # exfil-shaped, not allow-listed -> DEFAULT_DENY
]

# The group chat's own speaker hand-off calls (AutoGen's turn-taking / swarm hand-off
# machinery) — adjudicated the same way every agent tool is. The `transfer_` prefix is
# allow-listed so coordination itself is governed, not blocked.
HANDOFF_TOOLS = ["transfer_to_coder", "transfer_to_critic"]


def run_governance(policy: dict[str, Any], base_url: str | None) -> bool:
    print("== Part 1: every group-chat tool call passes the capability floor ==")
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

    print("  speaker hand-off tools:")
    for tool in HANDOFF_TOOLS:
        show("GroupChat", tool, "ALLOW")
    print("  agent turn tools:")
    for t in GROUP_CHAT:
        show(t.agent, t.tool, t.expect)

    print()
    return ok


# --------------------------------------------------------------------------- #
# The conversation-state-preservation model (deterministic token accounting).  #
# --------------------------------------------------------------------------- #
def run_conversation_model() -> bool:
    print("== Part 2: conversation-state-preservation model (modeled, not wall-clock) ==")

    # The shared conversation base every turn carries: the system prompt, the agent
    # roster, and the tool schemas — identical bytes on every turn.
    base_tokens = 1500
    # New tokens each agent appends to the shared transcript on its turn.
    per_turn_tokens = 250
    # Total turns across the group chat (3 agents, ~4 rounds of round-robin).
    turns = 12

    # Naive baseline: no cross-turn shared prefix, so the WHOLE accumulated transcript is
    # re-prefilled every turn. Before turn k the transcript is base + (k-1)*per_turn, and
    # summing k = 1..T gives a cost that grows with the square of the turn count.
    naive = turns * base_tokens + per_turn_tokens * (turns * (turns - 1) // 2)
    # fak: the conversation prefix is prefilled once (a kernel KV-cache object) and reused
    # across every turn; only each turn's new tokens are fresh prefill -> linear in turns.
    fak = base_tokens + turns * per_turn_tokens
    factor = naive / fak

    print(f"  shared conversation base (system+roster+tools) : {base_tokens} tok")
    print(f"  new tokens appended per turn                   : {per_turn_tokens} tok")
    print(f"  group-chat turns                               : {turns}")
    print(f"  naive prefill (whole transcript re-sent/turn)  : {naive} tok")
    print(f"  fak  prefill (conversation prefix reused)      : {fak} tok")
    print(f"  modeled prefill reduction                      : {factor:.2f}x")
    print()
    print("  This is deterministic token accounting (the multi-turn-conversation")
    print("  specialization of fak's shared-prefix reuse), NOT a measured wall-clock. The")
    print("  authoritative numbers live in BENCHMARK-AUTHORITY.md; the host-runnable")
    print("  deterministic ladder is `go run ./cmd/fanbench --grid canonical`. The")
    print("  live-model AutoGen-vs-native wall-clock comparison is the deferred bench-node")
    print("  run tracked by D-001 (#255).")
    print()

    # The model is only meaningful if reuse actually saves prefill at this geometry.
    return factor > 1


def main() -> int:
    policy = _load_policy()
    base_url = os.environ.get("FAK_BASE_URL") or None

    gov_ok = run_governance(policy, base_url)
    model_ok = run_conversation_model()

    passed = gov_ok and model_ok
    print(f"summary: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
