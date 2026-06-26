# fak + AutoGen multi-agent group chat

This example covers the [AutoGen](https://microsoft.github.io/autogen/) **group-chat**
pattern — several agents (here a Planner, a Coder, and a Critic) taking turns in one
shared conversation, with a UserProxy executing the tools they propose — and what `fak`
adds to it:

1. **Governance / tool-call routing.** Every tool call any agent proposes — and every
   speaker hand-off — passes the capability floor in [`policy.json`](policy.json) before
   it runs. A refusal is a verdict, not a crash (deny-as-value).
2. **Conversation-state preservation.** An AutoGen group chat's shared state *is* the
   growing conversation transcript; fak prefills the shared conversation prefix once and
   reuses it across every turn instead of re-prefilling the whole transcript each turn.

It is the group-chat companion to the generic AutoGen recipe in
[`docs/fak/agent-framework-integration.md` → AutoGen](../../docs/fak/agent-framework-integration.md#autogen),
which covers the one-line base-URL repoint and the `FunctionTool` wrapper. This directory
adds the **multi-agent** specialization: the example group chat, the conversation-state
model, and the performance comparison.

This is the issue [#248](https://github.com/anthony-chaudhary/fak/issues/248) (Track D ·
**D-004**) deliverable. It maps to that issue's acceptance as: **AutoGen agents via fak**
(the Mode A repoint, below) · **conversation preservation** (Part 2 + the KV-cache
narrative) · **multi-agent example** (this group chat) · **benchmark vs native** (the
modeled reduction here; the live-model wall-clock is the deferred bench-node run owned by
[D-001 #255](https://github.com/anthony-chaudhary/fak/issues/255)).

## Run it

The proof is **dependency-free** — it does not import `autogen` / `autogen_agentchat`,
does not call a model, and needs no key, network, or GPU. It completes in under a second
on a typical laptop. From the repository root:

```bash
python examples/autogen-groupchat/groupchat_demo.py        # -> ends with "summary: PASS"
go run ./cmd/fak policy --check examples/autogen-groupchat/policy.json   # the chat's floor is valid
```

Captured output is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md). The verdicts the demo
prints are exactly what the kernel returns — cross-check any line:

```bash
go run ./cmd/fak preflight --policy examples/autogen-groupchat/policy.json --tool delete_repo --args '{}'
# verdict=DENY reason=POLICY_BLOCK by=monitor    (the Coder's blocked irreversible call)
go run ./cmd/fak preflight --policy examples/autogen-groupchat/policy.json --tool post_to_pastebin --args '{}'
# verdict=DENY reason=DEFAULT_DENY by=monitor    (an exfil-shaped tool nobody allow-listed)
go run ./cmd/fak preflight --policy examples/autogen-groupchat/policy.json --tool get_diff --args '{}'
# verdict=ALLOW reason=NONE by=monitor           (the Critic's get_ tool)
```

To run the same adjudication through a live gateway instead of the local evaluator,
start a `fak serve` and set `FAK_BASE_URL`:

```bash
FAK_BASE_URL=http://127.0.0.1:8080 python examples/autogen-groupchat/groupchat_demo.py
```

The printed verdicts are identical either way — the local evaluator mirrors the kernel's
name-level allow-list decision (explicit deny → exact allow → prefix allow → fail-closed
`DEFAULT_DENY`).

## Conversation-state preservation

A group chat is multi-agent, but it is single-conversation: AutoGen's group-chat manager
keeps one shared message list, and on every turn the agent whose turn it is sends the
model the **whole accumulated transcript** — the system prompt, the agent roster, the
tool schemas, and every message exchanged so far — plus its own new contribution. The
shared base and the earlier turns are the same bytes on turn 12 as they were on turn 1;
only the newest tokens are new.

Without a cross-turn shared prefix, that transcript is re-prefilled on **every** turn, so
the prefill cost grows with the *square* of the turn count and the conversation dominates
the chat's token bill. fak removes the repeated work by treating the conversation prefix
as a kernel KV-cache object — prefilled once, spliced on the original bytes (a memcpy,
never a re-marshal), and reused across every turn so the provider's prompt-cache prefix
stays byte-identical. Only each turn's new tokens are fresh prefill.

This is the same lever fak applies to any agent fleet (do shared prefill once; later
turns read it for free), specialized to the one structure — a long shared conversation —
that AutoGen multi-agent chats accumulate the most of. It does **not** change AutoGen's
topology: agents still take turns, the UserProxy still runs their tools, and hand-offs are
unchanged. fak only governs *which* tool calls run and prefills the conversation prefix
once.

## Performance comparison

`groupchat_demo.py` Part 2 computes the conversation-state model deterministically. For a
group chat with a 1500-token shared base, 250 new tokens per turn, and 12 round-robin
turns (3 agents × ~4 rounds):

| | prefill tokens | how |
|---|---|---|
| Naive (whole transcript re-sent each turn) | `12×1500 + 250×(12·11/2)` = **34,500** | the transcript is re-prefilled on every turn (quadratic in turns) |
| fak (conversation prefix reused) | `1500 + 12×250` = **4,500** | the shared prefix is a reused KV-cache object (linear in turns) |
| **Modeled prefill reduction** | **7.67×** | on the shared conversation prefix |

This is **deterministic token accounting** — the multi-turn-conversation specialization of
fak's shared-prefix reuse — **not** a measured wall-clock, and it is honest about that.
The two authoritative paths:

- **Host-runnable deterministic ladder:** `go run ./cmd/fanbench --grid canonical` emits
  the D-001 coordination-overhead-vs-N ladder (N = 1, 100, 500, 1000) as in-process
  kernel arithmetic. Every published number is governed by
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).
- **Live-model wall-clock (deferred):** the AutoGen-vs-native end-to-end wall-clock
  comparison is the bench-node run tracked by **D-001 (#255)** — it needs AutoGen
  installed and a live model on a bench node, which is out of scope for this
  dependency-free example. No wall-clock figure is asserted here.

## Wiring a real AutoGen group chat

You do not reimplement any of the demo's plumbing in a real chat. AutoGen v0.4 (AgentChat)
drives models through a model client that takes a custom `base_url`; point it at
`fak serve` and run the group chat as usual.

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 1. Repoint every agent's model client at the fak gateway (Mode A — one line each).
#    For an unrecognized local model id, AutoGen also needs `model_info=...`; see the
#    migration guide's AutoGen section.
fak_client = OpenAIChatCompletionClient(
    model="gpt-4o", base_url="http://127.0.0.1:8080/v1", api_key="fak-local")

planner = AssistantAgent("Planner", model_client=fak_client)
coder   = AssistantAgent("Coder",   model_client=fak_client)
critic  = AssistantAgent("Critic",  model_client=fak_client)

# 2. The group chat shares one growing conversation across all three agents — which is
#    exactly the prefix fak prefills once and reuses across turns.
team = RoundRobinGroupChat(
    [planner, coder, critic],
    termination_condition=MaxMessageTermination(12),
)
# await team.run(task="...")   # run the multi-agent conversation as usual.
```

For tool-site governance (Mode B), wrap each tool's callable with the shared
`guarded(...)` helper from the integration guide before handing it to AutoGen's
`FunctionTool`, so a poisoned result is quarantined before it enters the shared
conversation:

```python
from autogen_core.tools import FunctionTool   # v0.4 tool wrapper

def _run_tests(target: str) -> str:
    return my_runner.run(target)   # your real executor

run_tests = FunctionTool(
    guarded("run_tests", _run_tests),          # guarded() from the integration guide
    description="Run the test suite for a target.")
# Register run_tests on the agent's tools=[...] as usual.
```

The tool **names** in [`policy.json`](policy.json) must match the names your agents
register (AutoGen's `transfer_to_*` hand-off tools and your `FunctionTool` names) —
verify against your installed AutoGen version, as the integration guide notes. Anything
not on the allow-list hits the fail-closed `DEFAULT_DENY`.

## What this does not claim

- It does **not** run a live AutoGen group chat or any model — it proves the verdict
  boundary and the conversation-state model with a dependency-free adapter. The final
  binding belongs in an app pinned to the AutoGen version it ships with.
- The 7.67× figure is a **modeled** token reduction at one illustrative geometry, not a
  measured wall-clock. The live-model comparison is the deferred D-001 (#255) bench-node
  run.
- The floor bounds **which tools** run, by tool *name* — it does not filter the
  *arguments* of an allow-listed tool (argument predicates are a separate, partial
  surface; see [`POLICY.md`](../../POLICY.md)).

## Sources

- AutoGen v0.4 AgentChat teams (`RoundRobinGroupChat`) and the OpenAI model client:
  <https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html>.
- Generic AutoGen recipe (Mode A repoint + `FunctionTool` wrapper):
  [`docs/fak/agent-framework-integration.md` → AutoGen](../../docs/fak/agent-framework-integration.md#autogen)
  · [`docs/fak/migration-guide.md` → AutoGen](../../docs/fak/migration-guide.md#migrating-from-autogen).
- The shared-prefix geometry and published numbers:
  [`cmd/fanbench`](../../cmd/fanbench/main.go) (`--grid canonical`) ·
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) ·
  [`docs/explainers/compounding-benefits-of-a-saved-call.md`](../../docs/explainers/compounding-benefits-of-a-saved-call.md).
- Track-D parity status (where this child sits):
  [`docs/notes/track-d-agent-framework-parity-tracking-304.md`](../../docs/notes/track-d-agent-framework-parity-tracking-304.md).
