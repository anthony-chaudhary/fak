---
title: "Agent runtime: category, ownership, interfaces, and proof"
description: "fak is the operator-controlled agent-runtime boundary for cache and context, model routing, tool authority, memory, observability, and native inference; this map separates that category from its Fused Agent Kernel architecture and shows current interfaces."
slug: agent-runtime
keywords:
  - fak agent runtime
  - managed agent runtime
  - tool-call loop
  - runtime ownership
  - agent integration
  - fak agent offline
  - kernel adjudication
  - result quarantine
date: 2026-07-15
---

# Agent runtime — category, ownership, interfaces, and proof

**Audience:** people evaluating the category and builders integrating with or
extending fak's managed, host-side agent loop.

**fak is an agent runtime: the operator-controlled boundary for cache and
context, model routing, tool authority, memory, observability, and native
inference.** **Agent runtime** is the simple public category. **Fused Agent
Kernel**, shortened to **agent kernel**, is the technical architecture and
historical name: the in-process checkpoint where model proposals meet operator
policy, reusable state, and evidence.

The category describes fak's owned control surface across its operating modes;
it does not claim every possible orchestration or runtime feature is shipped, or
that every mode activates every capability. Gateway and governed-client modes
retain a client-owned loop, while the managed host-side mode below owns that
loop. [`CLAIMS.md`](../../CLAIMS.md) and the selected mode's documentation remain
authoritative for shipped status.

In the managed host-side mode, the runtime owns the model/tool loop and sends
every managed tool call through the Fused Agent Kernel seam before a tool can
run. The operator chooses the integration mode and supplies task and
configuration; the model backend produces model turns; a tool implementation
owns only the admitted external side effect.

> **Default:** prove this boundary first with the deterministic, no-key
> `fak agent --offline` path. Then choose a live-provider or server integration from the
> interface table below.

## Ownership at a glance

| Owner | Responsibility | Boundary it does not own |
|---|---|---|
| Operator, client, or integration | Selects the runtime mode and supplies the task, credentials, configuration, and policy inputs. | The client does not authorize a managed tool call merely by requesting it. |
| Host-side agent runtime (`internal/agent`) | Drives model turns, preserves the conversation, presents tools, receives proposed calls, and continues after admitted results. | It does not bypass the kernel to execute managed calls. |
| Kernel adjudication and policy | Evaluates each proposed call against structural policy and returns an allow or deny verdict. | It does not invent the client task or perform the external side effect. |
| vDSO and reuse path | Resolves eligible repeated work locally and records dedup/reuse evidence. | A local hit is still part of the mediated path, not a client-side shortcut. |
| Engine and model backend | Produces the next model turn through the configured planner/provider seam. | Model output is a proposal, not authority to run a tool. |
| Tool dispatcher and implementation | Executes an allowed call and returns a typed result. | It receives no permission to execute a denied call. |
| Context MMU and result-admit path | Admits safe result content and quarantines content that must not re-enter model context. | Tool success alone does not make returned text safe context. |
| Trace and report surface | Records turns, calls, verdicts, reuse, quarantine, completion, and comparable-arm metrics. | Evidence describes the run; it does not replace the gate. |

The package name `internal/agent` means the **host loop and wire servers**, not the
untrusted guest program being guarded. Its source-level trust-line authority is
[`internal/agent/doc.go`](../../internal/agent/doc.go).

## One call, end to end

1. The client supplies a task to the selected runtime entry point.
2. The runtime asks the configured planner or model backend for the next turn.
3. The model either returns a final answer or proposes one or more tool calls.
4. For each managed call, the runtime enters the kernel path: vDSO lookup, adjudication,
   structural repair when applicable, and an allow or deny verdict.
5. A denied call returns a typed denial receipt. An eligible repeat may resolve locally;
   otherwise an allowed call reaches the configured tool engine or dispatcher.
6. The result-admit path screens the returned content. Safe content enters the next model
   turn; unsafe content is quarantined rather than copied into model context.
7. The loop continues until completion or its turn limit, while the trace/report surface
   records what happened at each boundary.

That ordering is the key contract: **model proposal → kernel verdict → optional effect →
result admission → continuation**. For the syscall analogy behind the contract, see
[Tool call is a syscall](tool-call-is-a-syscall.md).

## Choose an interface

| Builder job | Interface | What it gives you |
|---|---|---|
| Prove or inspect the reference loop locally | `fak agent --offline` | A deterministic A/B run using the mock planner, local tools, policy denial, reuse, and result screening; no provider key or network is required. |
| Exercise the reference loop with a live compatible provider | `fak agent` plus provider configuration | The same turn-and-tool path with live model calls; use the CLI help and environment/config authorities for the selected provider. |
| Host the native managed runtime as a service | `fak serve --native` | A long-lived agent application runtime that hosts sessions and the loop. Start with [Runtime vs client](runtime-vs-client.md), then the [server quickstart](../fak/server-quickstart.md). |
| Put an existing model client behind fak | `fak serve` | The gateway runtime for compatible model traffic; this governs calls but does not move the client's own agent loop into fak. See [Gateway](gateway.md). |
| Govern an existing local harness | `fak manage` | A governed client path for a harness you already run. It is a client relationship, not a second server layer. See [Runtime vs client](runtime-vs-client.md). |

The runtime's inputs are the task/conversation, tool definitions and dispatcher, policy,
and planner/model configuration. Its observable outputs are the answer or completion,
typed tool results and receipts, and trace/report evidence. Exact CLI flags and wire
shapes remain authoritative in [`cmd/fak/`](../../cmd/fak/) and the
[API reference](../fak/api-reference.md).

## Category and support boundary

| Dimension | Current context |
|---|---|
| Category | Agent runtime names fak's operator-controlled boundary across modes; agent kernel names its technical architecture. Neither label overrides feature-level claim status. |
| Mode | The detailed flow on this page describes fak-managed host-side execution. Gateway-only and guarded-client modes are valid choices, but they retain a client-owned loop. |
| Generation | Current behavior; code and tests are authoritative when this explanation and runtime diverge. |
| Lifecycle | `gen/now` builder route. Historical design notes are provenance, not this contract. |
| Support boundary | The deterministic offline proof is the first supported diagnostic. Live-provider behavior also depends on the selected provider, credentials, model compatibility, and deployment configuration. Tool side effects remain the tool implementation's responsibility after admission. |

For security guarantees and their scope, use [`SECURITY.md`](../../SECURITY.md). For
claim status, use [`CLAIMS.md`](../../CLAIMS.md); this route does not promote a simulated
or scoped result into a broader product claim.

## Next action

Run exactly this from the repository root:

```bash
fak agent --offline
```

The proof passes when the fak column reports `task completed (booked) YES`, the headline
reports `poisoned result blocked : YES`, and it reports
`destructive op prevented : YES`. The complete first-run sequence and expected output
live in the [repro packet](../repro-packet.md).

## Deeper routes

- [Runtime vs client](runtime-vs-client.md) — choose among gateway runtime, native agent
  runtime, and governed client roles.
- [Gateway](gateway.md) — understand the model-traffic boundary.
- [Policy in the kernel](policy-in-the-kernel.md) — configure structural admission.
- [One binary, one surface](one-binary-one-surface.md) — see why the paths share one
  checkpoint.
- [Agent integration architecture](../fak/agent-integration-architecture.md) — follow the
  broader integration and component map.
