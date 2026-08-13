---
title: "One useful system: fak's end-to-end value-chain spine"
description: "The narrow theory and milestone path joining a self-hosted kernel, managed harnesses, microagents, fleet operations, and UI without treating breadth as proof of value."
---

# One useful system: the end-to-end value-chain spine

## The theory

**fak becomes valuable when one long-lived kernel turns many agent entry points into one
observable, bounded, reusable execution system.** Local versus remote hosting, native versus
third-party harnesses, microagents, fleet skills, and UI are deployment choices around that
same loop; they are not five products that must all mature before a user can benefit.

The loop is:

> **ask → admit → run → observe → improve or stop**

Every layer must preserve one receipt for that loop:

| Layer | User job | Narrow contract |
|---|---|---|
| kernel, local or remote | reuse context and enforce the floor | one gateway, one policy point, measured usage |
| managed/native harness | start or attach the agent correctly | invocation resolves to the same kernel contract |
| microagent runtime | run bounded small workers cheaply | separate sessions over shared scheduling and context |
| fleet skills | choose, lease, verify, and retry work | effects are witnessed; concurrency is explicit |
| UI/managed service | answer “what is running, why, and what did it cost?” | render the receipt; do not invent a second control plane |

This is a **control-loop theory**, not a feature-stack theory. A layer earns its place only
when it shortens, cheapens, or makes safer the same completed user job. Merely exposing a
surface, supporting another provider, or drawing another dashboard does not prove value.

## The popular-use wedge

The first wedge is intentionally ordinary:

> An operator starts one kernel, runs a coding harness for the main task, lets it delegate
> two bounded subtasks, and can see that all three traversed the same managed execution
> boundary and terminated with receipts.

This is broad enough to exercise the architecture but narrow enough to demonstrate in one
command today. It avoids betting the proof on a custom agent protocol, a hosted control
plane, a large fleet, or a polished TUI.

## Milestone 0 — executable substrate (shipped spine)

```powershell
fak micro --selfcheck --json
```

The command starts an in-process `internal/gateway` HTTP kernel with the deterministic Mock
engine, drives **two** `internal/microagent` workers through one shared session gateway and a
one-slot scheduler, then reads the effects back. PASS requires:

- two successful agents;
- two `/v1/chat/completions` requests observed at the kernel boundary;
- two independently retired session-table entries; and
- nonzero provider-shaped token usage.

It requires no key, external model, network, or GPU. That makes it a reliable architecture
witness, **not** a model-quality or cost-savings claim. The dated real-process witness remains
[`_witnesses/microagent-real-kernel-2026-08-12.md`](./_witnesses/microagent-real-kernel-2026-08-12.md),
and paired tuned-baseline quality/cost evidence remains issue #6520.

## Build path — one receipt, successively less simulated

### M1: managed-harness parity

Make `fak manage` / `fak m` the popular entry point and prove Claude Code, Codex, and one
additional harness resolve to the same provider, base URL, policy, hooks, and child argv as
the mature guard path. The witness is #6541's machine-readable cross-harness parity packet.
Do not build a new harness yet; first prove that existing popular harnesses attach cleanly.

### M2: useful delegated task

Run one pinned, small repository task through both a tuned managed CLI and real-kernel
microagent arm. Capture task success, wall time, provider-reported usage, kernel verdicts,
and retries. This is #6520. Promote microagents only where the measured quality/cost/latency
trade is net-true; “one process can host many” is not enough.

### M3: fleet composition

Let the existing fleet skills choose and lease two independent subtasks, but execute their
model turns through the M1/M2 kernel contract. Fold only witnessed effects. The milestone is
one parent receipt linking task intent → leases → agent sessions → commits/effects, not a new
dispatch taxonomy. Tracked by #6554.

### M4: receipt-first UX

Render that parent receipt in the smallest existing operator surface: running/queued/stopped,
current model and policy, elapsed time/usage, last decision, and verified effect. Start
read-only. Add controls only after their state transition is already authoritative in the
kernel. This prevents a dashboard-shaped second control plane. Tracked by #6555.

### M5: managed deployment

Package the same kernel contract for a remote or managed endpoint: identity, durable receipts,
quotas, upgrades, and reconnect. Local and managed modes pass the same conformance witness;
managed service value comes from operations, not semantic fork. Tracked by #6556.

## Gates that prevent boiling the ocean

1. **One popular job before breadth.** Coding-harness delegation is the wedge until M2 proves
   useful outcomes. New verticals wait.
2. **One receipt across layers.** A milestone that cannot extend or render the same receipt is
   a separate product proposal.
3. **No modeled performance headline.** Architecture witnesses are labeled simulated/offline;
   value claims require tuned-baseline measurements.
4. **UI follows authority.** Read-only rendering precedes control; no UI-owned lifecycle state.
5. **Managed follows parity.** Remote packaging starts only after local and external-kernel
   paths satisfy the same contract.
6. **Fan-out follows a working spine.** Follow-ons are independently shippable and must not
   destabilize the one-command witness.

## What this does not unify

It does not force every agent into one implementation, replace provider APIs, merge native and
third-party harness code, or claim microagents should handle every task. The unification is the
managed execution contract and receipt. Diversity above and below that seam is expected.


