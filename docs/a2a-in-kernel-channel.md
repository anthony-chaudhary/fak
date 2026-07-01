---
title: "The in-kernel agent-to-agent message channel"
description: "fak's a2achan: a capability-floored, Ref-backed mailbox that delivers an addressed value from one agent to another — gated by the same default-deny floor that gates a tool call, across in-kernel agents, sessions, and context windows."
---

# The in-kernel agent-to-agent message channel (`a2achan`)

Most "agent-to-agent" work is a *transport*: a way for two agents to find each
other and exchange bytes over HTTP. fak already has that story at the fleet edge
(see [`a2a-value-opportunities.md`](a2a-value-opportunities.md)). `a2achan` is the
other half — the part the transport projects onto: an **in-kernel** primitive that
delivers an addressed value from one agent to another, **gated by the same
default-deny floor that gates a tool call**. A message is not a `memcpy`; it is an
adjudicated transfer.

## The gap it fills

fak had every *half* of a channel but no channel:

- `abi.Ref` carries the `Taint` + `ShareScope` a cross-agent message needs ("never
  shared more widely than its scope; sharing a result shares its taint") — but a
  `Ref` never *routes* anywhere.
- The async `Submit`/`Reap` seam is 1:1 — the same goroutine submits and reaps;
  there is no recipient identity, queue, or notify.
- `session`, `recall`, and the vDSO/emitter fan-out move per-session drive state,
  read-only memory images, or cache-invalidation broadcasts — none *delivers* an
  addressed value from agent A to a different agent B.

`a2achan` is that missing piece, and it reuses the existing currency (`Ref`
provenance) and the existing registries (the kernel's adjudicator + result-admitter
chains) rather than inventing a parallel security surface.

## The model

```
   agent "alpha"                a2achan.Bus (process-global)            agent "bravo"
   ------------                 ----------------------------            ------------
   Send(to, body) --> [ a2aGate: capability floor ] --deny--> refused (deny-as-value)
                              |  allow
                              v
                      queue[ChannelKey] : [ Msg{From,To,Body Ref,Seq} , ... ]
                              |
                              |  Recv(to) blocks (ctx-aware) until a message arrives
                              v
                      [ a2aIngress: quarantine screen ] --quarantine--> HELD (not admitted)
                              |  allow
                              v
                                                                  Msg delivered (Body keeps its Taint)
```

- A **`ChannelKey{Locale, ID}`** names a mailbox.
- A **`Message`** is a `Ref` `Body` (its `Taint`+`Scope` ride unchanged), the
  `From` principal + `To` address, and a per-bus monotonic `Seq` (a deterministic
  delivery order).
- The **`Bus`** is a process-global mutex+condvar mailbox: one ordered queue per
  `ChannelKey`. `Recv` blocks until a peer `Send`s — async rendezvous between
  concurrent in-kernel agents, with no new kernel code.

## The capability floor on messages

`Send`/`Recv` (and `Publish`/`Subscribe`) fold a **registered** adjudicator
(`a2aGate`) and ingress admitter (`a2aIngress`) — the *same* registries the kernel
walks for every tool call, so the message floor is first-class in the kernel, not a
side library. The rules are fail-closed:

| Situation | Verdict | Reason |
|---|---|---|
| Send without the negotiated `CapA2ASend` | `Deny` | `DEFAULT_DENY` (no send-right) |
| `TaintQuarantined` body | `Deny` | `TRUST_VIOLATION` (poison never leaves) |
| `ScopeAgent` (private) body to *another* agent's channel | `Deny` | `TRUST_VIOLATION` (widen `Scope` to share) |
| `ScopeFleet`/`ScopeTenant` body, not quarantined, cap held | `Allow` | — |
| Recv without `CapA2ARecv` | `Deny` | `DEFAULT_DENY` (no receive-right) |
| On ingress, a `TaintQuarantined` delivered message | `Quarantine` | held out of context |

The default `Ref` (`Tainted`, `ScopeAgent`) is therefore **undeliverable across
agents by construction** — to share, the sender must *explicitly widen* the body's
`Scope`, an auditable act. An admitted message keeps its `Taint`, so the receiver
cannot re-share it past its `Scope`. Refusals cite the closed core vocabulary; no
new reason is minted into the 12-reason set.

## One shape, three locales

The *same* `Send`/`Recv` serve all three communication locales — only the
`ChannelKey`'s `Locale` + `ID` differ. Sessions and windows are the same mailbox,
addressed differently, not three mechanisms.

| Locale | `ID` is… | What it bridges |
|---|---|---|
| `InKernel` | a rendezvous name within one process | two concurrent goroutine-agents |
| `Session` | a peer's `ToolCall.TraceID` | a cross-session handoff |
| `Window` | a continuation id minted on compaction | an explicit handoff across a context window |

The `Window` case is the interesting one: today fak's continuity across a
context-window compaction is *implicit* (a pruned span pages back in from the
lossless recall store). A `Window` channel makes the handoff **explicit and
adjudicated** — the summarizing window `Send`s its handoff (a summary + open-task
`Ref`) to its own continuation, and the resuming window `Recv`s it, so a quarantined
span cannot ride the handoff into the next window.

## Two delivery shapes (and other options)

Point-to-point (`Send`/`Recv`) is one message, one receiver. `Publish`/`Subscribe`
is its dual — one adjudicated message fanned out as an independent copy to every
current subscriber's private inbox — under the *same* floor (a `Publish` folds the
same `gateSend`, so publishing a private or quarantined body is refused
identically). These are two *delivery shapes* over one floor, not two security
surfaces. Request/reply (correlated `Send` + a reply channel) and shared-state (the
existing `Ref` CAS pool) are the natural further options; they compose from the same
primitives.

## Bounded worker corrections

The orchestrator-to-worker correction path is a typed protocol over the same bus,
not an exception to the trust floor:

- The worker exposes a live `WorkerStatus` row. Its `Digest()` is the status
  witness.
- The orchestrator sends a `CorrectionRequest` that cites that digest, the worker
  id, issue, task id, lane, and a message bounded by `DefaultCorrectionMaxBytes`.
- `SendCorrection` refuses stale or missing status evidence as `UNWITNESSED`,
  oversize text as `OVERSIZE`, malformed shape as `MALFORMED`, and worker/issue/lane
  mismatch as `TRUST_VIOLATION`.
- On allow, the correction is delivered as a shared-scope `a2achan` message. The
  worker `AckCorrection`s with the same correction id and a `WorkerAction` whose
  `correction_id` is the structural "reflected in next action" witness.

This gives an orchestrator a bounded mid-flight steering channel without letting a
raw `SendMessage` launder trust or widen privilege. The issue witness is
`TestCorrectionChannelAckAndActionWitness`: an in-scope correction is allowed,
acked, and reflected in the worker's next action; the paired
`TestCorrectionOutOfScopeStillTrustViolation` proves an out-of-scope correction is
still refused with `TRUST_VIOLATION`.

## Where it sits in the shared-state ladder

`a2achan` is the **live message** rung of fak's shared-state story. It proves that
an addressed value can move between agents under the same floor as a tool call. It
does **not** by itself make a mutable shared whiteboard, durable mailbox, or
collaborative editor.

The next rungs add separate contracts:

- a live shared object needs a stable object name, version base, and deterministic
  update/conflict rule;
- a durable handoff needs a session-image-backed mailbox or task store so a
  `Session`/`Window` message survives a process boundary;
- a disaggregated tier needs digest/provenance/deletion witnesses over bytes held
  outside this process or engine;
- a user-level collaborative surface needs human-authored patches with base
  digests, scope, durability, and typed conflicts.

See [Shared state ladder](shared-state-ladder.md) for the vocabulary that keeps
those layers separate.

## Try it (no key, no model)

```bash
go run ./cmd/a2ademo
```

It exercises point-to-point delivery, the floor refusing a private/quarantined/
uncapped send, a cross-session handoff, a cross-context-window self-handoff, and
pub/sub fan-out — and exits non-zero if any leg fails, so running it *is* a witness:

```
[2] the capability floor (default-deny on messages, like tool calls)
    alpha SEND private -> another agent's channel  -> DENY (TRUST_VIOLATION)
    alpha SEND quarantined -> work                 -> DENY (TRUST_VIOLATION)
    alpha SEND with NO send-right                  -> DENY (DEFAULT_DENY)
```

## Relationship to the fleet A2A edge and MCP

- `a2achan` is the **in-kernel substrate**: in-process, adjudicated, Ref-backed.
- The **fleet A2A HTTP edge** (`tools/fleet_agent_link.py`,
  [`a2a-value-opportunities.md`](a2a-value-opportunities.md)) is the *out-of-kernel*
  projection — discovery, task lifecycle, multi-tenant routing — that should map
  `SendMessage`/`GetTask` onto this substrate, not reinvent the floor.
- **MCP** stays the model/tool/context boundary; it is not the peer-agent channel.

## Honest scope + roadmap

- **Shipped, in-process:** the `InKernel` locale, the capability floor, pub/sub, and
  ctx-aware blocking `Recv` are real and race-tested (`go test -race
  ./internal/a2achan`). `Session`/`Window` share the identical code keyed
  differently and work in-process today.
- **Named next rungs (not claimed):** durable cross-process delivery (a
  session-image-backed mailbox so a `Session`/`Window` message survives a process
  boundary; a compaction trigger that mints the `Window` continuation id); routing
  `Send`/`Recv` as *true* kernel syscalls by wiring the registered-but-dormant
  `abi` Op table (`LookupOp` is never called on the hot path today — the kernel
  dispatches to the engine); and the fleet A2A edge projecting this substrate.

The design deliberately makes **zero ABI edits** and registers **no engine**
(`abi.Engine("")` picks the lowest-id engine, so a "comm" engine would silently
hijack the process default) — it rides the registries the kernel already walks.
