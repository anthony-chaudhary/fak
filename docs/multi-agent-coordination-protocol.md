---
title: "RFC: the fak Multi-Agent Coordination Protocol (D-007)"
description: "The single normative spec for agent-to-agent coordination in fak: the message format, the shared-state API, and the coordination primitives — all carried over the same default-deny capability floor that gates a tool call. Binds the three shipped pillars (a2achan, sharedtask, comm) under one protocol and maps issue #241's acceptance to its shipped artifacts + test witnesses."
---

# RFC: the fak Multi-Agent Coordination Protocol

> **Status:** Draft (rungs 1–3 shipped in-process; durable cross-process backing is the named next rung).
> **Issue:** [#241](https://github.com/anthony-chaudhary/fak/issues/241) · **Slug:** D-007 · **Epic:** [#304](https://github.com/anthony-chaudhary/fak/issues/304) (Track D — Agent Framework Parity).
> **Sibling epic:** [#639](https://github.com/anthony-chaudhary/fak/issues/639) (MPI-shaped message-passing primitives).
> **House rule:** every primitive named here is on disk with a package test; the honest-scope section says plainly what is in-process today versus durable across a process boundary. No throughput or latency number is asserted here.

This is the authoritative spec the issue's *"RFC/spec document"* acceptance names. The
three other acceptance items — message passing, shared KV/cache space, coordination
primitives — already ship as test-witnessed kernel packages; until now they were
described only in scattered design docs. This RFC pulls them into **one protocol** and
states the invariant that makes it fak-native: **every coordination act is an
adjudicated tool call — fail-closed, scope- and taint-bounded, refusable with a closed
reason vocabulary.** Coordination in fak is not a side library with its own security
surface; it rides the registries the kernel already walks for every tool call.

---

## 1. Why a protocol (the gap it closes)

Most "agent-to-agent" work is a *transport*: a way for two agents to find each other and
move bytes over HTTP. fak already has that story at the fleet edge
([`a2a-value-opportunities.md`](a2a-value-opportunities.md), the out-of-kernel Agent Link
in [`agent-machine-link-protocol.md`](agent-machine-link-protocol.md)). What was missing
is the **in-kernel substrate** the transport projects onto: a way for one agent to hand a
value to another, share mutable state, and synchronize a wave — **under the same
default-deny floor that gates a tool call**, so a poisoned result or a private payload
cannot cross an agent boundary just because it travelled through a "coordination" call
instead of a "tool" call.

fak had every *half* but no whole: `abi.Ref` carries the `Taint`+`ShareScope` a
cross-agent message needs but never *routes* anywhere; the async `Submit`/`Reap` seam is
1:1 with no recipient identity; `session`/`recall`/the vDSO fan-out move drive state,
read-only memory images, or cache-invalidation broadcasts — none *delivers* an addressed
value from agent A to a different agent B. This protocol is the missing whole, assembled
from the existing currency (`Ref` provenance) and the existing registries (the
adjudicator + result-admitter chains).

The protocol has three layers, each a shipped package:

| Layer | What it carries | Package | Try it |
|---|---|---|---|
| **§3 Message passing** | one addressed value, now, A→B | [`internal/a2achan`](https://github.com/anthony-chaudhary/fak/blob/main/internal/a2achan/a2achan.go) | `go run ./cmd/a2ademo` |
| **§4 Shared state** | a named record / KV space many agents co-edit | [`internal/sharedtask`](https://github.com/anthony-chaudhary/fak/tree/main/internal/sharedtask) | `python tools/shared_task_contract.py validate-sequence examples/shared-task-record` |
| **§5 Coordination primitives** | broadcast / scatter / gather / barrier over a wave | [`internal/comm`](https://github.com/anthony-chaudhary/fak/blob/main/internal/comm/comm.go), [`internal/agenttopo`](https://github.com/anthony-chaudhary/fak/blob/main/internal/agenttopo/agenttopo.go) | `go test ./internal/comm` |

---

## 2. The adjudication invariant (the spine)

Everything below obeys one rule, and the rule is the contribution:

> **A coordination act is a synthetic tool call.** A `Send`, a `Recv`, a `Publish`, a
> `Broadcast`, a `Scatter`, a `Barrier` — each folds the **same** registered adjudicator
> + ingress admitter the kernel walks for a real `tool_call`. There is no collective and
> no message that is exempt from refusal.

Three consequences are normative for any conforming implementation or adapter:

1. **Fail-closed by default.** The default `abi.Ref` is `(Tainted, ScopeAgent)` — private
   and quarantine-eligible. Such a body is **undeliverable across an agent boundary by
   construction**. To share, the sender must *explicitly widen* the body's `Scope`
   (`ScopeFleet` / `ScopeTenant`), an auditable act — never an implicit side effect of
   "sending."
2. **Provenance rides the value, unchanged.** A coordination op copies an `abi.Ref`
   through; it never re-marshals or re-labels the body. *Sharing a result shares its
   taint* — an admitted message/broadcast keeps its `Taint`, so a receiver cannot
   re-share it past its `Scope`. Quarantined bytes are **held out of the receiver's
   context** on ingress, never admitted.
3. **Refusal is a value from a closed vocabulary.** A denied coordination act returns an
   `abi.Verdict` citing the core reason set (`DEFAULT_DENY` for an un-negotiated
   capability; `TRUST_VIOLATION` for a scope/taint breach). **No new reason is minted** —
   the 12-reason core set is unchanged, so a coordination refusal is auditable by exactly
   the machinery that audits a tool-call refusal.

This is why the protocol is fak-native rather than a generic message bus: the security
floor is *the same object* on the coordination path and the tool-call path.

---

## 3. Message passing — the message format (`a2achan`)

The live-message rung: deliver one addressed value from agent A to a different agent B.

### 3.1 Addressing and the message

A mailbox is named by a `ChannelKey`; a delivered unit is a `Message`:

```go
type ChannelKey struct {
    Locale Locale  // InKernel | Session | Window
    ID     string  // rendezvous name | peer TraceID | window continuation id
}

type Message struct {
    From string      // the sending principal
    To   ChannelKey  // the destination mailbox
    Body abi.Ref     // the payload — its Taint + Scope ride unchanged
    Seq  uint64      // per-bus monotonic; fixes a deterministic delivery order
}
```

Two keys are equal iff **both** fields match: a `Session` channel and an `InKernel`
channel that happen to share an `ID` are distinct mailboxes — the `Locale` is part of the
identity. The `Body` is an `abi.Ref` (inline or CAS-backed); its `(Taint, Scope)` are the
share bound and are never widened by transit.

### 3.2 One shape, three locales

The *same* `Send`/`Recv` serve all three communication locales; only the key differs.
Sessions and windows are the same mailbox addressed differently — **not** three
mechanisms.

| Locale | `ID` is… | What it bridges | Status |
|---|---|---|---|
| `InKernel` | a rendezvous name in one process | two concurrent goroutine-agents | **shipped, race-tested** |
| `Session` | a peer's `ToolCall.TraceID` | a cross-session handoff | code-shared; durable backing = next rung |
| `Window` | a continuation id minted on compaction | an explicit handoff across a context window | code-shared; compaction trigger = next rung |

### 3.3 Two delivery shapes

Point-to-point and pub/sub are two *delivery shapes* over **one** floor, not two security
surfaces:

- **Point-to-point** — `Send(ctx, from, to, body, caps…) → Verdict` and
  `Recv(ctx, to, caps…) → (Message, Verdict, error)` (ctx-aware blocking; `TryRecv` is
  the non-blocking dual). One message, one receiver.
- **Pub/sub** — `Subscribe(topic) → (inbox, cancel)` and
  `Publish(ctx, from, topic, body, caps…) → (Verdict, fanout)`. One adjudicated message
  fanned out as an independent copy to every current subscriber's private inbox. A
  `Publish` folds the **same** send-time gate, so publishing a private or quarantined
  body is refused identically.

### 3.4 The capability floor on messages

`Send`/`Recv`/`Publish` fold a registered adjudicator (`a2aGate`, tools `a2a.send` /
`a2a.recv`) and ingress admitter (`a2aIngress`). The capabilities are
`CapA2ASend = "a2a.send"` and `CapA2ARecv = "a2a.recv"`, negotiated like any other. The
verdict table is normative:

| Situation | Verdict | Reason |
|---|---|---|
| `Send` without the negotiated `CapA2ASend` | `Deny` | `DEFAULT_DENY` (no send-right) |
| `TaintQuarantined` body | `Deny` | `TRUST_VIOLATION` (poison never leaves) |
| `ScopeAgent` (private) body to *another* agent's channel | `Deny` | `TRUST_VIOLATION` (widen `Scope` to share) |
| `ScopeFleet`/`ScopeTenant` body, not quarantined, cap held | `Allow` | — |
| `Recv` without `CapA2ARecv` | `Deny` | `DEFAULT_DENY` (no receive-right) |
| On ingress, a `TaintQuarantined` delivered message | `Quarantine` | held out of the receiver's context |

**Witness:** `internal/a2achan/a2achan_test.go` (determinism, fail-closed default,
taint/scope enforcement, async rendezvous, ingress quarantine-hold; `go test -race
./internal/a2achan`). Reference design: [`a2a-in-kernel-channel.md`](a2a-in-kernel-channel.md).

---

## 4. Shared state — the shared KV/cache space API (`sharedtask`)

The shared-state rung. fak keeps five related-but-different senses of "shared state"
separate (the [shared-state ladder](shared-state-ladder.md)) so they are not collapsed
into one over-claimed feature:

| Rung | Meaning | Shipped shape |
|---|---|---|
| Live addressed message | one value delivered now | §3 `a2achan` `Send`/`Recv`, `Publish`/`Subscribe` |
| Live shared object | a named mutable cell during one run | compose from refs + messages today; first-class region work is planned |
| Durable handoff | state that survives a process/session/window boundary | session-image + snapshot primitives; the `sharedtask` journal is a task-local materialized handoff |
| Disaggregated state | bytes live outside the record | shared refs carry digest, taint, scope, store, and a deletion certificate |
| User-level collaboration | a human + agents co-edit task state | the shared-task-record contract + in-memory fold |

### 4.1 The shared record / KV space

A coordinated wave's shared KV space is a **shared task record**: a single addressable
record (`task_id`, monotonic `rev`, a scoped body `Ref`, and append-only `notes` /
`artifacts` / `open_decisions`) that many agents and humans co-edit by **patch**, not by
unstructured chat. The envelopes are versioned JSON (`fak.shared-task.v1`,
`fak.shared-patch.v1`, `fak.shared-patch-result.v1`, `fak.shared-event.v1`,
`fak.shared-artifact-ref.v1`, `fak.shared-task-journal.v1`); the normative contract +
worked fixtures are in [`shared-task-record-contract.md`](shared-task-record-contract.md).

### 4.2 Merge semantics (normative)

| Operation | Auto-merge? | Rule |
|---|---:|---|
| append note / artifact / open decision | yes | id must be new; body is a scoped ref |
| replace `/title` or `/state` | no | requires the current base `rev`; a stale writer gets a typed conflict |
| replace `/body_ref` | no | external refs need a deletion certificate |
| replace an open-decision state | no | stale or missing decisions conflict |

Append-only edits commute on new ids; scalar edits are current-base and return a **typed
conflict** (base, current, proposed) when stale — so concurrent agents converge
deterministically instead of last-writer-wins clobbering. Scoped views (`View`,
`EventsView`, `SubscribeScopedView`) redact a snapshot and its event history by the
reader's scope, so a tenant-scoped reader never sees a fleet-scoped body.

### 4.3 The KV/cache connection

"Shared KV/cache space" has two faces in fak, both honored here:

- **Shared *state* KV** — the record above: a scoped, taint-tracked, patch-merged KV that
  agents read and write under the §2 invariant. Disaggregated bytes (`l3-kv` store) are
  admitted only with a digest and a deletion witness.
- **Shared *cache* KV** — the cross-agent prefix-cache reuse fak already ships: do the
  shared prefill work once, later agents read it for free (the addressable, bit-exact KV
  cache and the radix prefix pool). That reuse is the *performance* dual of this protocol;
  this RFC governs the *coordination* and *provenance* of shared bytes, not the cache
  arithmetic (see the cache docs for the measured reuse).

**Witness:** `internal/sharedtask/sharedtask_test.go`, `internal/sharedtask/live_test.go`;
the executable contract validator `tools/shared_task_contract.py` over
`examples/shared-task-record`.

---

## 5. Coordination primitives — the wave collectives (`comm`, `agenttopo`)

The synchronize-a-wave rung. A [`comm.Group`](https://github.com/anthony-chaudhary/fak/blob/main/internal/comm/comm.go) is an **ordered
set of member agents**: `Rank` is a member's position in the *sorted* member set, so the
same members always get the same ranks regardless of arrival order — rank is a
deterministic function of the member identities, never of arrival order or a member's
output.

### 5.1 The collectives

Each collective routes its admitting tool call through `abi.Kernel.Submit` (the
adjudication chokepoint) — there is no collective exempt from refusal. The `I*` variants
return `StatusPending` handles completed via `Kernel.Reap`; no ABI edit is needed.

| Primitive | Shape | Floor behavior |
|---|---|---|
| `Broadcast(payload)` | one `Ref` to every member | **refuses** to broadcast a `ScopeAgent`/private `Ref` to a multi-member group |
| `Scatter(goals)` | one per-rank goal `Ref` | per-rank `Submit`; each adjudicated |
| `Gather(outputs, reduce)` | fold member outputs in **rank order** | layout is deterministic even though member text is not |
| `Barrier()` | one adjudicated read-back descriptor per rank | a `dos-witness-claim`-shaped arrival fold, **not** a scheduler lock |
| `Split(color)` / `SplitLane()` | partition the group by color → lane | each color binds a `dos.toml` lane; overlapping lanes serialize by refusal at the arbiter |
| `Spawn()` | mint rank-stamped `Membership` for a wave | — |

### 5.2 Topology: declare vs search

- **Declare** — [`internal/agenttopo`](https://github.com/anthony-chaudhary/fak/blob/main/internal/agenttopo/agenttopo.go) declares a
  *named, validated DAG* over a `comm.Group`: who may hand a result to whom, every edge
  checked against the group, cycles refused, declaration order preserved. (The
  `MPI_Graph_create` analogue.)
- **Search** — `cmd/topobench` + `turnbench.TopologyGenome` *optimize* an anonymous shape,
  ranking topologies by measured prefix-reuse savings capped at the corpus divergence
  frontier — never an extrapolated number.

The full MPI-communicator analogy (the lane lease as `MPI_Comm_split`, `ShareScope` as the
communicator isolation scope) is documented in
[`comm-as-mpi-split.md`](comm-as-mpi-split.md), with the honest line that **no bytes move
at the lane-lease layer** — a lease coordinates *who may write which files*, it does not
transport a message.

**Witness:** `internal/comm/comm_test.go`, `internal/agenttopo/agenttopo_test.go`.

---

## 6. Conformance — issue #241 acceptance mapping

A conforming claim for D-007 is *evidence-backed*: each acceptance item maps to a shipped
artifact and a runnable witness, not to prose.

| Acceptance item | Shipped artifact | Witness |
|---|---|---|
| **Message passing between agents** | `internal/a2achan` — `Send`/`Recv`/`TryRecv`, `Publish`/`Subscribe`, three locales, the capability floor | `go test -race ./internal/a2achan`; `go run ./cmd/a2ademo` |
| **Shared KV/cache space** | `internal/sharedtask` shared record + scoped/taint-tracked refs + journal; cross-agent prefix-cache reuse | `go test ./internal/sharedtask`; `python tools/shared_task_contract.py validate-sequence examples/shared-task-record` |
| **Coordination primitives** | `internal/comm` Group collectives (broadcast/scatter/gather/barrier/split/spawn) + `internal/agenttopo` declared topology DAG | `go test ./internal/comm ./internal/agenttopo` |
| **RFC/spec document** | **this document** | renders + link-clean; binds the three pillars under the §2 invariant |

---

## 7. Honest scope, non-goals, and the roadmap

**Shipped (in-process):** the `InKernel` message locale, the message capability floor,
pub/sub fan-out, the `sharedtask` patch fold + scoped views + materialized journal, and
the `comm` adjudicated collectives + `agenttopo` declared topology — all race/contract
tested.

**Not claimed here:**

- **This is not MPI.** `comm`'s collectives borrow collective *names* and rank-order
  *structure*; they inherit no interconnect, message-rate, progress, or
  collective-latency property. They are explicitly **not** `internal/model`'s `DistComm`
  (the real cross-process tensor collective whose ranks are GPU shards of one model). A
  `comm.Group`'s ranks index detached OS processes that communicate only through git and
  leases.
- **Durable cross-process delivery is the named next rung.** The `Session`/`Window`
  locales share the `InKernel` code path and work in-process today; a session-image-backed
  mailbox (so a `Session`/`Window` message survives a process boundary) and the compaction
  trigger that mints a `Window` continuation id are **not** shipped.
- **No networked task-store daemon, no external L3 transport, no browser/editor UI** for
  the shared record — it is an in-memory reference fold plus a data-only journal.
- **No new ABI surface.** The protocol registers no `abi` engine and makes zero ABI edits;
  routing `Send`/`Recv` as *true* kernel syscalls (wiring the registered-but-dormant `abi`
  Op table) is a named future rung, not a current claim.

**Where the fleet edge fits:** the out-of-kernel A2A HTTP edge
([`a2a-value-opportunities.md`](a2a-value-opportunities.md),
[`agent-machine-link-protocol.md`](agent-machine-link-protocol.md)) is the projection of
this substrate — discovery, task lifecycle, multi-tenant routing — and should map its
`SendMessage`/`GetTask` onto §3/§4, **not** reinvent the floor. MCP stays the
model/tool/context boundary; it is not the peer-agent channel.

---

## 8. References

- **Message passing:** [`a2a-in-kernel-channel.md`](a2a-in-kernel-channel.md) ·
  [`internal/a2achan/doc.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/a2achan/doc.go)
- **Shared state:** [`shared-state-ladder.md`](shared-state-ladder.md) ·
  [`shared-task-record-contract.md`](shared-task-record-contract.md) ·
  [`internal/sharedtask`](https://github.com/anthony-chaudhary/fak/tree/main/internal/sharedtask)
- **Coordination primitives:** [`comm-as-mpi-split.md`](comm-as-mpi-split.md) ·
  [`internal/comm/doc.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/comm/doc.go) ·
  [`internal/agenttopo/doc.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/agenttopo/doc.go)
- **The fleet edge (out-of-kernel projection):**
  [`a2a-value-opportunities.md`](a2a-value-opportunities.md) ·
  [`agent-machine-link-protocol.md`](agent-machine-link-protocol.md)
- **Track-D status + the sibling epic:**
  [`notes/track-d-agent-framework-parity-tracking-304.md`](notes/track-d-agent-framework-parity-tracking-304.md)
  · epic [#639](https://github.com/anthony-chaudhary/fak/issues/639)
