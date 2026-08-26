---
title: "Session clients: full-power attachment and continuity"
description: "The product contract for attaching terminal, fleet, browser, phone, editor, or native clients to one logical fak session without capability loss."
---

# Session clients: full-power attachment and portable continuity

**Status:** normative product contract and generation constraint (2026-08-13)

This document answers one product question: **what does opening a fak session mean?**

It does not mean opening a dashboard card, a reduced chat replica, or a product layered over the
terminal product. It means attaching another client to the same logical session. The terminal is
one client. A Fleet workspace, browser, phone, editor, or future native app is another. A separate
product may deliberately consume a smaller public API, but it must say that it is a separate
product rather than presenting a partial projection as “the session.”

## Value frame

- **For:** an operator who starts work in a terminal and later continues from another screen,
  device, provider account, model, or compute placement.
- **Problem:** fak already has durable session state, control verbs, event streams, provider/account
  routing, and fleet projections, but a UI can still become a second, weaker control plane. A row
  can describe a session without opening the thing the terminal is operating.
- **Today:** terminal session control, Fleet observation, and the proposed per-agent browser console
  are planned as neighboring surfaces. Their shared identity, attachment, parity, and migration
  rules are not one contract.
- **Better because:** every first-party surface attaches through one session protocol. State lives
  behind the attachment, not in a terminal process or browser tab, so changing clients or execution
  placement does not silently fork the work.
- **Witness:** start one session in client A, attach client B by logical session ID, read the same
  addressed event tail and pending interaction, act from B, observe the resulting event in A, then
  move a later execution epoch to a different placement without changing logical session ID.

The real next-best alternative is a browser console specialized to the current gateway routes.
That can look complete quickly, but every control or state field added directly to it creates a
second product contract and makes later device/provider/compute movement harder.

## Problem centrality and all-work checks

**Centrality: Core.** Client-independent managed context is the user-facing form of fak's kernel
boundary, not a convenience shell around it.

| Check | Requirement |
|---|---|
| P1 — managed context | The authoritative transcript, tool effects, pending interactions, budgets, and compaction lineage belong to a logical session, never to one renderer. |
| P2 — net-true efficiency | Reattachment reuses that state and provider-cache lineage where valid; it must not replay the whole conversation merely to reconstruct a UI. Any migration claim includes checkpoint, transfer, warm-up, and lost-cache costs. |
| P3 — bounded adaptation | Provider/account/model/compute changes occur at typed execution-epoch boundaries with policy, capability, budget, and safe-point checks. No client can mutate placement by editing local UI state. |
| P4 — integrated operations | Discovery, attach, control, approval, replay, migration, audit, and recovery use the same session identity and emit the same journaled facts across terminal and non-terminal clients. |

## The invariant

> Selecting a first-party session opens a full session attachment. It does not navigate to a
> separately implemented approximation of that session.

“Full” means **semantic and control parity**, not pixel parity. A phone may lay out information
more compactly than a terminal. It may not omit a currently valid action without declaring the
missing capability and offering a handoff to a capable client. A terminal may render ANSI while a
browser renders structured blocks. Both consume and change the same authoritative state.

For a native fak loop, the canonical seam is a structured session protocol. For a wrapped program
whose only truthful interface is a PTY, fak may expose a lossless PTY attachment mode. The browser
must not scrape terminal output when structured events exist, and fak must not invent structured
controls that the wrapped runtime cannot actually perform.

## One identity, several replaceable bindings

A session is not a PID, terminal, gateway port, provider thread, account, model, machine, or browser
URL. Those are bindings with narrower lifetimes.

| Layer | Identity / state | Lifetime and rule |
|---|---|---|
| Logical session | `session_id`; objective; addressed event journal; transcript/tool-effect lineage; policy; budgets; pending interactions; checkpoint lineage | Stable across client and placement changes. This is what a user opens. |
| Execution epoch | `execution_epoch`; runtime adapter; provider thread/run ID; provider, account reference, model, compute target; process/gateway coordinates; cache lineage | Replaceable at a safe boundary. Every replacement is journaled; old coordinates never become session identity. |
| Client attachment | `attachment_id`; client kind; advertised capabilities; replay cursor; focus/presence; short-lived control lease | Disposable and reconnectable. Disconnecting a client does not end the session. |
| Presentation | viewport, theme, local keymap, notification preference, draft text not yet submitted | Client-local unless explicitly synchronized. It cannot be authoritative execution state. |

Credentials are references resolved under destination policy, never portable session payload. A move
to another device or compute node proves that the destination can resolve an allowed account; it
does not copy bearer tokens through the event journal or snapshot.

### Explicit clear is a new logical session

Replacing a provider thread during resume, compaction, migration, or account/model movement does
not change the logical session. An explicit user `clear`/`new` command does: it declares the prior
conversation finished and opens a new logical session while the guard process remains alive.

The provider adapter must report the typed boundary rather than relying on terminal text. fak keeps
the old session terminal and auditable, switches omitted-trace traffic to a fresh child trace, clears
conversation-local roots and cache affinity, refreshes the context-token axis, and carries cumulative
hard envelopes. The detailed field contract and provider matrix live in
[provider clear/new boundaries](integrations/provider-session-reset.md).

## The attachment contract

All first-party clients use the same conceptual operations, whether the transport is local IPC,
HTTP/SSE plus command POSTs, WebSocket, or a future relay:

1. **Discover** — resolve `session_id` to its current execution epoch and reachable attachment
   endpoint without making the endpoint part of identity.
2. **Describe** — return the durable session descriptor, current placement, pending interaction,
   available controls, and a monotonically addressed event head.
3. **Attach** — authenticate, advertise client capabilities, receive an `attachment_id`, and replay
   from an explicit logical cursor. Delivery IDs and TCP connection lifetime are not replay state.
4. **Observe** — receive the same ordered transcript, tool call/result, control transition,
   approval, budget, checkpoint, placement, and terminal-frame facts visible to every other client.
5. **Act** — submit typed input or a typed control against `session_id` plus expected
   `execution_epoch`. The kernel validates policy and stale-epoch preconditions; the renderer does
   not directly operate a process.
6. **Detach** — release focus/control leases while the logical session and execution continue
   according to policy.
7. **Move** — request a new execution epoch with explicit placement constraints. Checkpoint,
   destination admission, state transfer, credential resolution, and cutover are kernel-owned
   transitions, not UI choreography.

Every successful state-changing request produces an addressed journal event before a client may
present it as complete. Reconnect uses the last applied logical event address. This is the same
replay discipline required by the open run-progress work in #6486.

### Single-writer interaction, multi-reader presence

Many clients may observe concurrently. Exactly one attachment at a time holds the short-lived
**input lease** for an interactive turn or PTY. Taking it is explicit and visible to all clients;
stale clients receive `STALE_EPOCH` or `LEASE_NOT_HELD`, never a best-effort duplicate submission.
Approvals and destructive controls retain their existing policy/confirmation gates even when the
input lease is held.

This is not “one active browser.” A terminal can watch while a phone answers an approval, then take
input back. Presence is ephemeral; history and pending interactions are durable.

## Capability parity is negotiated, not assumed

`Describe` exposes a versioned capability set derived from the runtime and policy, for example:

- transcript/event replay and live stream;
- text input and interrupt;
- pending approval read/respond;
- steer, pause, resume, drain, terminate, checkpoint, and move;
- tool-call detail and artifact transfer;
- structured rendering and/or lossless PTY frames;
- placement choices permitted by policy.

A first-party client must render every advertised action or visibly mark it unavailable with a
reason and a handoff URI/command. It may not silently hide controls to simplify its product. New
kernel controls appear through capability discovery rather than custom frontend releases wherever
possible. A conformance fixture compares terminal and UI clients against the same descriptor and
action corpus.

## Portable continuity and placement changes

A placement is the tuple `(runtime adapter, provider, account ref, model, compute target)`. Changing
any member creates a new execution epoch; it does not create a new logical session.

The move state machine is:

```text
ATTACHED/RUNNING
  -> SAFE_POINT_REQUESTED
  -> CHECKPOINTED(address, digest, source_epoch)
  -> DESTINATION_ADMITTED(policy, capabilities, credentials, budget)
  -> RESTORED(new_epoch, placement, cache_lineage)
  -> CUTOVER_COMMITTED
  -> RUNNING
```

Failure before `CUTOVER_COMMITTED` leaves the source epoch authoritative when it is still healthy.
Failure after cutover is recovered from the committed checkpoint and journal. An active provider
call is not magically portable: fak waits for or induces a supported safe point, or returns a typed
`MOVE_UNSAFE` refusal. Model/provider changes disclose semantic degradation (unsupported tools,
context limit, unavailable provider-side cache) before commit.

State classes during movement:

- **Must move:** logical descriptor, addressed journal, transcript/tool effects, pending approvals,
  budgets, policy references, checkpoint lineage, and idempotency keys.
- **May be reconstructed:** read projections, search indexes, rendered terminal frames, derived
  health summaries.
- **Placement-local:** process handles, sockets, GPU allocations, provider cache handles, and
  secrets. Their replacement or loss is explicit in the new epoch record.

## Product boundaries

### First-party session surfaces

The terminal, Fleet workspace drill-in, browser console, and mobile client are views of this
contract. Their route is conceptually:

```text
select session -> resolve session_id -> describe -> attach/replay -> full live surface
```

A Fleet overview remains a projection. The moment the operator selects a row, it crosses to the
session attachment instead of growing its own per-row controls. A per-agent gateway URL is a
current transport coordinate discovered for an epoch, not the bookmark users carry between
devices.

### Clearly separate products

A read-only status wall, Slack digest, embedded approval widget, or third-party integration may
intentionally consume a subset. It must identify itself as a projection/integration, name its
capability ceiling, and never claim that selecting its card is equivalent to opening the session.
It can deep-link or hand off to a full client.

## Required end-to-end spine

The first implementation slice is deliberately vertical, not a frontend mock:

1. Add a durable logical-session descriptor and discovery record that can resolve a current local
   execution epoch without treating `gateway_url` as identity.
2. Implement `fak session open SESSION_ID` as the reference terminal client using describe,
   replay-from-address, attach, input-lease, one text input, and detach.
3. Implement one minimal browser page over exactly the same operations and capability descriptor;
   selecting its fixture session resumes at the same event address and can submit the next input.
4. Capture a two-client witness: terminal creates/opens, browser attaches, browser acts, terminal
   observes the same addressed result after reconnect.
5. Only then expand renderer polish, remote relay, migration envelopes, and mobile layout.

The reference CLI is essential: it prevents the protocol from becoming “whatever the web page
needs” and makes terminal/UI equivalence executable.

## Acceptance witnesses for the full objective

The objective is not complete until all are captured against real objects rather than renderer-only
unit tests:

- **Client equivalence:** terminal and browser attach to one `session_id`, report the same
  `execution_epoch`, event head, pending interaction, and capability digest.
- **Cross-client action:** an input/control submitted from either client is addressed once and
  observed by the other after disconnect/reconnect; duplicate replay causes no duplicate effect.
- **Device continuity:** a second authenticated device discovers by logical ID and resumes without
  copying a local transcript or knowing the old gateway port.
- **Provider/account/model continuity:** a safe-point move changes those bindings, preserves logical
  ID and journal lineage, and records capability/cache changes.
- **Compute continuity:** restore on another sanctioned compute target preserves the same invariants
  and has a witnessed rollback/refusal path.
- **Parity:** every advertised runtime action is available in both reference terminal and browser
  clients or visibly refused with the same typed reason.
- **Recovery:** client loss, source-process loss, stale epoch, and interrupted cutover each replay to
  one authoritative state with no double input or approval.
- **Security:** discovery authorization, input leases, policy checks, destination credential
  resolution, and secret non-export are independently tested.

## Relationship to existing plans

- [`fleet-ui-generation-plan.md`](fleet-ui-generation-plan.md) remains the overview/read-model
  generation map. Its drill-in destination is this attachment contract; controls do not accrete in
  the Fleet renderer.
- [`gateway-port-ui-plan.md`](gateway-port-ui-plan.md) remains useful for a same-machine console
  spine. Its per-agent URL is a discovery bootstrap for the current execution epoch, not durable
  identity, and its page must grow by consuming this contract rather than custom routes.
- [`notes/PORTABLE-SESSION-IMAGE-AND-SNAPSHOT-2026-06-24.md`](notes/PORTABLE-SESSION-IMAGE-AND-SNAPSHOT-2026-06-24.md)
  supplies checkpoint/image groundwork. A portable image is an input to `Move`, not itself the
  user-visible session.
- [`notes/SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md`](notes/SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md)
  supplies durable control-state groundwork. This contract makes that state client-independent.
- [`notes/CONCEPT-STUDY-HERMES-WEBUI-2026-08-11.md`](notes/CONCEPT-STUDY-HERMES-WEBUI-2026-08-11.md)
  identifies replay-addressed reconnect and runtime-owned state as the immediate external lesson.

## Non-goals

- Rebuilding a shell, editor, issue tracker, or provider UI inside fak.
- Making every integration a full session client.
- Claiming active-token-stream migration where a runtime has no safe checkpoint.
- Treating synchronized UI preferences or copied transcript text as session portability.
- Freezing transport details before the two-client spine proves the semantic contract.

## Shippable implementation backlog

The contract is split into independently witnessed leaves rather than silently deferred prose:

- [#6547](https://github.com/anthony-chaudhary/fak/issues/6547) — the immediate two-client spine:
  logical descriptor/discovery, `fak session open`, minimal browser client, addressed replay, and
  input lease.
- [#6548](https://github.com/anthony-chaudhary/fak/issues/6548) — safe-point execution-epoch moves
  across provider, account, model, and compute placement with admission and rollback.
- [#6549](https://github.com/anthony-chaudhary/fak/issues/6549) — authenticated cross-device
  discovery and remote attachment without making an endpoint the durable identity.
- [#6550](https://github.com/anthony-chaudhary/fak/issues/6550) — generated capability-parity and
  recovery conformance across reference terminal and browser clients.

[#6486](https://github.com/anthony-chaudhary/fak/issues/6486) owns the replay-address seam reused by
these leaves; [#6476](https://github.com/anthony-chaudhary/fak/issues/6476) remains the read-only
Fleet overview spine rather than absorbing session operation.
