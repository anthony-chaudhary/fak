---
title: "The session read/query/observe plane — how an outside process reads a running agent, safely"
description: "Design note + SOTA survey for the read-side sibling of the out-of-band operator control plane (#2753). Once a session is running, how does an OUTSIDE PROCESS — a peer agent, a monitor, a supervisor, an indexer, a CI gate — read its transcript, query parts of it, and subscribe to its events, without becoming an exfiltration or injection channel? Benchmarks fak's shipped read seams (fak_context_* MCP tools, the /v1/fak read routes, the coherence/audit CDC feeds, the resume identity map) against the field, names the honest gaps, and sequences a closed, permissioned, evidence-first read/query/observe vocabulary."
---

# The session read/query/observe plane — how an outside process reads a running agent, safely

The out-of-band operator control note
([`OUT-OF-BAND-OPERATOR-CONTROL-2026-07-05.md`](OUT-OF-BAND-OPERATOR-CONTROL-2026-07-05.md),
epic #2753) answered one half of "how does something outside a running session interact
with it": the **write** half — a closed, typed, capability-checked, witnessed set of
*control* operations (pause / resume / redirect / throttle / budget …). This note is
its mirror: the **read** half. Once a session is running, how does an outside process —
not only a human operator, but a **peer agent, a fleet monitor, an external supervisor,
a transcript indexer, a CI gate, a dashboard** — *read* what it is doing, *query parts*
of its transcript and context, and *subscribe* to its events?

The one-line thesis: **fak already owns the hard substrate for a read plane (a
trace-addressed session, a taint-stamped context store, two cursor-drained CDC feeds, a
durable UUID↔trace identity map, and the `dos_status`/`dos_verify` "evidence not
self-report" discipline) — but the read surface is fragmented, mostly metadata-only,
process-ephemeral, and weakly permissioned. The frontier is a closed, capability-scoped,
taint-safe, evidence-first read/query/observe *vocabulary* over a session's transcript,
context, and decisions — the exact read-side analog of the closed control vocabulary the
control plane is building.** And it is what makes the interesting end possible: a
`read → decide → control` external supervisor that reads a session as *evidence*, checks
it, and steers it — a loop, not a dashboard.

## The interaction spectrum (the concept the goal names)

"How outside processes interact with the agent, from simple (read the transcript / query
parts of it) to much more complex" is a spectrum with four rungs, ordered by how much the
outside process *couples* to the running session:

1. **READ** — pull a snapshot of session state (last N turns, context accounting, run
   drive-state, the dropped originating task). Stateless, one-shot.
2. **QUERY** — ask a *structured question* over the transcript/context/decisions without
   slurping the whole thing ("which tools failed", "what files were touched", "the last
   decision about auth", "spans mentioning X"). A projection, not a dump.
3. **OBSERVE** — subscribe to a *filtered event stream* and be woken as the session
   emits turns / tool-terminals / decisions. Push, long-lived, cursor-resumable.
4. **CONTROL** — act on what was read: steer / redirect / hold / approve. The write
   plane (#2753). The capstone is closing 1–3 into 4: an external supervisor loop.

Rungs 1–3 are this note's plane; rung 4 is #2753; the capstone wires them. The rest of
computing separates these too — Temporal splits **Query** (sync read-only) from **Signal**
(async write) from **Update** (validated read-write); A2A splits `tasks/get` (read) and
`tasks/resubscribe` (observe) from `tasks/cancel` (control). fak should draw the same line
and, uniquely, make every read *evidence-qualified* and *taint-safe* by construction.

## Why fak is unusually well-positioned (the substrate already exists)

- **A trace-addressed session.** Everything is keyed by the gateway-minted `X-Trace-Id`
  (`internal/gateway/gateway.go`, `traceSeq`); the "default trace" resolves to the wrapped
  `fak manage` session, so a co-located tool needs no identity to read its own session.
- **A taint-stamped context store.** The inbound steer bus already screens every append
  through `ctxmmu.ScreenBytes`, stamps `abi.TaintTainted`/`TaintQuarantined`, and writes a
  hash-chained quarantine journal that stores *only reason/stub/length, never the raw
  bytes* (`internal/gateway/steer_class.go`). The read plane needs the **symmetric outbound
  discipline**: never emit sealed/quarantined span bytes outward. The taint high-water is
  already observable at `GET /v1/fak/trace/{trace_id}` (`observeTrace`).
- **Two cursor-drained CDC feeds.** `GET /v1/fak/changes` (the live coherence bus,
  `CoherenceEvent`, principal-scoped, `internal/gateway/coherence.go`) and
  `GET /v1/fak/events` (the durable hash-chained audit journal, `journal.Row`), with a
  Debezium-compatible projection (`internal/gateway/debezium.go`, #3171) and a documented
  five-habit consumer contract
  ([change-data-capture-for-agents](../explainers/change-data-capture-for-agents.md)). The
  *observe* transport is solved — for coherence/audit events. It has never been pointed at
  **transcript** events.
- **A durable identity join.** `internal/resume/identitymap.go` is an append-only,
  GC-immune `resume_identity.jsonl` mapping a Claude Code transcript UUID ↔ gateway trace
  (`IdentityRow{uuid, trace, handle, account, via}`, `FoldIdentity`). It is *how* an
  external process would address a session — but it is only readable off disk.
- **Evidence, not self-report.** `dos_status` folds a run digest with **no `claimed`
  field by construction** ([status-a-peer-can-trust](../explainers/status-a-peer-can-trust.md));
  `dos_verify`/`dos_commit_audit`/`dos_recall` are read-models over the log. The read plane
  should inherit this: an external read returns *witnessed* facts, so a supervisor's
  decisions rest on evidence rather than on the agent's narration of itself.

## What fak ships today (honest inventory)

| Read op | fak surface (verified seam) | Status |
|---|---|---|
| Context accounting (one session) | `fak_context_value` MCP + `GET /v1/fak/ctxvalue?trace=` (`CtxValueReportFor`, `ctxvalue.go:529`) — tokens/turns/phase/step-advice, **no message text** | **shipped** |
| Context accounting (all sessions) | `GET /v1/fak/ctxvalue` (no trace ⇒ `ctxValueSnapshot`) | **shipped** |
| Dropped-span index | `fak_context_spans` (`ctxspans.go:81`) — SAFE metadata only (`id/descriptor/bytes/sealed/tombstoned/restorable`), **never span bytes** | **shipped** |
| Restore one dropped span → bytes | `fak_context_restore` (`ctxrestore.go:152`) — the **only** wire seam that returns transcript *content*; trust-gated (sealed/tombstoned refused), and only the compaction-dropped first-user-turn | **shipped, narrow** |
| Run drive-state (one / all) | `GET /v1/fak/session/{id}` (`SessionState`), `GET /v1/fak/sessions` (`ListSessions`) — run-state/budget/priority/pace/parent | **shipped** |
| Session-table revision stream | `GET /v1/fak/session/changes` — cursor-drained Rev bumps | **shipped** |
| Cross-agent coherence feed | `GET /v1/fak/changes` + `fak_changes` (principal-scoped CDC) | **shipped** |
| Durable audit journal feed | `GET /v1/fak/events` (hash-chained `journal.Row`) + Debezium projection | **shipped (#3171)** |
| Taint high-water (one trace) | `GET /v1/fak/trace/{trace_id}` (`observeTrace`) | **shipped** |
| Run digest, no self-report | `dos_status` (liveness/progress/region/resume) | **shipped** |
| Resume/heal self-observe | `fak_resume_history` MCP | **shipped** |
| Lifecycle "which sessions ran" | `internal/sessionjournal` + `fak sessionjournal report` (LIVE/CRASHED/STALE/CLOSED) — **filesystem only** | **shipped, off-wire** |
| UUID ↔ trace identity join | `internal/resume/identitymap.go` — **filesystem only, no route** | **shipped, off-wire** |
| Raw transcript parse | `internal/resume/transcript/transcript.go` (`LoadFile`/`LoadFileTail`) over `~/.claude*/projects/*.jsonl` — **filesystem only** | **shipped, off-wire** |
| **Live full/partial transcript query over the wire** | — (must read `~/.claude*/projects/*.jsonl` off disk) | **gap** |
| **Durable store behind span/context reads** | in-memory `s.ctxRestore` / `s.ctxValue`, generationally reset; lost on restart / session end | **gap** |
| **Cross-trace enumeration + addressing over the wire** | identity map is off-disk; caller must already know the trace id | **gap** |
| **Per-principal scoping / taint-safe outbound on the read seams** | context tools have no per-principal ACL; on default loopback (no `RequireKey`) any local process can `restore` the default trace's dropped originating task | **gap** |
| **Transcript-level event subscribe / re-attach for a peer** | the CDC feeds carry coherence/audit, not turns/tool-calls/decisions | **gap** |
| **Session state as MCP resources (expose-as-view for peers)** | — | **gap** |
| **read → decide → control external supervisor loop** | read and control planes exist but are unbridged; no documented, regime-gated supervisor pattern | **gap** |

Auth baseline (the load-bearing caveat): every gateway read is wrapped in
`withAuth`, but auth only engages when `RequireKey` is configured
(`internal/gateway/http.go:316`). The common local `fak manage` deployment sets no key, so
**every read seam above is reachable unauthenticated by any loopback process**, and none
of the context tools carry a per-principal ACL. That is fine for a single-tenant laptop
and a latent hazard the moment a session's transcript is worth reading from another
principal.

## The state of the art (2025–26), read side

The control note surveyed the write side; here is where the field is on *reading* a live
agent:

- **Temporal — Query.** A first-class, synchronous, **read-only** handler on a running
  workflow, strictly separated from Signal (write) and Update (validated read-write).
  Queries do not mutate history and cannot have side effects. This is the canonical
  "read a running execution without perturbing it" primitive.
- **A2A — `tasks/get` + `tasks/resubscribe`.** The task *is* the read surface: `tasks/get`
  reads current state/artifacts; `tasks/resubscribe` re-attaches a dropped client to the
  same task's event stream. Reading and re-observing are distinct verbs from `tasks/cancel`.
- **AG-UI — `STATE_SNAPSHOT` / `STATE_DELTA`.** A typed event channel for the agent↔UI
  read path: a full snapshot then RFC-6902 JSON-Patch deltas, so an observer keeps a live
  mirror of agent state cheaply. The push-observe rung as a protocol.
- **LangGraph — `get_state` / `get_state_history` / streamed state.** The checkpointer
  makes the run's working state externally readable and *time-travelable* (read any prior
  checkpoint), and `stream_mode` exposes values/updates/messages as they are produced.
- **MCP — resources + `notifications/progress`.** The standard way one agent exposes
  readable state to another is as **resources** (addressable, listable, readable URIs) plus
  a request-scoped progress channel. fak already speaks MCP; a session's queryable
  transcript/context is a natural resource tree.
- **CDC / Debezium — log-based read models.** "Don't ask the writer what changed; tail the
  authoritative log by offset." fak already implements this for coherence/audit
  (the [CDC explainer](../explainers/change-data-capture-for-agents.md)); the unclaimed move
  is to add the **transcript** as a third feed on the same cursor discipline.
- **Observability (OTel logs/traces, `journalctl`, `kubectl logs --follow`).** The generic
  "tail a process's structured output by cursor, filter, follow" shape the OBSERVE rung
  generalizes — with fak's twist that a filtered read is *taint-screened*, so following a
  session can never surface a quarantined span.

The convergent shape: **a read is offset-addressed, projection-not-dump, and separated
from write.** fak's differentiators, available to no one else on this list: every read can
be **taint-safe** (screened against the same quarantine ledger the write path stamps) and
**evidence-qualified** (OBSERVED/WITNESSED, no self-report), because the kernel already
sits between the agent and its record.

## The core design move: a closed read/query/observe vocabulary

Mirror the control plane's central move. The control plane replaced freeform prose with a
closed set of typed control ops, each with four fixed properties (who may send, when it
applies, the witness-of-applied, the structured refusal). The read plane replaces the
current grab-bag of ad-hoc routes and in-memory tools with a **closed set of typed read
ops**, each with four fixed properties:

- **who may read it** — a capability + principal on the request (an external reader is a
  principal, exactly like an operator sender is), so a peer reads only what it is scoped
  to, and a cross-tenant read is refused, not merely undocumented;
- **what it discloses** — a stated projection (metadata / redacted / full), so "context
  accounting" and "raw span bytes" are different, separately-gated disclosure levels, and
  the default is the narrowest;
- **the taint/evidence qualifier** — every returned datum carries OBSERVED vs WITNESSED
  (the [conflation](../explainers/) discipline), and **no sealed/quarantined span bytes
  ever cross the boundary** — the outbound mirror of the inbound steer screen;
- **the structured refusal** — a closed reason when a read is illegal (unknown trace,
  insufficient capability, a projection that would leak a quarantined span), from the same
  closed-vocabulary discipline as every other fak refusal.

The existing read seams register into this vocabulary (as the control verbs registered
into `sessionctl`), unifying the fragmented surface — `fak_context_*`, the `/v1/fak` read
routes, the CDC feeds, `dos_status`, and the off-wire `sessionjournal`/`identitymap`/
`transcript` readers — under one grammar and one front door.

## The spine and the fan-out (the epic)

**Spine (S):** this note + a closed read/query/observe-op vocabulary type (the enum, the
four per-op properties, and the *existing* read seams registered into it) with the
taint-safe-outbound + evidence-qualifier contract generalized across all of them. Unifies
the fragmented read surface under one named plane and one grammar; the read-side twin of
`internal/sessionctl`.

**Children (the gaps, worst-first — each with its own witnessed done-condition):**

1. **Taint-safe outbound + per-principal scoping floor** (closes the auth gap). Every read
   seam gains a capability + principal check and an outbound screen that refuses to emit
   sealed/quarantined span bytes; a cross-principal read of the default trace's dropped
   originating task is refused with a closed reason. Worst-first: today it is unauthenticated
   on loopback. Witness: a red-team test that a non-owning principal cannot `restore` a
   quarantined/sealed span, and that the screen is byte-exact.
2. **Structured session-query route — "query parts of the transcript"** (the goal's literal
   ask). A closed query grammar over a session's turns / tool-calls / decisions —
   `last-n-turns`, `tool-failures`, `files-touched`, `decisions-about <term>`,
   `spans-matching <term>` — served over the wire from a durable projection, taint-filtered,
   *without* returning the whole JSONL. Witness: the route answers each query kind against a
   fixture session, and the "raw bytes" projection is separately gated.
3. **A durable store behind the span/context reads** (closes the ephemerality gap).
   Persist/project `s.ctxRestore` and `s.ctxValue` onto the content-addressed session ledger
   (compose with sessionledger #2392, do not duplicate its storage) so spans and context
   accounting survive a gateway restart and a finished session. Witness: a restart replays
   the spans; a closed session is still queryable.
4. **Expose the identity map + a unified session directory over the wire** (closes the
   addressing + two-directories gaps, C7/#3791). `GET /v1/fak/identity` (UUID↔trace, from
   `resume/identitymap.go`) and a directory that folds `sessionjournal` (lifecycle) with the
   gateway session table (drive-state) into one "which sessions exist and how do I address
   them" view. Witness: an external process resolves a transcript UUID to a live trace and
   back over HTTP, with the OBSERVED qualifier.
5. **Transcript-level event subscribe / re-attach for a peer** (the OBSERVE rung; the A2A
   `resubscribe` / AG-UI `STATE_DELTA` analog). A filtered, taint-safe, cursor-resumable
   stream of turn / tool-terminal / decision events on the same CDC discipline as
   `/v1/fak/changes` — but over transcript events, a *third* feed, not the coherence bus.
   Witness: a peer tails a session, drops, and re-attaches by cursor with no gap and no
   quarantined span.
6. **Session state as MCP resources (expose-as-view for peers).** Project the queryable
   transcript/context/decisions as MCP *resources* (list/read/directory), so any MCP client
   — a peer agent — reads a session the standard way, per-principal scoped. Witness: an MCP
   `resources/list` + `resources/read` returns the taint-filtered projection.
7. **The `read → decide → control` external-supervisor loop** (the complex capstone; folds
   in fleet-scoped query). Wire this read plane into #2753's control plane as a documented,
   regime-gated pattern: an external supervisor process READs (this plane) → CHECKs
   (`dos_verify`/`dos_commit_audit`/`dos_recall`) → CONTROLs (steer/redirect/hold), and the
   query grammar folds across many live sessions ("which sessions are stuck on a confirm
   gate", "which touched file X"). Regime-gate every intervention (from #2533: interventions
   harm high-scoring sessions). Witness: a reference supervisor drives a fixture fleet
   read-only, then intervenes only on a demonstrated stuck session.

## Honest fences

- **Not yet, by construction.** Nothing here is shipped beyond the read seams inventoried
  above. Each child is a named gap with its own witnessed done-condition. The one wire seam
  that returns transcript *content* today (`fak_context_restore`) returns exactly one
  compaction-dropped span; do not overstate it as "read the transcript."
- **The read plane is an exfiltration surface — treat it as one.** A read that leaks a
  quarantined or sealed span is the outbound dual of a prompt injection: the same kernel
  that screens the inbound steer must screen the outbound read. Per-principal scoping and
  the taint-safe-outbound screen (child 1) are the *floor*, shipped before any richer read,
  not a later hardening pass.
- **Read is not control, and reading must not perturb.** Following Temporal's Query
  discipline: a read has no side effects and cannot advance the loop. The capstone
  supervisor's *control* actions go through #2753's typed, witnessed, capability-checked
  verbs — never as a side effect of a read.
- **Evidence, not self-report.** A read returns witnessed facts wherever a witness exists
  (`dos_status`'s no-`claimed`-field posture, generalized). Where only the agent's own
  account exists, it is labeled OBSERVED, never laundered into WITNESSED.
- **Compose, don't duplicate.** Storage is sessionledger #2392; the observe transport is the
  CDC cursor discipline (#3171); control is #2753; identity is `resume/identitymap.go`; the
  lease/presence read plane (#2297) is the precedent pattern. This plane is the *read/query
  API surface and its closed vocabulary*, standing on all of them.

## Related

- **#2753** — out-of-band operator control (the *write* sibling; this note is its mirror).
- **#2388** — the owned turn / admit-once transcript / live structured progress SSE (the
  loop-internal producer whose events the OBSERVE rung would expose externally).
- **#2392** — sessionledger (the content-addressed transcript substrate the durable read
  store projects from).
- **#2397** — agentgraph / principal-tagged messaging (the principal identities a scoped
  read is keyed on).
- **#2297 / #2254** — the lease/presence read plane and the multi-node "planes" vocabulary
  (`GET /v1/leases`, `/v1/sessions`, OBSERVED qualifier) this extends from metadata to
  transcript/context/decisions.
- **#3171 / #3172** — the CDC change feeds + Debezium projection (the observe transport).
- **#2533** — trajectory-control (the regime-gate every capstone intervention must honor).
