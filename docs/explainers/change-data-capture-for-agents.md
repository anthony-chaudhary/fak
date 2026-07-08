---
title: "Change Data Capture for Agents — fak's Change Feeds in CDC Terms"
description: "fak already implements the Change Data Capture pattern family — a hash-chained change log, cursor-drained change feeds, a transactional outbox, and fold-to-read-model projections — over agent, cache, and session state. It is a change SOURCE for agent work, not a database-replication tool."
slug: change-data-capture-for-agents
keywords:
  - change data capture
  - CDC
  - CDC for AI agents
  - change data capture for agents
  - Debezium alternative
  - change feed
  - event log
  - transactional outbox pattern
  - CQRS read model
  - log-based CDC
  - cursor / offset / LSN
  - tombstone / delete event
  - stream agent state changes
  - what changed since offset
  - coherence bus
date: 2026-07-08
---

# Change Data Capture for Agents — fak's Change Feeds in CDC Terms

> **TL;DR:** If you know **Change Data Capture** (CDC) from databases, you already
> know `fak`'s change model. `fak` implements the whole CDC pattern family — an
> append-only **hash-chained change log**, **cursor-drained change feeds**
> (`GET /v1/fak/changes`, `GET /v1/fak/events`), a **transactional outbox**, and
> **fold-to-read-model** projections — but over **agent, cache, and session state**,
> not your business tables. It is a change **source** for agent work; it is **not** a
> Debezium-style database-replication tool, and it never ingests your Postgres WAL.

**Concept served:** *the agent kernel already speaks CDC* — one of the
[popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md) concepts,
here to make `fak` legible to the infra engineers who search for "change data
capture" and "Debezium" and to draw the honest line before anyone assumes the wrong
thing.

## Does fak do change data capture?

Yes — over agent state. CDC's core move is *"don't ask the writer what changed;
observe the authoritative change log."* `fak`'s core move is *"don't trust the
agent's self-report; observe the authoritative artifact (git / journal / lease)."*
Those are the **same discipline** in different domains, so the CDC vocabulary maps
onto `fak` almost one-to-one.

## The map: CDC concept ↔ fak seam

| CDC concept | fak's implementation | status |
|---|---|---|
| **Change log / WAL** (capture point) | `internal/journal/journal.go` — append-only, **hash-chained** JSONL; `Emit(abi.Event)` is the per-event tap, `appendLocked` stamps `Seq`/`TSUnixNano`/`PrevHash`/`Hash` | shipped |
| **Cursor-tailed feed** (LSN / Kafka offset) | `internal/gateway/coherence.go` `CoherenceEvent{Kind, Seq, Tags, Witness, WorldVer, TrustEpoch, principal}`, drained by `?since=` at `GET /v1/fak/changes` (live coherence bus) and `GET /v1/fak/events` (durable journal) | shipped |
| **Tombstone / delete event** | the `revocation` event kind — a refuted world-state witness triggers causal eviction across every consumer | shipped |
| **Change key** | `Tags` (the invalidation scope of a mutation) / `Witness` (the refuted content hash) | shipped |
| **Source metadata block** (`source.lsn`, `source.ts_ms`) | `Seq` + `WorldVer` (consistency clock) + `TrustEpoch` (integrity clock) + `principal` | shipped |
| **Per-topic / per-tenant routing** | principal-scoped `drain(principal, sinceSeq)` — a tenant sees its own mutations plus global broadcasts, never a peer's | shipped |
| **Transactional outbox** | `internal/slackoutbox/` — durable enqueue on an append-only spool, a single serialized drainer, nonce idempotency, at-least-once with dead-lettering | shipped |
| **CQRS read model / projection** | fold/replay: `cmd/fak/audit_replay.go`, `cmd/loophealth/fleet.go`, `internal/cachevalueledger` `FoldTrendGate`, over the shared `internal/jsonlledger` primitive; and the truth verbs `dos_verify` / `dos_status` / `dos_commit_audit` are read-models derived from the log, never from self-report | shipped |
| **Debezium-compatible envelope** (`op`/`before`/`after`/`source`) | an optional export projection over the feeds | proposed ([#3171](https://github.com/anthony-chaudhary/fak/issues/3171)) |
| **Unified work-change stream** (commit + verdict, lease epoch, verdict flips on one cursor) | the outbox insight applied to agent work | proposed ([#3172](https://github.com/anthony-chaudhary/fak/issues/3172)) |

## Is fak like Debezium or Kafka Connect?

No — and the difference is the honest part of the pitch:

- **Source, never a sink.** `fak` emits changes about *agent work*. It does **not**,
  and should not, ingest a user's Postgres WAL / MySQL binlog. If you need to
  replicate your database into a warehouse, use Debezium — that is not `fak`'s job.
- **Agent state, not business tables.** The "rows" `fak` captures are cache
  mutations, revocations, session drive-state revisions, and (proposed) committed
  work — not your `orders` table.
- **Semantic events, not row images.** A `fak` change event is a typed fact (a
  mutation with a scope, a revocation with a witness), not a full before/after row
  image. Git already holds the before/after of code.
- **HTTP long-poll, not Kafka.** The cadence is turn/commit-scoped. There is no
  Kafka/Flink runtime — the transport is the existing `?since=` cursor over plain
  HTTP. The CDC "initial snapshot then stream" bootstrap is already `git log` plus
  the ledger tail.

## How do I consume the feed?

The same way you consume any log-based CDC feed — by **offset**:

1. Hold a cursor (start at `0`). `GET /v1/fak/changes?since=<cursor>` returns every
   event with `Seq > cursor` plus your next cursor.
2. **At-least-once + bounded retention.** The feed is a bounded ring; a consumer
   that falls behind the retained window sees a `Seq` gap and **re-syncs to head**.
3. **Idempotency.** Dedupe by `Rev` (session feed) or `Witness` (revocation) so a
   replayed event is safe to apply twice.
4. **Visibility.** A tenant drains only its own mutations plus global broadcasts.

The consumer contract and a reference consumer are being written up in
[#3173](https://github.com/anthony-chaudhary/fak/issues/3173).

## Why does an agent kernel need CDC at all?

Because a fleet of agents sharing one gateway needs to learn *precisely what another
agent changed or refuted* — so each can re-plan and evict its own private cache
instead of getting a blunt "something, somewhere, changed." That cross-agent
coherence signal is exactly a change feed: typed write **mutations** and integrity
**revocations**, ordered by a shared cursor. The same log then powers the
read-models that answer "did this actually ship?" from evidence rather than from an
agent's narration — the [verify-don't-trust](verify-dont-trust.md) discipline, which
is CDC's "observe the log, don't ask the writer" applied to agent work.

## See also

- [Verify, don't trust: what DOS actually checks](verify-dont-trust.md) — the
  read-model side: a done-claim graded against git, not against the worker's word.
- [vDSO revoke as a comm-style revocation](vdso-revoke-as-comm-revoke.md) — the
  tombstone/eviction mechanism behind a `revocation` event.
- [Status a peer can trust](status-a-peer-can-trust.md) — a fail-closed run digest
  folded from the log with no `claimed` field, by construction.
- [What fak is not — the honest boundary](what-fak-is-not.md) — the companion fence;
  "not a Debezium-style DB-replication CDC" belongs on the same map.
- [The fak/DOS glossary](glossary.md) — the newcomer vocabulary, one line each.
