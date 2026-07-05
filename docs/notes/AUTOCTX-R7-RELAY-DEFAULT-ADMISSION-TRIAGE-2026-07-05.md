---
title: "Triage: R7 relay-by-default admission for headless goal sessions (#2205)"
description: "Generation-horizon classification and mechanism-vs-witness gap for the autoctx R7 rung — admission-level relay-by-default for headless goal sessions — with the stale blocker (#1860 now closed) recorded as an invalidating assumption. Triage only; #2205 stays OPEN."
date: 2026-07-05
---

# Triage: R7 relay-by-default admission for headless goal sessions (#2205)

Status: **triage / classification only — #2205 stays OPEN.** This note records
the generation-horizon decision, the mechanism-vs-witness gap, and one stale
blocker premise, so the next dispatcher (Claude or opencode) starts warm
instead of re-deriving the map. It does **not** implement R7: admission-wiring
plus a dispatch dogfood witness are net-new product work above this dispatch's
`triage only` risk allowance, and the dogfood witness needs a live headless
worker rotating ≥2 legs — a host capability this triage pass cannot honestly
produce or fake.

Date: 2026-07-05. Lane routed: `docs` (the issue's file-tree). Labels on the
live issue at triage time: **none**. Parent epic: [#2198](https://github.com/anthony-chaudhary/fak/issues/2198)
(spine: [`CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md`](CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md)).

## The ask (verbatim invariant)

> Admission-level default: a headless session with a goal gets a relay policy
> with no operator opt-in (interactive sessions stay opt-in). Includes the
> refusal path: a session that cannot relay (no durable store, no goal) is
> admitted as legacy with a structured reason, never silently.

Done condition: a headless dispatch worker crossing its leg ceiling rotates
without any operator action or per-session flag; the decision lands in the
ledger. Witness: one worker rotates ≥2 legs with flat peak context and the
`RELAY_*` reason rows present; no dispatch config names the relay.

## Generation horizon: `gen/next`

Classified from issue evidence, not guessed (the dispatch frame's proof bar):

- The parent epic **#2198** declares itself `Generation: next` / `Area:
  managed-context` / `Priority: P1` with `Labels-intended: managed-context,
  epic, gen/next`. R7 is a child rung (R1–R8) of that `gen/next` foundation
  epic, filed as `feat(autoctx)`.
- The blocking mechanism epic **#1860** carried `gen/next` (+ `agentic-serving`,
  `priority/P1`). R7 inherits that horizon.
- The `gen/next` definition in [`docs/generation.md`](../generation.md) — "near-term
  foundation that should be runnable by agents soon, but still needs a gate,
  dogfood run, schema, or default-exposure proof" — is an exact fit: R7's Done
  condition **is** a default-exposure flip plus a dogfood witness.
- It is **not** `gen/now`: the current product path does not yet admit a
  headless goal session onto a relay policy by default, and the witness the
  issue names (a live worker rotating ≥2 legs with `RELAY_*` rows) does not
  exist yet.

## Mechanism-vs-witness gap (why it stays OPEN)

The relay **mechanism** is real, but the **admission-default** rung R7 owns is
unbuilt:

- Mechanism present: `cmd/fak/chatrelay.go`, `cmd/fak/relay_handoff_rotate_close_test.go`,
  `cmd/fak/relay_resume_test.go`, `internal/chatrelay/`. The reason vocabulary
  is specified in [`RELAY-REASON-VOCABULARY-2026-07-01.md`](RELAY-REASON-VOCABULARY-2026-07-01.md),
  with the baton in [`RELAY-BATON-SCHEMA-2026-07-01.md`](RELAY-BATON-SCHEMA-2026-07-01.md)
  and the operator path in [`RELAY-OPERATOR-RUNBOOK-2026-07-04.md`](RELAY-OPERATOR-RUNBOOK-2026-07-04.md).
- Admission-default absent: a source search for an admission-time relay-by-default
  seam (`RELAY_*ADMIT`, `relayByDefault`, `admissionRelay`) finds nothing in the
  Go tree. The reason-vocabulary note states plainly: **"No gate consumes these
  tokens yet."** So the default policy, the legacy-with-structured-reason refusal
  path, and the ledger rows R7 requires are all still to build.

R7 is therefore correctly `P2` on *build readiness* — it is admission-wiring +
a dogfood witness away from done — even though its named blocker is now retired
(next section).

## Invalidating assumption (the stale blocker)

The issue body asserts, present-tense, **"Blocked by: #1860 … #1860 builds the
relay mechanism with opt-in semantics."** That premise is stale: **#1860 is
CLOSED** ("Epic: Perpetual sessions — bounded-context relays instead of
compaction"). The mechanism epic is done; the admission-default rung was
deliberately deferred to R7 (per #2198's "relay mechanics live in #1860 … this
epic owns only the binding property, the joins, and the zero-knob defaults"),
not left unbuilt inside #1860.

Consequence for dispatch: R7 is no longer *blocked by an in-flight mechanism
epic*. It is *unblocked-but-unbuilt* — the next honest state is a ready
`feat(autoctx)` implementation ticket (admission seam + refusal reason + ledger
row + dogfood), not a P2-blocked-on-#1860 hold. The "P2 (blocked)" label in the
body should be read as build-readiness, not a live dependency wait.

## Triage evidence (generation contract)

- **Promotion evidence** (what would move R7 toward `now`): the admission seam
  lands with a test proving a headless goal session is admitted onto a relay
  policy with no flag, an interactive session stays opt-in, and a no-durable-store
  or no-goal session is admitted legacy with a structured `RELAY_*` refusal
  reason (never silent); then a dispatch dogfood shows one worker rotating ≥2
  legs with flat peak context and `RELAY_*` rows in the ledger.
- **Demotion / retirement evidence** (what would push it back or close it): the
  zero-knob doctrine in #2198 is retired or the relay mechanism is superseded by
  a different bounded-context strategy, making admission-default relay the wrong
  default; or a witness shows headless goal sessions already never overflow, so
  no admission default is needed.
- **Invalidating assumption** (named, per the frame): "R7 is blocked by the
  in-flight #1860." **Invalidated** — #1860 is closed; R7 is unblocked-but-unbuilt.

## Recommended intake repair (label-only, no scope change)

To end the classification drift on the live issue without touching build state:

- Add `managed-context`, `generation`, `gen/next` (the epic's own
  `Labels-intended`; all three labels exist in the repo).
- Set milestone `Generation G1 - Next Gen` to agree with `gen/next` (per the
  generation contract's stream↔milestone agreement rule).
- Leave state OPEN; the "Blocked by #1860" line is stale but is body prose, not
  a label — a future editor should reword it to "unblocked-but-unbuilt after
  #1860 (closed)."

These are recommendations recorded for an operator/peer with issue-write scope;
this triage pass commits the classification note only.
