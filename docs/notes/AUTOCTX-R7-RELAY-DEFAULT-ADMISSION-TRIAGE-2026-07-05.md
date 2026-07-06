---
title: "Triage: R7 relay-by-default admission for headless goal sessions (#2205)"
description: "Generation-horizon classification and mechanism-vs-witness gap for the autoctx R7 rung — admission-level relay-by-default for headless goal sessions. Corrected 2026-07-05: the relay lives in internal/relay (tracks A-F only); tracks G (policy) + H (driver) of #1860 are still OPEN, so R7 is genuinely blocked on the unbuilt rotation driver, not unblocked-but-unbuilt. Triage only; #2205 stays OPEN."
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

The perpetual-session relay **mechanism is only partly built**, and the part R7
sits directly on top of — the rotation driver and policy — does **not** exist
yet. (Correction, 2026-07-05: an earlier revision of this note attributed the
mechanism to `cmd/fak/chatrelay.go` / `internal/chatrelay/`. That is wrong —
`internal/chatrelay` is the Slack↔chat-endpoint bridge, unrelated to the relay.
The perpetual-session relay lives in `internal/relay/`.)

- Present (tracks A–F of #1860): `internal/relay/` holds the baton, codec,
  safepoint, externalize gate, reload, and stale-baton pieces. The only **wired**
  relay reason constants are `RELAY_NOT_EXTERNALIZED` (`externalize_gate.go`),
  `RELAY_BATON_STALE` (`stale.go`), and the advisory `RELAY_ARMED`
  (`internal/session/observe.go`, explicitly behavior-free — "does not change
  `State.Run`"). The vocabulary is specified in
  [`RELAY-REASON-VOCABULARY-2026-07-01.md`](RELAY-REASON-VOCABULARY-2026-07-01.md),
  the baton in [`RELAY-BATON-SCHEMA-2026-07-01.md`](RELAY-BATON-SCHEMA-2026-07-01.md),
  the operator path in [`RELAY-OPERATOR-RUNBOOK-2026-07-04.md`](RELAY-OPERATOR-RUNBOOK-2026-07-04.md).
- **Absent (tracks G + H of #1860):** there is no rotation policy and no relay
  driver. The `[relay]` policy table (#1888), the two-phase arm/fire state machine
  (#1889), Envelope-axis arm triggers (#1890), `RELAY_NO_PROGRESS` (#1893), the
  driver loop (#1894), lease continuity (#1895), done-check + `RELAY_GOAL_DONE`
  (#1896), the hard-ceiling park path (#1898), and the `Recontinue` wiring (#1899)
  are **all still OPEN**. `RELAY_ROTATED`, `RELAY_GOAL_DONE`, `RELAY_NO_PROGRESS`,
  and `RELAY_PARKED_UNSAFE` appear only in test strings and comments — never as
  wired constants. Nothing rotates.
- Admission-default absent: a source search for an admission-time relay-by-default
  seam (`RELAY_*ADMIT`, `relayByDefault`, `admissionRelay`) finds nothing in the
  Go tree, as expected — there is no rotation driver for such a seam to arm.

R7 is therefore genuinely **blocked**, not merely build-ready: its admission
default is a policy flip on top of a rotation driver (H) and policy (G) that are
unbuilt, so its witness (a live worker rotating ≥2 legs with `RELAY_*` rows) is
unreachable today and cannot be honestly claimed done. The "P2 (blocked)" in the
body is a live dependency on #1860's G/H children, corrected below.

## Invalidating assumption (corrected 2026-07-05)

The tempting shortcut is: **"#1860 is CLOSED, therefore its mechanism shipped,
therefore R7 is unblocked."** That inference is false and an earlier revision of
this note made it. The epic *umbrella* #1860 was closed (resolving commit
`e0d85e0`, `dos commit-audit` OK), but its **G (rotation policy) and H (relay
driver) children were not shipped** — #1888, #1889, #1890, #1893, #1894, #1895,
#1896, #1898, #1899 are all still OPEN, and the tree confirms `internal/relay`
holds tracks A–F only with no rotation driver wired.

Consequence for dispatch: R7 **is** still blocked — not on the in-flight epic
umbrella (that is closed), but on the specific unbuilt G/H driver/policy children
of #1860. The next honest state is **not** a ready implementation ticket: R7
cannot be built or witnessed until at least the driver loop (#1894), the `[relay]`
policy table (#1888), and the arm/fire state machine (#1889) land so a headless
leg can actually rotate. Only then does R7 reduce to a small admission-default
flip + refusal reason + dogfood. The "P2 (blocked)" in the body is accurate.

## Triage evidence (generation contract)

- **Promotion evidence** (what would move R7 toward `now`): FIRST the G/H
  prerequisites land — minimally the driver loop (#1894), the `[relay]` policy
  table (#1888), and the arm/fire state machine (#1889) — so a headless leg can
  rotate and `RELAY_ROTATED` becomes a wired constant, not a test string. THEN
  the admission seam lands with a test proving a headless goal session is admitted
  onto a relay policy with no flag, an interactive session stays opt-in, and a
  no-durable-store or no-goal session is admitted legacy with a structured
  `RELAY_*` refusal reason (never silent); then a dispatch dogfood shows one
  worker rotating ≥2 legs with flat peak context and `RELAY_*` rows in the ledger.
- **Demotion / retirement evidence** (what would push it back or close it): the
  zero-knob doctrine in #2198 is retired or the relay mechanism is superseded by
  a different bounded-context strategy, making admission-default relay the wrong
  default; or a witness shows headless goal sessions already never overflow, so
  no admission default is needed.
- **Invalidating assumption** (named, per the frame): "a CLOSED epic umbrella
  (#1860) means its mechanism shipped, so R7 is unblocked-but-unbuilt." **False** —
  checking the G/H children shows #1888/#1889/#1890/#1893/#1894/#1895/#1896/#1898/#1899
  still OPEN and no rotation driver in `internal/relay`. R7 stays blocked on the
  unbuilt driver/policy. (A second latent assumption: that admission must route
  through a standalone driver at all — if a later design admits the relay at the
  `internal/sessionreset` seam, the blocker set narrows; re-check before dispatch.)

## Recommended intake repair (label-only, no scope change)

To end the classification drift on the live issue without touching build state:

- Add `managed-context`, `generation`, `gen/next` (the epic's own
  `Labels-intended`; all three labels exist in the repo). *(Applied — the issue
  carries all three as of 2026-07-05.)*
- Set milestone `Generation G1 - Next Gen` to agree with `gen/next` (per the
  generation contract's stream↔milestone agreement rule).
- Keep state OPEN; the "Blocked by: #1860" body line should be sharpened, not
  removed — the live dependency is the OPEN G/H children of the closed umbrella
  (#1888, #1889, #1894 at minimum), so reword it to "Blocked by: #1860's
  rotation-driver/policy children (#1888/#1889/#1894 …)".

## Mis-close correction (2026-07-05)

The dispatch loop's close-resolved arm closed #2205 citing the *first revision
of this very note* (`687cf4d`, a docs-only triage commit whose subject carried
`(#2205)`) as the resolving commit. A triage note cannot resolve an
implementation rung: the R7 witness (a headless worker rotating ≥2 legs with
wired `RELAY_*` ledger rows) does not exist, and the G/H prerequisites are still
OPEN. #2205 was reopened accordingly, per the close comment's own "Reopen if
this does not fully resolve it" escape hatch. Separate defect for the loop: a
`docs(...)`-typed / triage-verbed commit referencing `(#N)` must not arm
close-resolved for #N.

These are recommendations recorded for an operator/peer with issue-write scope;
this triage pass commits the classification note only.
