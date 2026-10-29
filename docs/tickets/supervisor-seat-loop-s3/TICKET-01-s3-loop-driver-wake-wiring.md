# feat(dispatch): supervisor-seat loop driver + wake wiring behind the envelope - S3 remainder of #13464

<!-- fak-supervisor-agent-key: supervisor-agent/seat-loop-driver -->

GitHub issue: [#14092](https://github.com/anthony-chaudhary/fak/issues/14092).

routing
lane: cmd
paths: ["internal/supervisoragent/loop.go", "internal/supervisoragent/loop_test.go", "cmd/fak/dispatch_tick_preflight.go", "docs/tickets/supervisor-seat-loop-s3/TICKET-01-s3-loop-driver-wake-wiring.md"]
expected_steps: 4

**Dispatch — operator** — S1+S2 of the seat loop shipped (`55339ecd3a`, published to origin/main 2026-10-28): the addressed payload-free `DecisionRecord` journal and the pure deterministic-first `Decide` seam. What remains of #13464's definition of done is unbuilt: no loop driver cycles the shipped contracts, no wake wiring exists in `cmd/fak` dispatch, and the epic's keep-bit soak still has no sample producer.

## Process cause
Process cause: none

## Value
Centrality: Enabling (the bounded judgment tier that keeps dispatch moving without a human hand-adjudicating every tie; also the decision-record source the seat loop exists to produce).

P1 Context: advanced — consumes only the shipped `SupervisorInput` / `SupervisorAction` / `Envelope` / `Journal` / `Decide` surfaces; no new authority.
P2 Net value: advanced — turns the shipped kernel into one bounded decision cycle and gives the epic's `warn`→`enforce` soak its producer.
P3 Adaptation: preserved — typed witnesses only; no transcript reading, no new spawn path, no second scheduler.
P4 Operations: preserved — dry-run/advisory land before any live act; fail-closed on a missing witness.

## Parent context
Child of #13464; this leaf covers its remaining unchecked definition-of-done items (loop driver, withheld-witness escalation at loop level, keep-bit sample producer). Grandchild of the supervisor-seat epic fak-private#2040. S1+S2 landed as `55339ecd3a` ("feat(supervisoragent): add addressed decision journal + pure Decide seam for the seat loop", Ref #13464). Design note: `docs/notes/CONCEPT-SUPERVISOR-AGENT-SEAT-2026-07-13.md` (bounded seat).

## Current state
`internal/supervisoragent` ships `input.go`+`assemble.go` (`SupervisorInput` + `Assemble`, closed payload-free witness projection), `action.go` (`SupervisorAction` + `Lower`, six-verb union over existing admission calls), `envelope.go` (`Envelope`/`Authorize`/`LowerInEnvelope`/`KeepBitVerdict`, ratchet-gated, fail-closed), `journal.go` (`DecisionRecord` + `Journal` append/replay/last, closed-field golden pin, input digest never contents), and `decide.go` (pure `Decide`, closed reason vocabulary, absent-witness→escalate on every path, envelope-gated proposals, green fleet holds). Nothing imports them in a cycle: there is no loop driver, no wake wiring in `cmd/fak` dispatch, and no keep-bit sample producer. #13464's DoD boxes 3-5 remain open.

## Why now
S1+S2 settled green on trunk, so the remaining scope is self-contained and bounded. The loop is #13464's last in-scope artifact: without it the epic's keep-bit soak has no sample producer, and the addressed decision records the seat loop was built to emit do not exist yet.

## Working spine
The parent's five named stages, minus the two already shipped:
1. **loop driver** — assemble at wake via `Assemble`; decide via the shipped pure `Decide`; lower ONLY through `LowerInEnvelope` so every executed action leaves its witnessed artifact; append exactly one addressed `DecisionRecord` per wake.
2. **dry-run/advisory mode** — proves zero admission calls end to end.
3. **wake wiring** — interrupt-driven, behind the envelope, in `cmd/fak` dispatch (`dispatch_tick_preflight.go`); a green fleet wakes nothing.
4. **keep-bit sample producer** — emits one sample per wake for the epic's `warn`→`enforce` soak.

## Core through-line
The loop driver + wake wiring + dry-run proof + keep-bit sample producer, all lowering to the shipped seams only.

## Gold-plating boundary
Transcript reading or health recognition (health stays a deterministic rung). New spawn paths or lane vocabulary (`dos_arbitrate` + dispatch admit only). Envelope earning formula (#4480 owns it) and counter definition. New escalation transport. Changing `SupervisorInput`'s field set or `SupervisorAction`'s verb set. Additionally: no polling or pacing of a healthy fleet (interrupt-driven wakes only), and no enforce-mode actuation in this ticket — advisory/dry-run lands first; the `warn`→`enforce` soak stays epic-gated.

## Definition of done
- [ ] Loop driver assembles at wake, lowers only through `LowerInEnvelope`, and appends exactly one record per wake; dry-run proves zero admission calls.
- [ ] Withheld witness forces escalation on every path at loop level (fail-closed), not only inside `Decide`.
- [ ] Keep-bit sample producer emits one sample per wake for the epic's `warn`→`enforce` soak.
- [ ] Loop-level fake-witness end-to-end test in the driver package.

## Witness
`go test ./internal/supervisoragent/... -count=1` green, including: `TestSupervisorSeatLoop` (fake-witness end-to-end: wake → assemble → decide → lower → journal, one record per wake), a dry-run test proving zero admission calls, a withheld-witness table proving escalation on every loop path, and a journal replay test. Plus `make ci` green.

## Acceptance gate
Done condition + `make ci` green; the loop witnesses merge before any live wiring widens past advisory.

## Work unit
One worker owns the loop driver + wake wiring + tests.

## Expected steps
4 (loop driver dry-run → loop-level fail-closed withheld-witness paths → wake wiring behind the envelope → keep-bit sample producer).

## Assumptions
- The shipped `Decide` ladder plus the typed witnesses cover the middle-tier cases without a transcript. (Falsifiable: a decision class that provably needs transcript content belongs in the human residual, not this seat.)
- The existing admission verbs cover every action the vocabulary needs; anything else escalates.

## Confusion risks
- This is NOT a "judge model": it never manufactures a health signal. If a change starts reading transcripts to infer health, it has crossed into the forbidden recognizer — stop.
- The loop must not become a second scheduler: it decides at wake time through existing admission verbs; it does not poll, pace, or own calendars.
- A green *absence* is not a green *witness*: missing witness → escalate, never act.

## Coordination
`internal/supervisoragent`, `cmd/fak` dispatch, and `internal/gateway` are contended lanes — arbitrate before writing; keep the diff to the named paths.

## Lane
cmd

## Likely files
- `internal/supervisoragent/loop.go` (+ `loop_test.go`)
- `cmd/fak/dispatch_tick_preflight.go` (wake wiring, behind the envelope)
- `internal/gateway/` (fleetmon/escalation witness heads, only if the input sources need them)

## Closure binding
Merged commits whose bodies carry `supervisor-agent/seat-loop-driver` close this leaf. Landing satisfies #13464's remaining DoD items; the parent closes on its own binding (live dry-run receipt on the epic plus the soak verdict).

## Trigger
S1+S2 settled (`55339ecd3a`); #13464's remaining DoD items (loop driver, loop-level fail-closed, keep-bit sample producer).

## Work estimate
Estimate: 3 points. Uncertainty: derived from the parent's 5-point estimate minus the shipped S1+S2 half.

## Overall completion contribution
Contribution: 3/3 points of the S3 leaf (the remainder of #13464).

## Completion standard
production

## Target operating envelope
- test packages: >= 1 package

## Witnessed operating envelope
- test packages: 1 package
