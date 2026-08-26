---
title: "Blast-radius containment W7 - surface - #2719"
description: "Ticket body for issue #2719 of the blast-radius containment cohort (#2712-#2720): the operator blast card: one cause, N affected, one fixing (`internal/blockerpost`)"
---

# W7 — surface — #2719

One ticket of the [blast-radius containment cohort](../blast-radius-containment-cohort.md) (#2712-#2720). The cohort page carries the problem statement, the ship order, the cohort map, and the `gh api` filing recipe; this page is the single ticket, so it can be filed as one issue body without editing anything out.


- **title:** feat(knownbad): operator blast card — 1 root cause → N affected, 1 fixing, N-1 parked (W7, epic #2712)
- **labels:** dispatch, dev-ex, operator, observability, priority/P2
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713, #2715
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2719

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713 and the blast-radius estimator W3. Extends `internal/blockerpost` and the dispatch scorecard/TUI.

## Current state

`internal/blockerpost` posts a single blocker to Slack with severity tiers (status/operator/clear); the dispatch TUI and scorecard show per-issue rows. Neither shows the **blast frame** — the one thing an operator needs to see at a glance: "1 root cause -> N affected, 1 fixing, N-1 parked, witness pending." Today a shared bug taxing the fleet is invisible as a *shared* event; it reads as N unrelated stuck workers.

## Why this is next

This is the **surface** step. Recognition (W2), estimation (W3), and holding (W4) can all be working while the operator still can't tell one shared bug from nine coincidences. One blast-framed card turns the whole containment state into a single legible line.

## Working spine

One card per **live** known-bad signature, fed from the ledger (#2713) + the blast estimate (W3):

- Fields: root-cause/signature, affected count, the elected fixer (W5), parked count, and witness status (pending/resolved).
- Severity: `status` (muted) while a fixer is claimed and progressing; escalate to `operator` (surfaced mention) when a signature has NO fixer after a threshold, or the witness is overdue.
- Empty ledger -> a quiet all-clear line (the existing `clear` tier), so a scheduled run honestly says "no shared blockers."

Rendered by a `fak knownbad report` verb and wired into the `blockerpost` feeder + a dispatch scorecard/TUI row.

## In scope

- `cmd/fak/knownbad.go` (or `cmd/fak/blockers.go`): a `report` that renders the per-signature blast card.
- `internal/blockerpost`: a fold from the live known-bad set to a `Blocker` (reuse `Severity`/`Blocker`/`Blocks`).
- A dispatch scorecard/TUI row for live signatures.
- Tests over a fixture (1 fixer + 5 affected -> one card; no-fixer -> operator severity; empty -> clear).

## Out of scope

Deciding the hold (W4), electing the fixer (W5), resolving (W6). This only *renders* the state the other children produce.

## Done condition

Given a live known-bad with 1 fixer + 5 affected, `fak knownbad report` renders one card "root cause X -> 6 affected, 1 fixing (@who), 5 parked, witness: pending"; a signature with no fixer past the threshold renders at `operator` severity; an empty ledger renders the quiet all-clear.

## Witness

`go test ./internal/blockerpost/... ./cmd/fak/...` green for the three render cases; a captured `fak knownbad report` (dry-run) card for each. Cite in the commit body.

## Acceptance gate

`make ci` green; the three render cases green; the cards captured (dry-run, posts nothing).

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak knownbad)` (or `(fak blockers)`) landing the report renderer + the blockerpost fold + the scorecard row + tests.

## Dependencies

- after: #2713 (ledger), W3 (affected set)
- related: #2712, W5 (fixer identity shown on the card)

## Likely files

- `internal/blockerpost/blockerpost.go`
- `internal/blockerpost/render.go`
- `cmd/fak/knownbad.go`
- `cmd/fak/dispatch_scorecard.go`

## Lane

dispatch

## Work unit

leaf

## Expected steps

6

## Assumptions

- The `blockerpost` severity tiers (status/operator/clear) map cleanly onto the fixer-present / no-fixer / empty cases, so no new transport is needed.
- The affected/fixer/parked counts come straight from W3 + the ledger; the card is pure rendering.

## Confusion risks

- This is a READ/render surface — it must not decide holds or elect fixers; it reflects the ledger + estimate.
- Default to dry-run: a scheduled render must never page on a `status`-tier card, only on `operator`.

## Coordination

Touches `internal/blockerpost` + a dispatch scorecard file (contended) — narrow pathspec; sequence after W3 + W5 so the fields exist.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per operator-surface seam; deduped by the `fak knownbad report` verb.
