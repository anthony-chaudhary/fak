

---

<!-- dispatch-contract sections appended 2026-07-05; derived from the issue prose above (Current state / Scope / Done condition / Witness already present in the body) + verified repo state (autoctx spine note, epic #2198, readout issue #1918). No intent change. Only the sections the contract review reported missing are added here; the existing body sections are NOT restated. The fleet worker's capability floor refuses outward `gh issue edit`/`gh issue comment`, so this backfill lands as a local contract overlay for the dispatcher's next tick; an operator applies it to the real body later. -->

## Parent context

Epic #2198 — the zero-knob automatic-context epic (spine: `docs/notes/CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md`), rung R8 of R1–R8. Extends the still-open readout issue #1918 ("ship the operator readout verb — one-screen continuous-usage contract from witnessed records"). Depends on R1 (the manual-overlay counter) and R2 (the residency+warmth one-ledger join). This rung is the readout EXTENSION that renders the doctrine sentence; it does not build the counter or the join.

## Why now

#1918's readout can describe context state but cannot yet PROVE the epic's doctrine sentence ("nobody managed context this period and nothing was silently lost") because the counter (R1) and the join (R2) it needs are separate rungs. R8 is the rung that folds those witnessed inputs into one screen — it is the epic's falsifiability payoff: the property is only demonstrable once a single screen re-derives it from records. It becomes dispatchable as soon as R1 and R2 land, and it is what turns the whole autoctx epic from doctrine into an observable, refusable readout.

## Working spine

Extend the #1918 operator readout with three witnessed roll-ups: (1) the manual-overlay counter and its trend (from R1's counter), (2) the joined residency+warmth ledger roll-up, provenance-separated (from R2's one-ledger join), and (3) the abstentions — where the kernel declined to manage automatically, each with its structured reason. Every number re-derives from journal/metrics/ledger rows; no self-report fields. The end-to-end path that becomes more true: one screen renders "nobody managed context this period and nothing was silently lost" (or its honest negation) from witnessed records only, and deleting a ledger row makes the screen refuse green with a structured reason.

## In scope

- Extend the #1918 readout to render: the manual-overlay counter + its trend; the joined residency+warmth ledger roll-up (provenance-separated); and the abstentions (kernel-declined-to-manage rows, with structured reasons).
- Wire every rendered number to re-derive from existing journal/metrics/ledger rows — no new self-report fields.
- The C9-style re-derivation check on the rendered numbers, and the refuse-green-on-missing-row behavior.

## Out of scope

- Building the R1 overlay counter or the R2 residency+warmth join — those are separate rungs this one CONSUMES. If either is absent, this rung is blocked on it, not a place to build it.
- The relay-by-default admission change (R7 / #2205) — a sibling rung, not part of this readout.
- Any new operator knob or self-report field; the readout is witnessed-records-only by construction (a knob would be the epic's defect).
- Re-speccing the #1918 base readout verb; this extends it, it does not replace it.

## Acceptance gate

`go test ./...` green for the touched readout/ledger packages (under WSL `./test.ps1` on this native-Windows host, per repo test policy) and `make ci` green; plus the witness below — the C9-style re-derivation check passing on the rendered numbers, and a test that deleting a ledger row makes the screen refuse green with a structured reason.

## Closure binding

Resolving commit cites #2206 in the subject and carries the ship trailer for its lane (`(fak <leaf>)` convention — e.g. `(fak cmd)` or `(fak gateway)` per the issue's named lane), per the repo's Conventional-Commits + `(fak <leaf>)` rule.

## Work unit

leaf

## Expected steps

6 — locate the #1918 readout render path; add the overlay-counter + trend roll-up sourced from R1's counter rows; add the joined residency+warmth roll-up (provenance-separated) from R2's ledger; add the abstentions section from structured-reason rows; add the C9-style re-derivation check + the delete-a-row-refuses-green test; run the focused gate + `make ci` and commit citing #2206.

## Assumptions

- R1 (the manual-overlay counter) and R2 (the residency+warmth one-ledger join) have landed and expose queryable rows; this rung reads them. If either is not yet present, this issue is blocked on that dependency, not a place to synthesize the missing rows.
- The #1918 readout verb exists (open, but with a render surface to extend) or lands first; R8 extends it rather than authoring a new verb.
- The abstention rows carry structured reasons in the existing closed-vocabulary shape, so the readout can render them without inventing reason text.

## Confusion risks

- This rung EXTENDS an existing readout; it does not create the counter, the join, or a second readout verb. Do not re-implement R1/R2 here.
- "Witnessed records only" is load-bearing: every rendered number must re-derive from journal/metrics/ledger rows. A number that comes from a self-report field is a defect, not a feature — the delete-a-row-refuses-green test exists to catch exactly that.
- The provenance separation (WITNESSED vs OBSERVED) in the residency+warmth roll-up must be preserved; do not merge the two provenance classes into one number.

## Coordination

- Touches the readout/ledger surface shared with #1918 (the base readout verb) and the R1/R2 rungs under epic #2198 — coordinate so the extension does not collide with in-flight base-readout or counter/join edits; take the covering lane under one lease.
- The structured-reason vocabulary for abstentions is shared with the rest of the kernel's refusal paths; reuse it, do not fork.

## Trigger

Picked up once R1 (counter) and R2 (join) are witnessed present and the #1918 readout has a render surface to extend — R8 is the fold-into-one-screen rung that becomes dispatchable when its inputs land.

## Batch policy

Standalone leaf. Sibling to #2205 (R7 admission default) under epic #2198 but independently dispatchable; do not batch — R8 depends on R1/R2, R7 depends on #1860.

## Likely files

- `cmd/fak/` (the #1918 operator readout verb's command surface — the render path this rung extends).
- `internal/gateway/` (usage/readout roll-up plumbing, per the issue's named `cmd` / `gateway` lane).
- `internal/journal/journal.go` (the journal/ledger rows every rendered number re-derives from, including the abstention structured-reason rows).

## Dispatch witness (2026-07-05): BLOCKED — the two inputs R8 extends do not exist at HEAD

<!-- Recorded 2026-07-05 by the fleet worker routed to #2206 (route: lane=docs).
This CORRECTS the "Assumptions" section above, which was written as a
contract-completeness backfill and assumed "R1 and R2 have landed." Verified
against `main`, that assumption is false for R2 and for the #1918 render
surface. No code change was made; this is a blocked-report, not a partial ship.
The worker's floor refuses outward `gh issue edit`/`gh issue comment`; an
operator or labeled-token pass can copy this onto the issue and add the labels
below. -->

R8's entire scope is to EXTEND the #1918 operator readout with the R1 counter
and the R2 one-ledger join. Two of those three inputs are absent at HEAD, so
there is nothing to extend and the "witnessed-records-only" numbers R8 must
render have no rows to re-derive from:

- **The #1918 readout verb does not exist.** No `fak context readout` (or
  equivalent) verb renders the seven-line continuous-usage screen. Grep for the
  readout's own field names (`resident_view`, `page_faults`, the
  objective/budget/assumptions/reset/memory/cache lines) across `cmd/fak/*.go`
  and `internal/devindex/verbs.go` returns no render surface; `git log
  --grep=#1918` shows the issue spec but no shipping verb. #1918 is still OPEN.
  R8 extends a render path that has not been authored — the epic's own honesty
  fence says so: "The bulk-closed managed-context children are design-shipped
  **until #1918 witnesses the product contract**."
- **R2 (#2200), the one-ledger join, has not landed.** `git log --all
  --grep="#2200"` / `--grep="one-ledger join"` returns zero autoctx commits
  (the only `R2 ` hits are an unrelated `supportmaturity` "R2 optimize" cell
  and a KV-admission gate). No live per-turn record joins residency actions and
  warmth deltas; R2 is still OPEN. Without it the "joined residency+warmth
  roll-up (provenance-separated)" R8 must render has no source rows, and the
  provenance separation (WITNESSED vs OBSERVED) the readout must preserve has
  no joined record to read.
- **R1 (#2199), the manual-overlay counter, HAS landed** as `internal/ctxknobs`
  + `fak index ctxknobs` (commit `f3af9901c`, `feat(ctxknobs): add
  manual-overlay counter + user-required ratchet (#2199)`), issue CLOSED. This
  is the one input that is real; it is not sufficient on its own to extend a
  readout that does not exist.

So R8 is blocked on **#1918** (the base readout render surface) and **#2200**
(R2, the join). Building it now would mean either authoring the #1918 verb
under a #2206-bound commit (that mis-binds the closure audit to R8) or
rendering a "joined" roll-up with no joined ledger behind it — an always-empty
or synthesized number whose "delete-a-row-refuses-green" witness is vacuously
green, which is exactly the fabricated pass the doctrine's honesty fence
forbids ("WITNESSED and OBSERVED are never summed; the vanity-metric failure
mode is the canonical warning").

### Misroute (same wall as sibling #2205 / #2272)

The issue header names **Lane: `cmd` / `gateway`** (a Go verb + gateway
roll-up), but this dispatch routed it to the **`docs` lane** whose required
`(fak docs)` trailer cannot bind a `cmd`/`gateway` fix. A genuine R8 fix
touches `cmd/fak/` + `internal/gateway/` + `internal/journal/`, outside the
docs file-tree; the closure auditor would refuse the very commit that resolved
it. Correct the route to `cmd` before re-dispatch — otherwise every worker
burns the tick on the trailer/lane impossibility before hitting the dependency
wall.

### Corrected trigger

R8 becomes dispatchable only after BOTH: (1) #1918 ships the base operator
readout verb with a render surface to extend, and (2) R2 (#2200) lands the live
one-ledger residency+warmth join with queryable rows. Until then this P2 sits
unbuildable while looking dispatchable. The "Assumptions" section above should
be read as *preconditions not yet met*, not as satisfied facts.

### Generation stream

gen/now (inherits the autoctx program horizon per the spine note's
classification: R1 `#2199` closed carrying `managed-context`/`gen/now`, so
R1–R8 inherit `gen/now` unless demoted). This rung's *dispatchability* is
gated on #1918 + #2200, which is orthogonal to the horizon. Labels the token
cannot apply but an operator should: `managed-context`; and a
`blocked-by:#1918`, `blocked-by:#2200` dependency marker so the loop stops
re-routing R8 before its inputs land.

### Verification of this witness

- `dos verify --plan autoctx --phase R8` → not shipped (no bound commit); this
  overlay is a blocked-report, not a claim of resolution.
- `#1918` state OPEN and no readout verb: `gh issue view 1918` +
  `git log --grep=#1918` (spec commits only).
- `#2200` state OPEN and no join commit: `gh issue view 2200` +
  `git log --all --grep="#2200"` (zero autoctx hits).
- `#2199` CLOSED and counter present: `git show --stat f3af9901c`
  (`internal/ctxknobs` + `fak index ctxknobs`).

### Batch policy

Re-grooms update this file in place; no new issue filed — the dependency stays
tracked on #1918 and #2200. This section supersedes the "Assumptions" block's
"R1 and R2 have landed" reading with the verified HEAD state above.
