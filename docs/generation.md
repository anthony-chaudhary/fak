# Generation Contract

Generation is the product horizon for a piece of fak work. It answers: "which
horizon is this optimizing for, and what evidence would move it closer to now?"
It is not priority, not a branch strategy, and not a runtime exposure flag.

This contract backs epic
[#1625](https://github.com/anthony-chaudhary/fak/issues/1625). The matching
issue labels are `gen/now`, `gen/next`, `gen/second-next`, and `gen/future`.

## Streams

| Stream | Label | Milestone | Meaning |
|---|---|---|---|
| now | `gen/now` | `Generation G0 - Now / Immediate` | Improves the current product, operator loop, or trunk hygiene with a clear witness and no dependency on a future architecture bet. |
| next | `gen/next` | `Generation G1 - Next Gen` | Near-term foundation that should be runnable by agents soon, but still needs a gate, dogfood run, schema, or default-exposure proof. |
| second-next | `gen/second-next` | `Generation G2 - Second Next Gen` | Architectural option that needs simulation, compatibility policy, or cross-generation dependency management before it can become active product work. |
| future | `gen/future` | `Generation G3 - Future` | Research, market narrative, standards analogue, or long-horizon option that should stay visible without pretending it is on the current release train. |

Every generation issue should also carry the `generation` label. The stream
label and milestone should agree. If they do not, treat the mismatch as intake
drift: fix the label or milestone before using the issue for dispatch.

## Orthogonality

Generation is independent of priority. A `gen/future` issue can be high-value or
urgent to study, and a `gen/now` issue can be small cleanup. Priority answers
"how valuable or urgent is this?" Generation answers "which horizon owns the
evidence?"

Generation is independent of shared trunk. All streams still land through
`main`, by explicit path, with the same witness, DCO, and ship-stamp rules as
any other fak work. A stream label never authorizes a feature branch, a side
worktree escape, or stale trunk hygiene.

Generation is independent of runtime feature gates. A generation label says why
the work exists and what evidence promotes it. A feature gate decides whether
the code is reachable, default-on, default-off, or operator-only at runtime.
Next-generation code can land inert behind a gate; now-generation docs can ship
with no runtime gate at all.

Generation is independent of completion percentage. Ongoing optimization
programs still report a frontier and trend. Discrete deliverables can report
completion. Do not turn a never-done optimization program into a fake percent
because it has a generation label.

## Promotion Verbs

- `promote`: move work closer to `now` because evidence retired the blocker that
  kept it later.
- `demote`: move work farther from `now` because an assumption failed, a witness
  regressed, or the current product path no longer needs it.
- `retire`: close or remove the item because it is obsolete, superseded, or no
  longer worth carrying.
- `park`: keep the item true-but-not-active when it remains useful context but
  has no current owner, witness path, or decision.

Promotion should preserve issue history. Prefer updating labels, milestone, and
the evidence comment over opening a duplicate issue in another stream.

## Evidence

Promotion evidence depends on the surface:

- Code or CLI: a focused test, command output, or verifier that proves the new
  behavior and covers the generation-specific claim.
- Operator or loop surface: a before/after readout that shows less ambiguity,
  less contention, clearer dispatch, or a retired stale assumption.
- Planning or research: a committed note, issue update, project-field change, or
  saved view that a later agent can use without re-reading the parent epic.
- Release exposure: a release-readiness or feature-gate witness showing the work
  can reach the intended users without changing the stream into a branch.

Demotion or retirement evidence is equally concrete:

- The named assumption failed against live repo, issue, benchmark, or operator
  evidence.
- The witness is stale and no cheap re-witness path exists.
- The item duplicates a shipped issue or an active issue with a stronger witness.
- A later stream's option cost now exceeds its expected value.
- A runtime gate or compatibility policy proved the item cannot be safely exposed.

## Debt Metric

The milestone report carries a lightweight `debt_score` for each generation
lane. It is an operator signal, not a priority score or gate:

```text
debt_score =
  stale_issues
  + 3 * missing_witnesses
  + 2 * unpromoted_bets
  + 2 * label_ship_mismatches
```

The inputs are intentionally cheap:

- `stale_issues`: open discrete child work in that lane. Until the report reads
  issue age directly, this is a stale-risk proxy, not a calendar-age claim.
- `missing_witnesses`: tracked generation items whose child signal could not be
  read. This is weighted highest because an unwitnessed stream cannot be safely
  promoted or demoted.
- `unpromoted_bets`: later-horizon in-flight work and ongoing programs that have
  not moved closer to `now`.
- `label_ship_mismatches`: unclassified tracked work, meaning the report can see
  shipped work but cannot bind it to a generation label.

Promotion evidence that should reduce debt: closed witnessed child work, a
previously unreadable issue becoming measurable, a later-horizon bet moving to a
nearer stream with evidence, or commit/release sidecars matching issue labels.

Demotion or retirement evidence that should increase or resolve debt: repeated
missing witnesses, stale-risk work with no owner, a later-horizon bet whose
assumption failed, or a label/ship mismatch that proves the stream cannot be
trusted.

Invalidating assumption: the current metric uses tracked epic child counts as a
cheap proxy for stale issues and promotion state. If GitHub issue age,
project-field history, or commit sidecar coverage becomes cheap to read, the
weights should be recalibrated against those stronger witnesses.

## Intake Rules

At issue creation:

- Add `generation` plus exactly one stream label.
- Bind the matching generation milestone when the stream is known.
- If the reporter cannot classify the stream, use `needs-triage` instead of
  guessing. Do not use `gen/future` as a dumping ground for unclear work.
- State the horizon-specific witness in the issue body: test, command, report,
  dashboard, project configuration, or research memo.

Issue views expose the four dispatchable stream lanes:

- `generation-now`
- `generation-next`
- `generation-second-next`
- `generation-future`

Those views require both the stream label and its matching milestone. A missing
issue in a view usually means the label and milestone are not bound yet.

## Examples

`gen/now`: add a report field that immediately helps the operator decide today's
dispatch lane, with a captured command output as witness.

`gen/next`: add a default-off route or schema that can be dogfooded by agents
after one focused gate lands.

`gen/second-next`: define cross-generation dependency edges or a compatibility
policy that later code can enforce.

`gen/future`: research a standards analogue or market-facing narrative, with a
memo that names the decision it could influence.

## Anti-Patterns

- Priority laundering: moving low-priority work into `gen/future` so it looks
  strategically important.
- Current-work laundering: marking speculative work `gen/now` because it feels
  important but has no current-product witness.
- Branch-by-label: treating each stream as a branch, worktree, or long-lived
  integration lane.
- Feature-gate conflation: using `gen/next` as a substitute for a default-off
  runtime gate.
- Permanent parking: leaving old `gen/future` issues without owner, assumption,
  or recheck date.
- Hidden demotion: changing labels without recording the evidence that justified
  the move.

## Commits And Releases

Do not put `gen/*` in every commit subject. Subjects stay optimized for the
existing witness path: Conventional Commits, an issue reference when the commit
resolves an issue, and the `(fak <leaf>)` ship stamp. Generation metadata is a
body sidecar when the commit advances a generation issue:

```text
feat(tools): preserve generation sidecars #1634 (fak tools)

Generation: gen/now
Closes #1634
```

`fak commit --preview` preserves that sidecar in text and JSON output. The
sidecar is normalized from `now`, `next`, `second-next`, or `future` to the
matching `gen/*` label. A malformed sidecar is advisory so old commits keep
working, but the fix is to use one of the four labels exactly.

Release notes preserve generation without changing subjects. The release
context reads the sidecar from git commit bodies, and the release-note renderer
adds a machine-readable `generations` frontmatter block, a `## Generation`
summary, and per-commit `[gen/*]` bullet suffixes when metadata exists. A
release with no generation sidecars keeps the old note shape.

Generation remains orthogonal in these artifacts:

- Priority still comes from issue labels, milestones, and operator decision.
- Shared trunk rules do not change; every generation lands through `main`.
- Runtime feature gates still decide exposure, not the generation label.

## Assumptions To Recheck

- GitHub labels, milestones, and the project Generation field remain available
  and cheap for agents to read.
- The named issue views stay synchronized with the GitHub saved views humans see
  in the browser.
- Future agents can preserve the stream labels through commits, release notes,
  reports, and issue handoffs without bloating every commit subject.
- Stream labels reduce contention instead of becoming another stale taxonomy. If
  they do not, the evidence should demote or retire the mechanism, not defend it.
