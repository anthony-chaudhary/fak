# Generation Lifecycle Simulation (#1656)

Issue #1656 asks for a concrete artifact that exercises generation promotion
and demotion before those rules govern real work. This note is that artifact in
repo-native form: a deterministic worked scenario for the four generation
lifecycle verbs from `docs/generation.md`.

## State Model

Streams are a distance-from-now ladder. `retired` is terminal, and `parked` is an
active/inactive bit that rides alongside the stream.

```text
gen/now <- gen/next <- gen/second-next <- gen/future  (+ retired, + parked)
```

The lifecycle verbs are:

- `promote`: blocker retired, so the item moves one stream closer to `gen/now`.
- `demote`: an assumption failed or witness regressed, but a nearer path still exists.
- `retire`: the item was superseded, option cost exceeds value, or an assumption failed with no nearer path left.
- `park`: the item is still true but inactive because it lacks an owner, witness, or decision.

Precedence is deterministic: `retire > demote > park > promote`. A negative signal
therefore dominates a promotion signal in the same tick.

## Worked Scenario

Initial portfolio:

| item | stream | priority | gate |
|---|---|---:|---|
| B-seam | gen/next | P1 | off |
| E-dup | gen/next | P1 | off |
| A-optimizer | gen/second-next | P2 | off |
| D-shaky | gen/second-next | P2 | off |
| C-memo | gen/future | P3 | none |

Evidence timeline:

| tick | item | evidence | verb | result |
|---:|---|---|---|---|
| 1 | B-seam | prerequisite landed | promote | gen/next -> gen/now |
| 2 | A-optimizer | simulation shipped, nearer horizon | promote | gen/second-next -> gen/next |
| 3 | D-shaky | assumption failed, nearer path exists | demote | gen/second-next -> gen/future |
| 4 | C-memo | no owner, witness, or decision | park | stays gen/future, inactive |
| 5 | E-dup | shipped item now covers it | retire | gen/next -> retired |
| 6 | D-shaky | assumption failed, no nearer path remains | retire | gen/future -> retired |
| 7 | C-memo | live decision now needs it | promote | gen/future -> gen/second-next, active |

Distribution before and after:

| stream | before | after |
|---|---:|---:|
| gen/now | 0 | 1 |
| gen/next | 2 | 1 |
| gen/second-next | 2 | 1 |
| gen/future | 1 | 0 |
| retired | 0 | 2 |
| parked active items | 0 | 0 |

## Orthogonality

The lifecycle changes above do not alter priority, shared trunk, or runtime
feature gates:

- `priority` is unchanged by promotion or demotion. Generation is a horizon label, not urgency.
- `gate` is unchanged by promotion or demotion. Runtime exposure is a separate decision.
- `trunk` remains `main`. A generation stream is a label partition, not a branch or worktree.

## Promotion And Demotion Criteria For This Artifact

Promotion evidence:

- Move this from a note to a Go implementation, for example `fak generation simulate`, with pure logic in an internal package and a thin `cmd/fak` shell.
- Feed it real issue relabel or milestone history and capture a before/after readout from actual repository evidence.

Demotion or retirement evidence:

- Retire this model if `docs/generation.md` changes the stream ladder or verb set so this table no longer projects the source doctrine.
- Park it if static issue review and operator judgment cover all generation moves and no one replays timelines.

Invalidating assumption:

- The model assumes evidence arrives as discrete per-item signals and that one signal moves an item by one stream. If real governance requires graded confidence, cross-item coupling, or stream skipping, this table is too coarse and must be replaced.

## Continuation Path

A future agent can continue without rereading the whole epic:

1. Port this table to Go, not Python.
2. Add tests for precedence, clamping, terminal retirement, and orthogonality.
3. Add a captured replay over real generation history.
4. Keep the implementation bound to `docs/generation.md`; update both in the same change when doctrine shifts.
