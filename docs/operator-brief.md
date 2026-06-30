# Operator Brief

`fak operator brief` is the human pacing layer over the existing control panes.
It does not replace `fak cadence`, `fak program report`, or `fak milestone
report`; it folds their JSON outputs into one question: what is the system state,
what choice is being asked of a human, what challenge should be understood, what
can agents handle, and what is just background signal?

The strategic sibling is [`human-operator-effectiveness.md`](human-operator-effectiveness.md):
that page defines human-steerability as an ongoing program, and `fak program
report` tracks it beside kernel and cache programs.

Generate inputs:

```bash
go run ./cmd/fak cadence --json > cadence.json
go run ./cmd/fak program report --json > program.json
go run ./cmd/fak milestone report --json > milestone.json
go run ./cmd/fak operator heaviness --json > heaviness.json
go run ./cmd/fak operator brief --cadence cadence.json --program program.json --milestone milestone.json --heaviness heaviness.json
```

Or collect the missing source reports live in one command:

```bash
go run ./cmd/fak operator brief --collect
```

Artifact inputs still win when provided, so an operator can mix fresh and cached
evidence. The optional `--heaviness` input folds the operator-surface pressure
scorecard into the same brief; when omitted, it is not treated as a missing pane.
Pass the previous operator brief with `--previous` to compress a check-in into
what changed, what resolved, and what stayed open:

```bash
go run ./cmd/fak operator brief --collect --cadence cadence.json
go run ./cmd/fak operator brief --collect --previous last-brief.json --json > current-brief.json
```

For a faster live collection, pass source-specific cached evidence through the
brief:

```bash
go run ./cmd/fak operator brief --collect \
  --scores-from scorecard.json \
  --epics-from epics-with-counts.json
```

The brief has four buckets:

| Bucket | Meaning |
|---|---|
| `human` | A missing or unmeasured pane, or an explicit release/auth/manual decision. This is the only bucket that fails `--check`. |
| `agent` | Work that should be delegated back to agents, such as maturity debt, roadmap work, or release mechanics. |
| `watch` | Measured challenges that should change attention, not necessarily stop the fleet: regressed frontiers, partial signals, stale publish lag. |
| `background` | Healthy measured state that should stay visible without paging anyone. |

When a heaviness payload is present, hard `heaviness_debt` becomes agent work
because the fix is usually a repo change such as indexing a doc-map surface or
wiring an appeal channel. Non-zero `heaviness_pressure` with no hard debt becomes
a watch item: it should shape review cadence and consolidation choices, not page
a human by itself.

It also carries operator-first fields in JSON and in the human render:

| Field | Meaning |
|---|---|
| `state` | The current pace mode (`intervene`, `delegate`, `review`, or `monitor`) plus a one-line recommendation for how to use human attention. |
| `attention` | A concrete human timebox and read order. It labels the attention level (`interrupt`, `delegate`, `review`, or `none`), the recommended minutes to spend, when to look, and which sections to scan first. |
| `human_use` | The division of labor for this snapshot: what judgment a person contributes, what agents should keep doing, what attention trap to avoid, and when to escalate. |
| `coherence` | Whether the source reports describe one snapshot. `coherent` means the source date/commit stamps agree; `partial` means a pane is missing; `mixed` means cached reports disagree and the brief creates a watch item before treating the state as whole. |
| `since_previous` | Optional temporal compression from `--previous`: new, resolved, and still-present `human`/`agent`/`watch` items plus pace changes. Background state is intentionally excluded so normal telemetry churn does not become rereading work. |
| `strengths` | Evidence-backed work the operator can trust or delegate, so the brief shows what is working instead of training people to look only for failures. |
| `choices` | Concrete decisions an operator can make now. A missing witness becomes an `intervene` choice; delegable work becomes a `delegate` choice; watchlist-only friction becomes a `review` choice. |
| `challenges` | The hard parts to understand even when no one is paged: missing signals, explicit operator decisions, regressed frontiers, or partial report signals. |
| `learning_agenda` | The one bounded learning focus for this snapshot: what to practice now, what to skip, and which brief sections to drill into first. |
| `learning` | Short interpretation notes that teach the operator how to read the current state: witness before judgment, delegation boundaries, pace control, or when to stay out of the loop. |

The point of `attention` is to keep operators from reading every transcript just
because many agents are active. A clear brief says `none` with a zero-minute
budget; a delegable brief asks for a five-minute dispatch-boundary check; a
watchlist brief asks for a bounded review; only the `human` bucket escalates to
an immediate interruption.

The point of `human_use` is to make the operator a scarce judgment resource
instead of an overflow worker. It names the human job for the current state
(restore a witness, confirm priority, tune pace, or stay out), the agent-owned
work, and the trap to avoid such as inferring health from a partial pane or
hand-driving routeable agent work.

The point of `coherence` is to keep cached evidence honest. Mixing yesterday's
milestone report with today's cadence report may still be useful, but it is not
one system snapshot. The brief marks that as `mixed`, adds a `sources` watch
item, and points the operator at `--collect` or a same-run regeneration.

The point of `since_previous` is to make repeated check-ins cheap. A human who
already read the last brief should not reread every lane just because many agents
ran. The delta compares only the attention-bearing buckets (`human`, `agent`,
and `watch`), reports new/resolved/persistent items, and moves
`since_previous` to the front of the read order when there is a real change.

The point of `learning_agenda` is to keep operator learning paced. It picks one
mode-specific focus: restore a witness before judging, practice the delegation
boundary, treat a watch item as a review signal rather than a page, or stay out
when the brief is clear. That prevents "learn the whole fleet" from becoming the
hidden tax of running many agents.

Use `--check` as a paging gate:

```bash
go run ./cmd/fak operator brief --cadence cadence.json --program program.json --milestone milestone.json --check
```

Exit 1 means the brief found a `human` item. Agent work and watchlist items keep
exit 0 because they are measured state, not an operator interruption.
