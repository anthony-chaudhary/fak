# Agentic dev observability score - 2026-07-01

This is the current score read for fak's own agentic-development process. It records
what is measured, what is unmeasured, and which backlog tickets now carry the gaps.

## Clean surfaces

| Surface | Command | Result |
|---|---|---|
| Gateway observability | `python tools/observability_scorecard.py --json` | `OK`, score `96.8/A`, `observability_debt=0` |
| Agent readiness | `python tools/agent_readiness_scorecard.py --json` | `OK`, score `100.0/A`, `friction_debt=0` |

Remaining observability-scorecard tail: `metric_doc_coverage=46` with `54/241` emitted
`fak_*` metric families surfaced in docs, dashboards, or alerts. This is advisory, not
hard debt, and is tracked by #1997.

## Portfolio fold

The current full `python tools/scorecard_control_pane.py --json` run reports:

- `finding=scorecard_unmeasured`
- `total_debt=1151`
- `grade_debt=32`
- `measured=32`
- `errored=9`

This is still not a complete full-family score because nine Go-backed cards error before
emitting JSON: `concept-usage`, `maturity`, `growth-debt`, `support-maturity`,
`milestone`, `milestone-climb`, `loop-index`, `operator-heaviness`, and `propagation`.
A direct command build succeeds when written to a temp output path, so this is now a
scorecard-helper failure, not a whole-`cmd/fak` compile failure.

Representative scorecard errors:

```text
pkg\scorecard\scorecard.go:191:20: undefined: ValueFromScore
```

```text
pkg\scorecard\scorecard.go:133:19: undefined: anyFloat
```

The top severity buckets visible in the measured part of the fold are:

```text
code F(8), code-slop F(8), tooling-quality F(8),
repo-hygiene C(2), release-readiness C(2),
seo B(1), ui-quality B(1), dogfood-loop B(1), claim-repro B(1)
```

Earlier in the same read, dirty-tree compile blockers were volatile as peers advanced.
That remains an observability problem for the control pane: repeated Go-backed failures
need one unique blocker summary rather than 9-25 per-card echoes.

## Tickets filed

- #1997 - tier and surface the `metric_doc_coverage` tail.
- #1998 - surface and gate stale dogfood report age before live issue sync.
- #2043 - surface unique Go compile blockers when control-pane cards are unmeasured.
- #2066 - restore the `pkg/scorecard` helper symbols blocking Go-backed cards.

## Next measurement gate

Fix #2066, then rerun:

```bash
python tools/scorecard_control_pane.py --json
```

Only after `errored=0` should the remaining grade-debt rows be treated as the next
dispatch list.
