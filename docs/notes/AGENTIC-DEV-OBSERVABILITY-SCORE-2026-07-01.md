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

`python tools/scorecard_control_pane.py --json` currently reports:

- `finding=scorecard_unmeasured`
- `total_debt=1126`
- `grade_debt=31`
- `measured=27`
- `errored=14`

The failing set is Go-backed scorecards, so this is not yet a trustworthy full-family
score. The checkout does not compile, and the control pane repeats the same build
failure across many scorecards instead of summarizing the unique blockers.

Observed compile blockers:

```text
cmd/fak/fleet.go:590:6: truncate redeclared in this block
    cmd/fak/benchmarks.go:209:6: other declaration of truncate
```

```text
internal/vcacheextract/codex.go:17:2: "github.com/anthony-chaudhary/fak/internal/vcacheobserve" imported and not used
```

The top severity buckets visible in the partial fold are:

```text
code F(8), code-slop F(8), tooling-quality F(8),
repo-hygiene C(2), release-readiness C(2),
seo B(1), steerability B(1), claim-repro B(1)
```

## Tickets filed

- #1997 - tier and surface the `metric_doc_coverage` tail.
- #1998 - surface and gate stale dogfood report age before live issue sync.
- #2043 - surface unique Go compile blockers when control-pane cards are unmeasured.

## Next measurement gate

Clear or isolate the dirty-tree Go build blockers, then rerun:

```bash
go build ./cmd/fak
python tools/scorecard_control_pane.py --json
```

Only after the control pane measures every scorecard should the remaining grade-debt rows
be treated as the next dispatch list.
