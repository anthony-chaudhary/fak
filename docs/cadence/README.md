# Cadence report

`fak cadence` folds the four things worth watching on a regular cadence into one
control-pane report: the quality scores, feature maturity, work done, and the
release state. Before this, each lived in a different place. The scorecard
control pane reports scores (daily, in `garden.yml`). `fak maturity` reports the
capability lifecycle ladder on demand. The release-status fold reports releases
(every six hours, in `release-cadence.yml`). Work done had no cadence report at
all.

Run it:

```bash
go run ./cmd/fak cadence            # human snapshot of all four dimensions
go run ./cmd/fak cadence --json     # the control-pane envelope
go run ./cmd/fak cadence --check    # advisory gate (see below)
```

The four dimensions:

- **scores** come from `tools/scorecard_control_pane.py`: the portfolio debt
  across every scorecard, the grade-weighted severity debt, and the trend
  against its pinned baseline.
- **maturity** comes from `fak maturity`: the feature lifecycle index, ladder-skip
  debt, next-work backlog size, and per-rung distribution.
- **work** is read straight from git over a trailing window (7 days by default,
  `--window N` to change it): the commit count, and the subset that carry a
  `(fak <leaf>)` ship trailer.
- **releases** come from `tools/release_status.py` run offline: the latest tag
  and the next release action.

## The advisory gate

`--check` exits non-zero only when a dimension could not be measured, so the
report is incomplete. A regressed score or maturity ladder-skip does not fail it.
Those gates already live in the scorecard ratchet (`ci.yml`), and the cadence
report should not fail twice for the same reason. Regressions show up as advisory
lines instead.

## The durable ledger

`--append-history` writes one row to `history.jsonl` so the trend is visible
across weeks, not just against a single pinned baseline. Each row is one
`fak-cadence-ledger/1` line:

| field | meaning |
|---|---|
| `date` / `commit` / `generated_at` | when and at which commit the tick was taken |
| `verdict` | the folded report verdict |
| `scores_debt` / `scores_grade_debt` / `scores_measured` / `scores_trend` | raw portfolio debt, normalized severity debt, scorecard count, and the score trend |
| `standing_score` / `standing_delta` | the unbounded cadence standing: starts at 1000, then rises or falls by normalized health deltas |
| `standing_health_bp` / `standing_difficulty` / `standing_difficulty_delta` | the 0..100% normalized health input and the denominator/difficulty that made that tick harder or easier |
| `maturity_score` / `maturity_debt` / `maturity_backlog` | lifecycle index, ladder-skip debt, and next-work count |
| `maturity_proposed` / `maturity_prototyped` / `maturity_tested` / `maturity_dogfooded` / `maturity_default` | per-rung distribution, so the complete-but-not-dogfooded tail is trendable |
| `work_window_days` / `work_commits` / `work_ships` | the work-done window and counts |
| `release_version` / `release_action` | latest tag and next release action |

The standing fields are the durable alternative to eyeballing whether a bounded
`100` still means the same thing after the scorecard set changes. Each tick first
normalizes scorecard severity and maturity into a health percentage, records the
difficulty that produced it, then accumulates only the health delta into
`standing_score`. A harder tick with the same normalized health records a higher
difficulty and a flat standing; a real improvement can keep pushing standing
above its starting point, and a regression can pull it back down.

To extend the ledger, run the append and commit the one file by path:

```bash
go run ./cmd/fak cadence --append-history
git commit -s -- docs/cadence/history.jsonl -m "docs(cadence): record cadence tick (fak docs)"
```

The weekly `cadence.yml` workflow runs the report and surfaces it to the run's
step summary plus a downloadable artifact. It is dry-run-first: scheduled ticks
extend the ledger locally but do not push. A manual dispatch with dry_run=false
is the explicit arm that commits and pushes the row, matching the release-cadence
convention that scheduled jobs report (they don't auto-commit the shared trunk —
only an explicit manual arm does).
