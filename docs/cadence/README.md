# Cadence report

`fak cadence` folds the three things worth watching on a regular cadence into one
control-pane report: the quality scores, the work done, and the release state.
Before this, each lived in a different place. The scorecard control pane reports
scores (daily, in `garden.yml`). The release-status fold reports releases (every
six hours, in `release-cadence.yml`). Work done had no cadence report at all.

Run it:

```bash
go run ./cmd/fak cadence            # human snapshot of all three dimensions
go run ./cmd/fak cadence --json     # the control-pane envelope
go run ./cmd/fak cadence --check    # advisory gate (see below)
```

The three dimensions:

- **scores** come from `tools/scorecard_control_pane.py`: the portfolio debt
  across every scorecard, plus the trend against its pinned baseline.
- **work** is read straight from git over a trailing window (7 days by default,
  `--window N` to change it): the commit count, and the subset that carry a
  `(fak <leaf>)` ship trailer.
- **releases** come from `tools/release_status.py` run offline: the latest tag
  and the next release action.

## The advisory gate

`--check` exits non-zero only when a dimension could not be measured, so the
report is incomplete. A regressed score does not fail it. That gate already lives
in the scorecard ratchet (`ci.yml`), and the cadence report should not fail twice
for the same reason. A score regression shows up as an advisory line instead.

## The durable ledger

`--append-history` writes one row to `history.jsonl` so the trend is visible
across weeks, not just against a single pinned baseline. Each row is one
`fak-cadence-ledger/1` line:

| field | meaning |
|---|---|
| `date` / `commit` / `generated_at` | when and at which commit the tick was taken |
| `verdict` | the folded report verdict |
| `scores_debt` / `scores_trend` | portfolio debt and its direction |
| `work_window_days` / `work_commits` / `work_ships` | the work-done window and counts |
| `release_version` / `release_action` | latest tag and next release action |

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
