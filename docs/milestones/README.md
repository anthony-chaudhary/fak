# Milestone tracking ledger

This directory holds the durable state behind `fak milestone report` / `fak milestone
post` — the milestone-based tracking report that folds the two milestone signals (the
model×backend maturity **climb** and the epic **roadmap**) into one card and trends
them across weeks.

| File | What it is |
|---|---|
| `history.jsonl` | The append-only trend ledger (`fak-milestone-ledger/1`, one dated row per tick). `fak milestone report --append-history` appends to it; the report reads the last row to compute the per-tick trend. |
| `STATUS.md` | The generated climb-status snapshot (`fak milestone status-doc --write-doc`). |
| `baseline.json` · `tracked-epics.json` | The seeded baseline and the tracked-epic set (`fak milestone report --epics-from …`). |

## Editing the tracked-epic set as data (`--epics-from`, #1439)

The set of epics the roadmap dimension tracks is an in-code default
(`internal/milestonereport.TrackedEpics`), but it is *also* a committed, reviewable
data file — so adding or removing a tracked epic is a diff of
[`tracked-epics.json`](tracked-epics.json), not a code edit. Pass it (or any
`[]EpicSpec` file) to override the default:

```sh
fak milestone report --epics-from docs/milestones/tracked-epics.json
```

Absent the flag the report uses the in-code default (zero-config); the committed
seed mirrors that default exactly, a parity guarded by
`TestSeedMirrorsTrackedEpics` so the two surfaces can never silently drift.

The file takes one of two shapes:

- **specs-only** — a `specs` array of `{Number, Title, Generation, Label}`. The
  resolver reads each epic's children live via `gh` (the seed is this shape).
- **specs + pre-resolved counts** — add a `counts` array
  (`{Number, Closed, Total, Source}`) and the fold runs **fully offline**: no `gh`,
  deterministic. This is the hermetic path a gh-free CI run and the milestone tests
  take (`TestEpicsFilePreResolvedCountsFoldsOffline`).

A named `--epics-from` file that is missing, malformed, or has an empty `specs`
array is a **loud error**, never a silent fall-back to the default — a typo'd path
must not quietly track the wrong set (`TestLoadEpicsFileErrors`). The same flag is
accepted by `fak milestone scorecard` and threads through `fak operator --collect`.

## The scheduled milestone tick (#1440)

`fak milestone post` only posts when invoked, so the "tracking" in a tracking report
needs a recurring tick that (a) appends a durable ledger row so the trend accrues
unattended and (b) posts the card to #milestones on a cadence. That tick is **not** a
standalone cron: it rides the consolidated weekly cadence run in
[`.github/workflows/cadence.yml`](../../.github/workflows/cadence.yml) (Mondays 08:17
UTC), alongside the scores / work-done / releases dimensions, so one operator read
covers all four (issue #1443 folded it in). Each scheduled run:

1. `fak milestone report --append-history` — appends one dated row to `history.jsonl`.
   Same-day re-runs are idempotent: the ledger's trend read excludes a row with the same
   `generated_at`, and the commit step is a no-op when nothing changed.
2. `fak milestone status-doc --write-doc` — refreshes `STATUS.md`.
3. Commits `history.jsonl` + `STATUS.md` back to `main` and pushes, so the trend accrues
   durably across weeks (the acceptance for a *tracking* report).
4. `fak milestone post` — renders and posts the card to #milestones
   (`FAK_MILESTONE_CHANNEL`, default `C0BDYFRSW6S`), gated on the `FAK_MILESTONE_TOKEN`
   secret; without the token the run still accrues the ledger and fails open on the post.

**Kill switch.** Set the repo variable `FAK_MILESTONE_TICK=0` to fall back to a dry-run
tick (extend the ledger locally, do not commit or post) — mirroring the release cadence's
`FAK_AUTO_RELEASE=0`. Any other value (unset included) keeps the tick armed. A manual
`workflow_dispatch` commits only when run with `dry_run=false`.

> Supersedes the retired standalone `milestone-tick.yml` (commit `6c874ba5`), which
> #1443 folded into `cadence.yml` so milestones ride the loops that already exist rather
> than a separate cron that would double-append the ledger.
