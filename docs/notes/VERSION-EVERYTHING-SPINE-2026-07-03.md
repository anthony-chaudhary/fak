---
title: "Version everything: the per-module version spine (2026-07-03)"
description: "A per-module version spine that derives each module's rev from trunk-commit history, shipped as fak version modules plus a delta-stamping ledger."
---

# Version everything: the per-module version spine (2026-07-03)

Status: spine SHIPPED (`internal/modver` + `fak version modules`); the program
around it is the follow-on backlog (see the companion issue plan below).

## The doctrine

The repo already versions the *release* (root `VERSION`, `vX.Y.Z` tags,
`internal/appversion`) and the *binary* (`internal/binstamp`, the `fak version`
build stamp). Nothing versioned the *parts*: ~380 `internal/` leaves and ~30
`cmd/` binaries grow at wildly different rates, and none of that growth was a
queryable fact. "How fast is the gateway moving? which leaves are dormant?
did the slop score improve as the module churned?" had no mechanical answer.

Two rules make per-module versioning survivable on this shared trunk:

1. **Derived, never declared.** A hand-maintained per-module version file
   (×410) would rot within hours on a multi-session tree and generate infinite
   merge noise. A module's version is computed from history alone:
   `rev` = the count of trunk commits touching the module — monotonic,
   conflict-free, needs no cooperation from committers. Rendered
   `r<rev>+g<shortsha>` (e.g. `internal/gateway r652+g1f75c56d`).
2. **Stamp deltas, not snapshots.** The ledger
   (`docs/nightrun/module-versions.jsonl`, schema `fak-module-versions/1`)
   appends one row per module *per change*, so it grows proportionally to real
   movement. Stamping twice at the same HEAD is a no-op (witnessed by
   `TestStampModverLedgerRoundtrip`).

## The e2e (minimal, working, witnessed)

```
fak version modules                 # table: r<rev>+g<sha>  last-touch  module
fak version modules --json          # the same report, machine-readable
fak version modules --stamp         # append changed-module rows to the ledger
fak version modules --scores S.json # join a flat {"module": score} map
fak version modules --coverage coverage.out --stamp   # fold + join + stamp, one command
```

Live run at head c9a8c26a: 410 modules versioned; first stamp seeded 410
ledger rows; the immediate second stamp wrote 0 ("ledger current — 0 of 410
modules moved"). Logic witnesses live in `internal/modver/modver_test.go`
(parse, ghost-module exclusion, score join, delta convergence) and
`cmd/fak/version_modules_test.go` (render + ledger roundtrip).

## Versions → scores (the trend lens)

`fak version score-adapter --input scorecard-files.json` converts a per-file
scorecard envelope (`{"files":[{"path":"internal/gateway/a.go","score":80}, ...]}`)
into that flat map. Paths use the same module classifier as the version snapshot.
The aggregation rule is deliberately simple and explainable: **each file has equal
weight, and a module's score is the arithmetic mean of its file scores**. Duplicate
normalized paths and paths that cannot map to a module are refused rather than
silently skewing or mis-joining the result. The output can be passed directly to
`--scores` through a temporary file or shell process substitution.

`fak version modules --coverage coverage.out --stamp` (#2467) is the same join
with the fold built in, so a coverage-carrying ledger stamp lands from ONE
command rather than an adapter step plus a temporary file. It folds a Go
coverage profile into per-module **statement** coverage — covered statements
over total statements, the number `go tool cover -func` reports — rather than a
per-file mean, so a 5-statement helper cannot outweigh a 500-statement core
file. The profile names files by import path; because this repo's Go module is
the repository root, trimming that prefix recovers the repo-relative path the
same `moduleOf` classifier reads, and `internal/<leaf>/<subpkg>` folds into the
one module `internal/<leaf>`. Rows outside the module (a dependency in a merged
`-coverpkg` profile) and rows under no tracked keyspace are skipped, never
misfiled; a malformed profile is refused rather than partially folded. The
joined score carries `provenance: witnessed` — it is measured off a real run's
artifact, not modeled. `--scores` and `--coverage` are mutually exclusive: both
write the same score column.

`--scores` joins any flat `{"module": number}` map into the report and the
ledger rows, so a score series reads against the version series: score deltas
between rev r47 and r52 of a module are now a query over one JSONL file, not
an archaeology dig. The scorecard fleet (slop, code, maturity, …) keys by
*metric*, not module — the adapter that folds scorecard output into per-module
maps is deliberately follow-on work, not spine.

## How it plugs into the rest of the system

- **fak manage / dos**: a guard decision can cite the module rev it judged
  (staleness tells beyond the binary build stamp); `dos verify` can bind a
  claim to "module X at rev N" instead of a bare SHA. A future
  `MODULE_REV_STALE` refusal token can catch a loop acting on recall from an
  old rev — the same class as `STALE_RECALL`, but mechanical per module.
- **super loops / dispatch**: `--stamp` is a natural nightrun/super-loop turn;
  rev velocity per lane feeds dispatch (a hot module is a collision-risk
  signal; a dormant one is a cheap-lane signal).
- **skills / plugins**: skills and plugin surfaces are unversioned key spaces
  today; the same derived-rev scheme extends to `docs/`, `tools/`, `.claude/`
  skills, and policy manifests — each is one `trackedRoots` entry plus a
  moduleOf rule.
- **releases / ongoing updates**: each ledger row carries `app_version`, so
  "what moved between 0.36 and 0.37" is a filter, and release notes can be
  partially generated from module rev deltas.
- **agentic dev velocity**: rev-per-day per module is the growth trend the
  goal asks for; joined with scores it distinguishes "churn that improved
  things" from slop.

## Follow-on program

The 40-issue backlog (QA, dogfooding, productization, guard/dos integration,
skills/plugins key spaces, trend reporting) lives in the companion plan
`docs/notes/VERSION-EVERYTHING-BACKLOG-2026-07-03.md`, wave-planned with
`fak issue cohort`.
