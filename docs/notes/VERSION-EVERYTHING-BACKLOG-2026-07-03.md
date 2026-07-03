# Version-everything backlog: epic + 44 contract-ready children (2026-07-03)

Companion to `VERSION-EVERYTHING-SPINE-2026-07-03.md`. The machine-readable
plan (the source of truth for filing) is
`version-everything-issue-plan-2026-07-03.json` beside this note — 45
candidates in `internal/issuecontract` Candidate shape.

Cohort verdict (`fak issue cohort --from-plan …`, 2026-07-03):
**44 dispatchable, 1 to-split (the epic — correct: it is the coordinating
parent, never a dispatch leaf), 0 triage, 0 refused**; 38 concurrency waves,
peak 6 leaves at once.

## Filing state

FILED 2026-07-03: epic **#2458** + 44 children **#2459–#2502**, all on
milestone `version-everything` (16), label `version-everything`, zero
failures. Each issue body opens with its `<!-- fak-issue-key: <key> -->`
dedupe marker (the plan JSON key), so a re-filing pass is idempotent and
`issue_closure_audit` can bind closes. Child order follows the plan's
candidate order: #2459 = `modver-tools-keyspace` … #2502 =
`modver-agentsmd-doctrine`.

## The map (by theme)

- **Key-space expansion** — version more of the system the same derived way:
  `tools/` scripts, `docs/` pages, skills, policy manifests, `dos.toml`
  reasons vocabulary, CI/cron workflow units, MCP index surfacing.
- **Versions → scores** — scorecard adapters (slop/code, coverage, maturity),
  `fak version trend`, score-regression advisory gate, provenance labels per
  the net-true-value standard.
- **QA / hardening** — merge-commit rev semantics (pin first: highest-risk
  correctness decision), rename continuity, ghost/tombstone report,
  Windows/WSL parity, rev-monotonicity property test, concurrent-stamp flock,
  ledger compaction, latency budget + incremental snapshot.
- **guard / dos integration** — `dos verify` claims bound to `module@rev`,
  a `MODULE_REV_STALE` refusal token, guard-audit rows carrying judged-surface
  revs, rev-velocity as a dispatch collision prior.
- **Productization / docs** — cli-reference + LEARNING-PATH, the
  `fak-module-versions/1` schema standards page, growth visualization demo,
  CLI ergonomics (`--top/--sort/--only`), info-pane lane rev, release-notes
  generation from rev deltas, per-module release staleness, milestone-report
  velocity, AGENTS.md doctrine paragraph, opt-in semver overlay for
  contract-carrying modules (abi, schemas), CLAIMS.md `module@rev` aging.
- **High-velocity process / super-loops** — nightly `--stamp` turn,
  commit-preview "this commit bumps modules X,Y" advisory, dormant-module
  finder as dispatch fuel, weekly trendreport fold.

Suggested first waves once filed: `modver-merge-commit-semantics` (pins the
spine's core semantics), `modver-trend-verb` + `modver-scorecard-adapter-slop`
(the versions→scores payoff), `modver-nightrun-stamp` (starts the time
series).
