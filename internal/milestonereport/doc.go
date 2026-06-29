// Package milestonereport folds the project's two milestone signals — the
// maturity CLIMB and the epic ROADMAP — into one read-only report envelope with a
// durable JSONL trend ledger, the milestone-shaped sibling of internal/cadencereport.
//
// Cadence trends the heartbeat numbers (scores / work / releases); this trends
// MILESTONES — how far the project has climbed, in two orthogonal senses:
//
//   - The CLIMB (the Maturity dimension): the model x backend grid's distribution
//     across the closed M0–M7 support-maturity ladder (internal/supportmaturity over
//     internal/covmatrix.Grid()). This is fully WITNESSED — every cell's rung is
//     lowered from the live grid, never self-reported — so the climb cannot be
//     inflated. It is the C1 keystone of the support-maturity epic (#1243) made
//     trendable: "is the grid climbing none->loads->runs->correct->optimized->parity
//     over time, or stalling?"
//   - The ROADMAP (the Epics dimension): per-tracked-epic child-issue completion,
//     read live from `gh`. A child signal is resolved by a PROVENANCE-honest priority
//     chain (a track label, then the epic body's task-list checklist) and each row
//     records WHICH source answered. An epic with no resolvable child signal is an
//     ERRORED row, never a fabricated 0%.
//
// The report is a REPORT CONTRACT, not a second quality gate (the same posture as
// cadencereport): --check fails ONLY when a dimension could not be MEASURED — i.e.
// the roadmap's `gh` read failed for every tracked epic. A 0%-complete epic or a
// regressed maturity centroid is a MEASURED fact surfaced as an advisory line, never
// a gate. The maturity dimension is pure (the grid is in-process and never errors),
// so it can never be the unmeasured dimension; only the `gh`-fed roadmap can.
//
// The pure surface (the dimension interpreters, the fold, the ledger parse/trend,
// render, gate) lives in milestonereport.go and is unit-testable with no process and
// no repo. The impure runners (the `gh` shell + the HEAD commit) live in collect.go.
package milestonereport
