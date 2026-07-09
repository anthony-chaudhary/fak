// Package brittleness is the DETECTOR-AND-CAPTURE for seams that "got lucky":
// process/commit/test outcomes that WORKED but only by timing, chance, or a
// symptom-patch that did not hold -- and the regressions those seams throw.
//
// Tier: foundation (1) - see internal/architest. This leaf imports only stdlib and
// pkg/scorecard (like its sibling detectors internal/orphanscan and
// internal/unwiredscore), so it declares at the lowest layer its imports allow; an
// upward import would fail the architest gate. See AGENTS.md and internal/architest
// for the layering contract.
//
// # Why this leaf exists
//
// fak already OBSERVES brittleness at the exact moment it is cheapest to know --
// and then throws the observation away:
//
//   - internal/affectedtests.ClassifyReruns names a package FLAKY_PASSED_ON_RETRY
//     (failed once, passed on a same-tree rerun) and then just drops the exit code
//     to 0 and moves on. The next agent to hit that same flake re-derives it from
//     scratch: another stash-and-rerun cycle proving a red that is not a regression.
//   - git history carries a file re-fixed by fix after fix (the earlier fix "got
//     lucky" and did not hold) and landings that were later reverted -- but nothing
//     folds that recurrence into a worst-first worklist a remediation loop can walk.
//
// The gap is not detection; it is CAPTURE WHEN THE SEAM IS FRESH. A flake, a
// recurring fix, a reverted landing is only cheaply explainable at the moment it is
// seen -- the failing tree is still on disk, the two fix SHAs are still adjacent,
// the revert reason is still in a human's head. Let that moment pass and the next
// agent pays the full re-derivation tax. This leaf is the durable capture: every
// Finding carries a Fresh evidence slice (the SHAs, the rerun tree-sha) recorded at
// detection time, and the fold ranks seams by severity-weighted recurrence so the
// worst repeat-offender is the first thing a remediation loop fixes.
//
// # Why the findings never gate (and pressure does the discriminating)
//
// A landed commit cannot be un-shipped on a no-rewrite shared trunk, so a
// history-window count can never be HARD, in-tree-mendable debt -- gating on it
// would red `make ci` for every peer with no mend available (the
// AGENTIC-DEV-ANTIPATTERNS note's load-bearing rule). So brittleness findings ride
// as pkg/scorecard SOFT signals: they never count as debt and never red a gate.
// The headline is brittleness_pressure -- an unbounded, severity-weighted recurrence
// sum that keeps discriminating past the point an A-F grade saturates -- the trend a
// continuous-improvement loop actually watches, not a pass/fail line.
//
// The pure core is Fold([]Finding) -> scorecard.Payload (data in, control-pane
// payload out; no disk, no git, no clock). The detectors that PRODUCE findings from
// a parsed git-log window live in detect.go, the flaky-capture bridge in flaky.go,
// and the CLI shell (cmd/fak/brittleness.go) owns the one git read.
package brittleness
