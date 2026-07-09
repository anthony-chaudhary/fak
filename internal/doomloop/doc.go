// Package doomloop is the two-axis doom-loop classifier: it folds a live
// worker's effort-vs-verified-progress sample window into a closed verdict and
// a graduated, reversible-first correction recommendation.
//
// Tier: foundation (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
//
// # Why this leaf exists
//
// fak already has FOUR doom-loop DETECTORS, and none of them closes the loop:
//
//   - fleetmon.Classify walks the whole fleet, but its liveness test
//     (isAdvancing) reads ACTIVITY only - a growing transcript OR burned CPU is
//     "healthy". That is the exact signature of a doom loop: an agent that keeps
//     talking and burning tokens while landing nothing. It is misclassified
//     healthy.
//   - trajctl's ActivityDivergenceScorer models "busy but not moving" (activity
//     high while the witnessed-commit curve is flat), but its nudge actuator is
//     unwired and it needs a declared Objective a live fleet worker rarely has.
//   - relay.NoProgressEscape counts K empty legs of verified progress, but it is
//     orphaned - nothing calls it - and works per-leg, not on a live worker.
//   - loopmgr.WitnessGap measures real progress (witnessed-done), but only on
//     ENDED runs, retrospectively.
//
// The gap every one of them leaves open: no component reads BOTH axes - effort
// AND verified forward progress - on a LIVE worker over time, and takes a
// graduated corrective action. This leaf is that missing decision core.
//
// # The two axes
//
// Effort is what the worker is SPENDING: transcript lines, assistant turns,
// tool-uses, or tokens - any monotone counter of work done. The core reads its
// DELTA between samples ("is it still burning?").
//
// Progress is VERIFIED forward progress, never a self-report: commits landed on
// the worker's region, intent-ledger steps witnessed - a monotone counter of
// landings the worker did not author the truth of. The core reads its delta
// ("did anything actually land?").
//
// A doom loop is the conjunction, sustained: effort delta positive (burning)
// AND progress delta flat (nothing landed), for K consecutive windows. The
// K-consecutive discipline is deliberate (borrowed from relay.NoProgressEscape):
// a single flat window is normal - a large commit legitimately lands only after
// several turns of work - so the core errs toward NOT intervening until the
// flat-under-burn pattern is unambiguous. Mid-trajectory intervention degrades
// otherwise-healthy sessions, so the trip threshold protects the common case.
//
// # Reversible-first correction
//
// The recommendation ladder is graduated and tops out below anything
// destructive: OBSERVE (watching a sub-threshold streak) -> NUDGE (a soft,
// reversible re-anchor packet delivered to the session steer channel) ->
// ESCALATE (a structured operator hand-off once the doom persists). The core
// NEVER recommends a hard kill/replace on its own - that rung is an operator
// decision the caller must opt into explicitly, so an automatic run can never
// tear down a worker without a human in the loop.
//
// The core is pure: no clock (sample times are injected), no I/O, fail-closed
// (too little evidence yields UNKNOWN + no action). The impure shell
// (cmd/fak/doomloop) gathers the real samples and owns the single side effect of
// delivering a correction.
package doomloop
