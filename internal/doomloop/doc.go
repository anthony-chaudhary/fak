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
//   - relay.NoProgressEscape counts K empty legs of verified progress, but it
//     works per-leg, not on a live worker. (It is no longer orphaned: relay's
//     IdleAwareEscape embeds it (idle.go) and superloop.go:259 drives ObserveLeg
//     /Advances from non-test code. Neither reads a live worker's effort curve,
//     so the gap below is unchanged - only the "nothing calls it" half is stale.)
//   - loopmgr.WitnessGap measures real progress (witnessed-done), but only on
//     ENDED runs, retrospectively.
//
// The gap every one of them leaves open is NOT the two-axis reading itself:
// trajctl's ActivityDivergenceScorer already folds effort against the
// witnessed-commit curve, live at every Stop-hook turn end. It is the
// conjunction none of them ship - a two-axis read that needs NO declared
// Objective (a bare fleet worker has none), paired with a graduated,
// reversible-first correction RECOMMENDATION off that read. This leaf is that
// missing decision core. (Delivering the correction is a separate seam - owned
// by the shell - and is NOT yet wired to a live transport; see cmd/fak/doomloop.)
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
// reversible re-anchor packet queued for the session steer channel - delivery
// is a separate, not-yet-wired transport) -> ESCALATE (a structured operator
// hand-off once the doom persists). The core
// NEVER recommends a hard kill/replace on its own - that rung is an operator
// decision the caller must opt into explicitly, so an automatic run can never
// tear down a worker without a human in the loop.
//
// The core is pure: no clock (sample times are injected), no I/O, fail-closed
// (too little evidence yields UNKNOWN + no action). The impure shell
// (cmd/fak/doomloop) gathers the real samples and owns the single side effect of
// delivering a correction.
//
// # Invariants and contract guards
//
// Invariant: doom loop detection is fail-closed and bounded.
//   - When sample evidence is insufficient (< MinSamples), ambiguous, or unreadable,
//     the classifier strictly defaults to VerdictUnknown with CorrectNone.
//   - Recommendations are strictly reversible-first (OBSERVE -> NUDGE -> ESCALATE)
//     and never trigger destructive teardowns or worker kills autonomously.
//
// Guard:
//   - A window is only considered burning-flat when effort delta > 0 and verified progress delta == 0.
//   - At least TripWindows consecutive burning-flat windows are required before tripping VerdictDoomLoop.
package doomloop
