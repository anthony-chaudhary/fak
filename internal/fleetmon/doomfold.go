package fleetmon

import "github.com/anthony-chaudhary/fak/internal/doomloop"

// doomfold.go is the #4148 liveness fold: it consults the shipped two-axis
// doom-loop classifier (internal/doomloop) so fleetmon.isAdvancing no longer
// calls a burning-flat worker "healthy". A worker whose transcript grows every
// sample while its VERIFIED progress (commits landed on its region) stays flat is
// the exact doom-loop signature; the activity-only liveness read is blind to it.
//
// The fold is OBSERVE-only (it reclassifies healthy -> attention; it never
// delivers a correction — that seam is owned by the doomloop shell) and BOUNDED:
// per worker it hands the classifier a fixed trailing window, so its cost is
// O(EscalateWindows), never O(full transcript history). That bound is the
// "meter the meter" contract, asserted in doomfold_test.go.

// doomWindowCap bounds how many trailing samples the fold hands the classifier
// per worker. The classifier only needs enough windows to confirm the longest
// streak it acts on (EscalateWindows), so a tail of EscalateWindows+1 samples is
// sufficient to preserve every NUDGE/ESCALATE decision AND caps the per-worker
// fold cost at O(EscalateWindows) regardless of how long the worker's real sample
// history is. An unset/invalid config folds the tuned default.
func doomWindowCap(cfg doomloop.Config) int {
	if cfg.TripWindows <= 0 || cfg.EscalateWindows < cfg.TripWindows {
		cfg = doomloop.DefaultConfig()
	}
	return cfg.EscalateWindows + 1
}

// boundedDoomTail returns at most n of the most-recent samples of s, so the fold
// reads a bounded window rather than the unbounded transcript history. n<=0 or
// len(s)<=n returns s unchanged.
func boundedDoomTail(s []doomloop.Sample, n int) []doomloop.Sample {
	if n <= 0 || len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}

// doomLoopConfirmed folds a worker's trailing verified-progress window through
// the shipped classifier and reports whether it is a CONFIRMED doom loop —
// burning effort with flat verified progress for >= TripWindows consecutive
// windows. It is pure, OBSERVE-only, and bounded (boundedDoomTail). Empty samples
// => (false, zero): absent progress evidence never manufactures a doom verdict.
func doomLoopConfirmed(samples []doomloop.Sample, cfg doomloop.Config) (bool, doomloop.Result) {
	if len(samples) == 0 {
		return false, doomloop.Result{}
	}
	if cfg == (doomloop.Config{}) {
		cfg = doomloop.DefaultConfig()
	}
	res := doomloop.Classify(boundedDoomTail(samples, doomWindowCap(cfg)), cfg)
	return res.Verdict == doomloop.VerdictDoomLoop, res
}
