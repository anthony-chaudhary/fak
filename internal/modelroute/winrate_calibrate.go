// winrate_calibrate.go — WIN-RATE THRESHOLD CALIBRATION: derive the
// ThresholdRouter cutoff FROM a reference prompt distribution instead of
// hand-picking a raw score, so an operator states an economic dial — the
// fraction of traffic that should land on the strong tier — and the cutoff is
// computed from how the scorer actually scores real subjects.
//
// BORROWED CONTRACT (ADAPT, not a port): lm-sys/RouteLLM @
// 0b64fdafe049e596a3f5657c219329f24af24198 (Apache-2.0; upstream dormant since
// 2024-08-10, so this is a design reference, never a live dependency). The
// contract adapted here is the calibration loop in
// routellm/calibrate_threshold.py:50-58: score every prompt of a reference
// dataset with the router's win-rate scorer, then set the cutoff to
// np.quantile(win_rates, 1 - strong_model_pct) (numpy's default linear
// interpolation), so that dispatching under the inclusive "strong iff
// win_rate >= threshold" rule sends approximately strong_model_pct of the
// reference distribution to the strong model. Upstream documents the worked
// example: 50% strong calls for the "mf" router yields threshold 0.11593.
// Re-expressed in idiomatic Go against this package's Subject vocabulary; no
// upstream code block is copied. Study note: fak-private
// docs/research/STUDY-PRODUCTION-SCALING-ECON-2026-11-07.md, candidate
// routellm-2. Ticket: oss-port-routellm-routellm-2.
//
// WHY THIS BELONGS ON THE SPINE. routellm-1 (winrate.go) shipped the dispatch
// contract with a USER-PICKED cutoff: the operator had to know what a win
// rate of 0.3 means in traffic terms. This file adds the missing calibration
// half: pick the cutoff FROM a reference prompt distribution so an operator
// states an economic dial — the fraction of traffic on the strong tier —
// instead of a raw score cutoff. The package stays pure (stdlib only): the
// scorer is injected, calibration does no I/O.
//
// NOT TO BE CONFUSED WITH crossaudit_calibration.go, which calibrates AUDITOR
// VERDICT QUALITY (precision/recall/false-positive rates over audit
// observations) — a different concern that merely shares the word
// "calibration".
package modelroute

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

var (
	// ErrInvalidStrongPct is a calibration dial outside the closed [0,1]
	// interval (or non-finite): no honest traffic split exists there.
	ErrInvalidStrongPct = errors.New("modelroute: strong_pct out of range")
	// ErrEmptyReference is a nil or zero-length reference prompt distribution:
	// a quantile over nothing is undefined, so calibration refuses rather
	// than invent a cutoff.
	ErrEmptyReference = errors.New("modelroute: empty reference prompt distribution")
)

// CalibrateThreshold picks the win-rate cutoff that sends approximately
// strongPct of the reference prompt distribution to the strong tier,
// mirroring RouteLLM's calibrate_threshold loop: score EVERY subject of
// reference once, in order, collect P(strong wins), and return the
// (1-strongPct)-th quantile of those scores under numpy's default linear
// interpolation.
//
// The (1-strongPct) inversion is the whole trick: dispatch is "strong iff
// win_rate >= threshold" (inclusive), so a threshold equal to the q-quantile
// of the scored distribution admits exactly the top (1-q) fraction of it.
// With q = 1-strongPct, roughly strongPct of the reference subjects clear
// the bar:
//   - strongPct = 1 ⇒ quantile at p=0 ⇒ the MINIMUM score; every reference
//     subject meets it (inclusive >=) ⇒ 100% of the reference goes strong.
//   - strongPct = 0 ⇒ quantile at p=1 ⇒ the MAXIMUM score; only subjects
//     tied at the maximum meet it ⇒ ~0% strong (ties aside).
//
// Refusals: a nil scorer reuses the winrate.go "no scorer wired" sentinel
// (ErrInvalidModelPair, the same defect ThresholdRouter.Validate reports); a
// non-finite or out-of-[0,1] strongPct wraps ErrInvalidStrongPct; an empty
// reference wraps ErrEmptyReference; a scorer output that fails
// inUnitInterval wraps ErrInvalidWinRate. A scorer ERROR (transport failure
// mid-scan) propagates UNTOUCHED — it is diagnostic, and the house test style
// asserts errors.Is on the original.
func CalibrateThreshold(scorer WinRateScorer, reference []Subject, strongPct float64) (float64, error) {
	if scorer == nil {
		return 0, fmt.Errorf("%w: no win-rate scorer wired for calibration", ErrInvalidModelPair)
	}
	if !inUnitInterval(strongPct) {
		return 0, fmt.Errorf("%w: strongPct %v must be a finite value in [0,1]", ErrInvalidStrongPct, strongPct)
	}
	if len(reference) == 0 {
		return 0, fmt.Errorf("%w: got %d reference subjects", ErrEmptyReference, len(reference))
	}
	scores := make([]float64, 0, len(reference))
	for i, s := range reference {
		w, err := scorer.CalculateStrongWinRate(s)
		if err != nil {
			return 0, err // transport/diagnostic: propagate untouched
		}
		if !inUnitInterval(w) {
			return 0, fmt.Errorf("%w: scorer returned %v for reference subject %d", ErrInvalidWinRate, w, i)
		}
		scores = append(scores, w)
	}
	sort.Float64s(scores)
	return quantileSorted(scores, 1-strongPct), nil
}

// quantileSorted returns the p-th quantile of PRE-SORTED input under numpy's
// default LINEAR interpolation: position idx = p*(n-1); result =
// sorted[floor(idx)] + (idx-floor(idx))*(sorted[ceil(idx)]-sorted[floor(idx)]).
// Provenance: this mirrors numpy.quantile's default method="linear"; the same
// semantics already exist as internal/agentxbench/metrics.go Quantile — the
// shape is borrowed, the body reimplemented locally because modelroute is
// declared pure (stdlib only) and must not import agentxbench.
// Handles: empty → 0; p <= 0 → first element; p >= 1 → last element; n == 1 →
// that single score (idx = 0 lands exactly on it).
func quantileSorted(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n == 1 || p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[n-1]
	}
	idx := p * float64(n-1)
	lower := int(math.Floor(idx))
	upper := int(math.Ceil(idx))
	fraction := idx - float64(lower)
	if lower == upper {
		return sorted[lower]
	}
	return sorted[lower] + fraction*(sorted[upper]-sorted[lower])
}

// CalibratedThresholdRouter is the dispatch-ready assembly of the calibration
// half and the routellm-1 dispatch contract: calibrate the threshold over
// reference at the strongPct dial, build the ThresholdRouter, and return it
// ONLY after it passes ThresholdRouter.Validate — so the returned router is
// guaranteed dispatch-ready (wired scorer, valid pair, threshold in [0,1];
// the calibrated value is always in [0,1] because every collected score was
// checked against inUnitInterval and linear interpolation cannot leave the
// convex hull). Every calibration refusal (nil scorer, bad dial, empty
// reference, bad scorer output) propagates unchanged through
// CalibrateThreshold; an unwired pair is refused here at Validate with
// ErrInvalidModelPair.
func CalibratedThresholdRouter(name string, scorer WinRateScorer, pair ModelPair, reference []Subject, strongPct float64) (ThresholdRouter, error) {
	threshold, err := CalibrateThreshold(scorer, reference, strongPct)
	if err != nil {
		return ThresholdRouter{}, err
	}
	r := ThresholdRouter{Name: name, Scorer: scorer, Pair: pair, Threshold: threshold}
	if err := r.Validate(); err != nil {
		return ThresholdRouter{}, err
	}
	return r, nil
}
