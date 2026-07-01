package vcachecal

// resumecal.go closes issue #1614: calibrate the provider prompt-cache TTL ASSUMPTION
// against REAL observed resume timing, instead of trusting the §5 hypothesis (the bare
// 5-minute Anthropic ephemeral-cache breakpoint) forever. probe.go's FitCalibration
// already fits T from ACTIVE replay samples (send an anchor, wait, re-send); this file
// is the passive counterpart — it fits T (or flags the current assumption as still
// sound) from PASSIVE observations already sitting in real resume history: how long a
// session sat idle before it was brought back, and whether the provider actually served
// the prior prefix warm on that first turn. No probe traffic, no network call, no
// clock — a pure fold over caller-supplied gap/outcome buckets (§11.4 Law D2,
// calibrate-don't-assume, applied to the resume boundary instead of a synthetic probe).
//
// The shape is deliberately bucketed (a gap RANGE plus warm/cold tallies), not a raw
// per-sample list, because that is exactly what a real resume corpus back-test already
// produces (internal/resume.BacktestReport.Buckets) — this leaf stays generic (it knows
// nothing about resume.ObservedTurn or transcripts) so cmd/fak can adapt ANY bucketed
// warm/cold observation source into it without an upward import (resume is tier 1;
// vcachecal is tier 2 — the join lives in the tier-4 shell, same seam vcacheobserve
// already uses to bridge Calibration against real Turn telemetry).

// ResumeGapBucket is one wall-clock idle-gap range's observed warm/cold tally from real
// resume history: N resumes landed with an idle gap in [LoSeconds, HiSeconds), of which
// WarmN actually found the prior prefix still served (a positive cache read) and ColdN
// did not. It carries no session content — only the gap range and the two counts, the
// same content-free discipline the rest of this package keeps.
type ResumeGapBucket struct {
	LoSeconds int64
	HiSeconds int64
	WarmN     int
	ColdN     int
}

// N is the total scorable resumes in this bucket (WarmN + ColdN).
func (b ResumeGapBucket) N() int { return b.WarmN + b.ColdN }

// WarmRate is WarmN / N, the fraction of resumes in this gap range the provider actually
// served warm. Zero when the bucket is empty (never a divide-by-zero, never a fabricated
// non-zero from nothing observed).
func (b ResumeGapBucket) WarmRate() float64 {
	if n := b.N(); n > 0 {
		return float64(b.WarmN) / float64(n)
	}
	return 0
}

// TTLCalibrationVerdict is the fitted assessment of an assumed provider cache TTL against
// real resume timing: whether it is well-calibrated, and — when it is not — a suggested
// revision plus the closed reason. Every count is real observed history; nothing here is
// assumed zero.
type TTLCalibrationVerdict struct {
	// AssumedTTLMillis is the TTL under test — the provider's documented/hypothesized
	// ephemeral breakpoint (e.g. Anthropic's 5-minute default) the caller wants checked.
	AssumedTTLMillis int64 `json:"assumed_ttl_millis"`

	// N is the total scorable resumes folded into the verdict, across every bucket.
	N int `json:"n"`
	// WithinTTLWarmRate is the observed warm rate for resumes whose idle gap was AT OR
	// BELOW the assumed TTL — the assumption predicts these should mostly be warm.
	WithinTTLWarmRate float64 `json:"within_ttl_warm_rate"`
	// WithinTTLN is the sample count backing WithinTTLWarmRate (0 => the rate is
	// undefined/zero, not evidence of anything).
	WithinTTLN int `json:"within_ttl_n"`
	// PastTTLWarmRate is the observed warm rate for resumes whose idle gap EXCEEDED the
	// assumed TTL — the assumption predicts these should mostly be cold.
	PastTTLWarmRate float64 `json:"past_ttl_warm_rate"`
	// PastTTLN is the sample count backing PastTTLWarmRate.
	PastTTLN int `json:"past_ttl_n"`

	// WellCalibrated is true when the assumed TTL's predictions hold within tolerance:
	// within-TTL resumes are reliably warm AND past-TTL resumes are reliably cold. False
	// means the assumption and reality have diverged enough to act on.
	WellCalibrated bool `json:"well_calibrated"`
	// Reason is the closed token explaining the verdict (see the Reason* constants).
	Reason string `json:"reason"`

	// SuggestedTTLMillis is a revised TTL fit from the buckets when WellCalibrated is
	// false: the right-censored estimate — the upper bound (HiSeconds) of the widest gap
	// bucket whose warm rate still clears WarmRateFloor, converted to millis. Zero when
	// no bucket clears the floor (every observed gap looks cold; do not widen past what
	// evidence supports) or when the current assumption is already well-calibrated (no
	// revision is offered because none is needed).
	SuggestedTTLMillis int64 `json:"suggested_ttl_millis,omitempty"`

	// Buckets echoes the input, sorted by LoSeconds, so the verdict is self-describing —
	// a caller can render the per-bucket warm-rate curve the verdict was fit from without
	// re-deriving it.
	Buckets []ResumeGapBucket `json:"buckets"`
}

// The closed reason vocabulary for a TTLCalibrationVerdict — emittable, verifiable,
// closed, in the spirit of the DOS refusal set: a caller routes on the token, never on
// free-text drift.
const (
	// ReasonCalibratedNoEvidence: no scorable resume samples at all — the assumption is
	// neither confirmed nor refuted; report it as calibrated (fail-open: an unmeasured
	// TTL is not evidence of miscalibration) but the caller should treat this as inert.
	ReasonCalibratedNoEvidence = "calibrated_no_evidence"
	// ReasonCalibratedWithinTolerance: both rungs (within-TTL mostly warm, past-TTL
	// mostly cold) hold within tolerance — the assumption matches real resume timing.
	ReasonCalibratedWithinTolerance = "calibrated_within_tolerance"
	// ReasonTTLTooShort: past-TTL resumes are STILL coming back warm at a rate above the
	// floor — the provider is holding the prefix longer than the assumed TTL, so a
	// resume this fast is being projected cold when it would actually hit — the exact
	// "assumed TTL under-states real reuse" miscalibration #1614 exists to catch.
	ReasonTTLTooShort = "ttl_too_short"
	// ReasonTTLTooLong: within-TTL resumes are coming back cold at a rate below the warm
	// floor — the provider is evicting the prefix sooner than assumed, so a resume this
	// slow is being projected warm when it would actually miss (the inverse, costlier
	// miscalibration: a caller skips re-priming a prefix that is already gone).
	ReasonTTLTooLong = "ttl_too_long"
)

// WarmRateFloor is the minimum observed warm rate a gap bucket must clear to count as
// "reliably warm" for the SuggestedTTLMillis fit and the within-TTL calibration check.
// 0.8 mirrors the conservative bar internal/resume's back-test-derived EffectiveReuse
// windows were fit against (the corpus stayed "≈98% warm at 5-15 min", comfortably
// above this floor) — high enough that a revision is never offered from a coin-flip
// sample, low enough that real provider-side jitter does not itself look miscalibrated.
const WarmRateFloor = 0.8

// ColdRateFloor is the minimum observed COLD rate (1 - WarmRate) a gap bucket must clear
// for the past-TTL rung to count as "reliably cold". Symmetric with WarmRateFloor: a
// past-TTL bucket is expected to be mostly cold, and a rate at or above this floor
// confirms the assumption rather than refuting it.
const ColdRateFloor = 0.8

// MinCalibrationSamples is the minimum N a rung (within-TTL or past-TTL) needs before its
// observed rate is trusted for the verdict. Below it the rung is skipped (neither confirms
// nor refutes) rather than swinging the verdict on a handful of resumes — the same
// evidence-before-action discipline FitCalibration applies to a probe sample.
const MinCalibrationSamples = 5

// CalibrateResumeTTL is THE pure calibration fold: given real resume history bucketed by
// idle-gap range (WarmN/ColdN per range — exactly internal/resume.BacktestReport.Buckets'
// shape, adapted by the caller) and the TTL currently assumed, it reports whether that
// assumption is well-calibrated against what actually happened, and if not, what to
// revise it to. Same buckets in, same verdict out — no clock, no I/O, no network.
//
// The check has two independent rungs, each skipped (not scored) below
// MinCalibrationSamples:
//
//  1. Within-TTL: buckets whose HiSeconds <= assumed TTL should show a warm rate at or
//     above WarmRateFloor (the assumption predicts these are still cached). A rate below
//     the floor means the provider is evicting SOONER than assumed (ReasonTTLTooLong —
//     the assumption over-promises warmth).
//  2. Past-TTL: buckets whose LoSeconds >= assumed TTL should show a warm rate BELOW
//     (1 - ColdRateFloor) (the assumption predicts these have aged out). A warm rate at
//     or above that means the provider is holding the prefix LONGER than assumed
//     (ReasonTTLTooShort — the assumption under-states real reuse, exactly the
//     "systematically landing outside the assumed window" case #1614 names).
//
// When a rung fails, SuggestedTTLMillis is fit as the right-censored bound: the HiSeconds
// of the widest bucket whose own warm rate still clears WarmRateFloor (mirroring
// FitCalibration's "largest confirmed-warm delay bounds T" logic in probe.go, but over
// passively-observed resume gaps instead of active replay). It is a suggestion, never an
// auto-applied constant — the caller (a `fak resume validate`-shaped report, or
// internal/resume's own hand-fit EffectiveReuse* constants) decides whether to adopt it.
func CalibrateResumeTTL(buckets []ResumeGapBucket, assumedTTLMillis int64) TTLCalibrationVerdict {
	sorted := sortedBuckets(buckets)
	v := TTLCalibrationVerdict{AssumedTTLMillis: assumedTTLMillis, Buckets: sorted}
	assumedTTLSeconds := assumedTTLMillis / 1000

	var withinWarm, withinN, pastWarm, pastN int
	for _, b := range sorted {
		n := b.N()
		v.N += n
		if n == 0 {
			continue
		}
		if b.HiSeconds <= assumedTTLSeconds {
			withinWarm += b.WarmN
			withinN += n
		}
		if b.LoSeconds >= assumedTTLSeconds {
			pastWarm += b.WarmN
			pastN += n
		}
	}
	if withinN > 0 {
		v.WithinTTLWarmRate = float64(withinWarm) / float64(withinN)
		v.WithinTTLN = withinN
	}
	if pastN > 0 {
		v.PastTTLWarmRate = float64(pastWarm) / float64(pastN)
		v.PastTTLN = pastN
	}

	withinOK := withinN < MinCalibrationSamples || v.WithinTTLWarmRate >= WarmRateFloor
	pastOK := pastN < MinCalibrationSamples || v.PastTTLWarmRate <= 1-ColdRateFloor

	switch {
	case withinN < MinCalibrationSamples && pastN < MinCalibrationSamples:
		v.WellCalibrated = true
		v.Reason = ReasonCalibratedNoEvidence
	case !pastOK:
		// Past-TTL resumes are still landing warm above the cold floor: the provider
		// holds the prefix longer than assumed — the TTL is too short.
		v.WellCalibrated = false
		v.Reason = ReasonTTLTooShort
		v.SuggestedTTLMillis = suggestTTLMillis(sorted)
	case !withinOK:
		// Within-TTL resumes are landing cold below the warm floor: the provider evicts
		// sooner than assumed — the TTL is too long (an over-promise).
		v.WellCalibrated = false
		v.Reason = ReasonTTLTooLong
		v.SuggestedTTLMillis = suggestTTLMillis(sorted)
	default:
		v.WellCalibrated = true
		v.Reason = ReasonCalibratedWithinTolerance
	}
	return v
}

// suggestTTLMillis fits a revised TTL as the right-censored bound over the OBSERVED
// buckets: the HiSeconds of the widest (largest LoSeconds) bucket whose own warm rate
// still clears WarmRateFloor with at least MinCalibrationSamples backing it. This is the
// same "the largest confirmed-warm observation bounds T" logic FitCalibration applies to
// active probe replay (probe.go), applied here to passive resume-gap buckets. Buckets
// MUST already be sorted ascending by LoSeconds (sortedBuckets guarantees this). Returns
// 0 when no bucket clears the floor — never fabricate a suggestion evidence does not
// support.
func suggestTTLMillis(sorted []ResumeGapBucket) int64 {
	var boundSeconds int64 = -1
	for _, b := range sorted {
		if b.N() < MinCalibrationSamples {
			continue
		}
		if b.WarmRate() >= WarmRateFloor {
			if b.HiSeconds > boundSeconds {
				boundSeconds = b.HiSeconds
			}
		}
	}
	if boundSeconds < 0 {
		return 0
	}
	return boundSeconds * 1000
}

// sortedBuckets returns a copy of buckets ordered ascending by LoSeconds, so
// CalibrateResumeTTL is order-independent in its input (like FitConcentration demands
// descending rank order, this leaf demands — and enforces — ascending gap order) and
// never mutates the caller's slice.
func sortedBuckets(buckets []ResumeGapBucket) []ResumeGapBucket {
	out := make([]ResumeGapBucket, len(buckets))
	copy(out, buckets)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1].LoSeconds > out[j].LoSeconds; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}
