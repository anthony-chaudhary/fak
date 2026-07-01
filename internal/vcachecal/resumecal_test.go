package vcachecal

import "testing"

// resumecal_test.go pins issue #1614's acceptance: calibrate the assumed provider cache
// TTL against REAL observed resume timing (bucketed warm/cold tallies), not a guessed
// constant. It mirrors probe.go's FitCalibration tests in spirit (measured vs. assumed,
// right-censored bound) but scores passively-observed resume gaps instead of active
// replay samples.

func TestCalibrateResumeTTLNoEvidenceIsCalibrated(t *testing.T) {
	v := CalibrateResumeTTL(nil, 300_000)
	if !v.WellCalibrated || v.Reason != ReasonCalibratedNoEvidence {
		t.Errorf("empty history: calibrated=%v reason=%q, want true/%q", v.WellCalibrated, v.Reason, ReasonCalibratedNoEvidence)
	}
	if v.SuggestedTTLMillis != 0 {
		t.Errorf("SuggestedTTLMillis = %d, want 0 (no evidence to fit from)", v.SuggestedTTLMillis)
	}
}

func TestCalibrateResumeTTLWithinToleranceMatchesAssumption(t *testing.T) {
	// Assumed TTL 300s (5m). Within-TTL buckets are reliably warm; past-TTL buckets are
	// reliably cold — the assumption matches reality on both rungs.
	buckets := []ResumeGapBucket{
		{LoSeconds: 0, HiSeconds: 60, WarmN: 20, ColdN: 0},
		{LoSeconds: 60, HiSeconds: 300, WarmN: 18, ColdN: 2},
		{LoSeconds: 300, HiSeconds: 900, WarmN: 1, ColdN: 19},
		{LoSeconds: 900, HiSeconds: 3600, WarmN: 0, ColdN: 20},
	}
	v := CalibrateResumeTTL(buckets, 300_000)
	if !v.WellCalibrated || v.Reason != ReasonCalibratedWithinTolerance {
		t.Fatalf("calibrated=%v reason=%q, want true/%q\n%+v", v.WellCalibrated, v.Reason, ReasonCalibratedWithinTolerance, v)
	}
	if v.SuggestedTTLMillis != 0 {
		t.Errorf("SuggestedTTLMillis = %d, want 0 (already well-calibrated)", v.SuggestedTTLMillis)
	}
	if v.N != 80 {
		t.Errorf("N = %d, want 80 (sum of all bucket counts)", v.N)
	}
}

// TestCalibrateResumeTTLTooShort is the #1614 headline case: resumes landing PAST the
// assumed 300s TTL are still coming back warm at a high rate — the provider is holding
// the prefix longer than assumed (fak's own EffectiveReuseSeconds widening, #940, is
// exactly this fact discovered by hand; this function discovers it mechanically).
func TestCalibrateResumeTTLTooShort(t *testing.T) {
	buckets := []ResumeGapBucket{
		{LoSeconds: 0, HiSeconds: 60, WarmN: 10, ColdN: 0},
		{LoSeconds: 300, HiSeconds: 900, WarmN: 18, ColdN: 2}, // past the 300s TTL, still ~90% warm
	}
	v := CalibrateResumeTTL(buckets, 300_000)
	if v.WellCalibrated || v.Reason != ReasonTTLTooShort {
		t.Fatalf("calibrated=%v reason=%q, want false/%q\n%+v", v.WellCalibrated, v.Reason, ReasonTTLTooShort, v)
	}
	// The widest bucket that still clears the 0.8 warm floor is [300,900) at 0.9 -> 900s.
	if v.SuggestedTTLMillis != 900_000 {
		t.Errorf("SuggestedTTLMillis = %d, want 900000 (the widest reliably-warm bucket's upper bound)", v.SuggestedTTLMillis)
	}
	if v.PastTTLWarmRate < 0.8 {
		t.Errorf("PastTTLWarmRate = %.2f, want >= 0.8 (the evidence the TTL is too short)", v.PastTTLWarmRate)
	}
}

// TestCalibrateResumeTTLTooLong is the inverse miscalibration: resumes WITHIN the assumed
// TTL are already coming back cold — the provider evicted sooner than assumed, so the
// current TTL over-promises warmth (a costlier mistake: a caller skips re-priming a
// prefix that is already gone).
func TestCalibrateResumeTTLTooLong(t *testing.T) {
	buckets := []ResumeGapBucket{
		{LoSeconds: 0, HiSeconds: 60, WarmN: 20, ColdN: 0},
		{LoSeconds: 60, HiSeconds: 300, WarmN: 2, ColdN: 18}, // still within the 300s TTL, but mostly cold
	}
	v := CalibrateResumeTTL(buckets, 300_000)
	if v.WellCalibrated || v.Reason != ReasonTTLTooLong {
		t.Fatalf("calibrated=%v reason=%q, want false/%q\n%+v", v.WellCalibrated, v.Reason, ReasonTTLTooLong, v)
	}
	// Only the [0,60) bucket clears the warm floor -> suggested TTL bounds to 60s.
	if v.SuggestedTTLMillis != 60_000 {
		t.Errorf("SuggestedTTLMillis = %d, want 60000", v.SuggestedTTLMillis)
	}
}

// TestCalibrateResumeTTLSkipsSparseRungs: a rung with fewer than MinCalibrationSamples is
// skipped (treated as non-refuting), so a handful of resumes cannot swing the verdict.
func TestCalibrateResumeTTLSkipsSparseRungs(t *testing.T) {
	buckets := []ResumeGapBucket{
		{LoSeconds: 0, HiSeconds: 60, WarmN: 20, ColdN: 0},   // within-TTL: 20 samples, reliably warm
		{LoSeconds: 300, HiSeconds: 900, WarmN: 2, ColdN: 0}, // past-TTL: only 2 samples (< MinCalibrationSamples)
	}
	v := CalibrateResumeTTL(buckets, 300_000)
	if !v.WellCalibrated {
		t.Errorf("a sparse past-TTL rung (n=2) must not refute calibration: %+v", v)
	}
	if v.PastTTLN != 2 {
		t.Errorf("PastTTLN = %d, want 2 (still reported, just not scored)", v.PastTTLN)
	}
}

func TestCalibrateResumeTTLBucketsSortedAndCopied(t *testing.T) {
	in := []ResumeGapBucket{
		{LoSeconds: 900, HiSeconds: 3600, WarmN: 1, ColdN: 9},
		{LoSeconds: 0, HiSeconds: 60, WarmN: 10, ColdN: 0},
	}
	v := CalibrateResumeTTL(in, 300_000)
	if len(v.Buckets) != 2 || v.Buckets[0].LoSeconds != 0 || v.Buckets[1].LoSeconds != 900 {
		t.Errorf("Buckets not sorted ascending: %+v", v.Buckets)
	}
	// The caller's slice must be untouched (order-independent, non-mutating fold).
	if in[0].LoSeconds != 900 {
		t.Errorf("caller's input slice was mutated: %+v", in)
	}
}

func TestResumeGapBucketWarmRateZeroSafe(t *testing.T) {
	var b ResumeGapBucket
	if b.WarmRate() != 0 {
		t.Errorf("empty bucket WarmRate = %g, want 0 (never a divide-by-zero)", b.WarmRate())
	}
	if b.N() != 0 {
		t.Errorf("empty bucket N = %d, want 0", b.N())
	}
}
