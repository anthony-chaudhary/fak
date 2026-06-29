package turnbench

import (
	"math/rand"
	"testing"
)

// stepSeries builds a deterministic series of nLeft samples around meanLeft followed by nRight
// samples around meanRight, each with Gaussian noise of scale sigma — an INJECTED STEP-CHANGE at
// index nLeft, the thing the gate must flag.
func stepSeries(seed int64, nLeft, nRight int, meanLeft, meanRight, sigma float64) []float64 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float64, 0, nLeft+nRight)
	for i := 0; i < nLeft; i++ {
		out = append(out, meanLeft+sigma*rng.NormFloat64())
	}
	for i := 0; i < nRight; i++ {
		out = append(out, meanRight+sigma*rng.NormFloat64())
	}
	return out
}

// noiseSeries builds a deterministic STATIONARY series — n samples around one mean with Gaussian
// noise and NO step. The detector must NOT flag this (no false red).
func noiseSeries(seed int64, n int, mean, sigma float64) []float64 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float64, n)
	for i := range out {
		out[i] = mean + sigma*rng.NormFloat64()
	}
	return out
}

// TestDetectChangePoints_FlagsInjectedStepChange is the issue's literal acceptance, positive half:
// an injected step-change is flagged — one persistent change point, located at the step, with the
// shift sign correct (an overhead regression is a positive shift).
func TestDetectChangePoints_FlagsInjectedStepChange(t *testing.T) {
	const boundary = 30
	series := stepSeries(20240617, boundary, 30, 100, 115, 3) // d = 15/3 = 5σ — a clear regression

	rep, err := DetectChangePoints("step-fixture", series, ChangePointConfig{})
	if err != nil {
		t.Fatalf("DetectChangePoints: %v", err)
	}
	t.Logf("step fixture: %s", rep.Note)

	if !rep.Shifted {
		t.Fatalf("an injected step-change was NOT flagged: %s", rep.Note)
	}
	if len(rep.ChangePoints) != 1 {
		t.Fatalf("want exactly 1 change point on a single-step series, got %d: %+v", len(rep.ChangePoints), rep.ChangePoints)
	}
	cp := rep.ChangePoints[0]
	if cp.Index < boundary-3 || cp.Index > boundary+3 {
		t.Errorf("change point located at index %d, want near the step at %d", cp.Index, boundary)
	}
	if cp.Shift <= 0 {
		t.Errorf("shift = %+.4g, want positive (the right segment is the higher-overhead regression)", cp.Shift)
	}
	if cp.PValue > rep.Alpha {
		t.Errorf("p-value %.4g exceeds alpha %.4g — should be significant", cp.PValue, rep.Alpha)
	}
	if len(rep.JSON()) == 0 {
		t.Error("empty JSON artifact")
	}
}

// TestDetectChangePoints_NoFalseRedOnStationaryNoise is the issue's literal acceptance, negative
// half: a stationary-noise series is NOT flagged (no false red).
func TestDetectChangePoints_NoFalseRedOnStationaryNoise(t *testing.T) {
	series := noiseSeries(771113, 60, 100, 3) // one mean throughout — pure benchmark noise

	rep, err := DetectChangePoints("noise-fixture", series, ChangePointConfig{})
	if err != nil {
		t.Fatalf("DetectChangePoints: %v", err)
	}
	t.Logf("noise fixture: %s", rep.Note)

	if rep.Shifted {
		t.Fatalf("stationary noise FALSE-RED: %d change point(s) flagged: %+v", len(rep.ChangePoints), rep.ChangePoints)
	}
}

// TestDetectChangePoints_FalsePositiveRateBounded is the principled "no false red": over many
// independent stationary-noise series the FRACTION that false-flag must sit near alpha, not just
// on one lucky seed — the witness analogue of sprt_test.go's 400-fixture rate assertions. A
// permutation test whose null is the distribution of the max is multiple-comparison-correct, so
// this rate is controlled BY CONSTRUCTION.
func TestDetectChangePoints_FalsePositiveRateBounded(t *testing.T) {
	const fixtures = 200
	seeds := deriveSeeds(424242, fixtures)
	falseReds := 0
	for _, seed := range seeds {
		series := noiseSeries(seed, 60, 100, 3)
		// Vary the permutation seed per fixture too, so the rate rests on the method, not one null.
		rep, err := DetectChangePoints("fp-fixture", series, ChangePointConfig{Seed: seed ^ 0x9E3779B9})
		if err != nil {
			t.Fatalf("DetectChangePoints: %v", err)
		}
		if rep.Shifted {
			falseReds++
		}
	}
	rate := float64(falseReds) / float64(fixtures)
	t.Logf("stationary-noise false-red rate = %.1f%% (%d/%d), alpha = 1%%", rate*100, falseReds, fixtures)
	if rate > 0.05 {
		t.Errorf("false-red rate %.1f%% exceeds the 5%% bound (alpha=1%%) — the gate would noise-red", rate*100)
	}
}

// TestDetectChangePoints_DetectionRateUnderStep is the power dual: over many independent step
// series the detector flags the regression nearly every time (and locates it near the step).
func TestDetectChangePoints_DetectionRateUnderStep(t *testing.T) {
	const (
		fixtures = 200
		boundary = 30
	)
	seeds := deriveSeeds(20260629, fixtures)
	detected, located := 0, 0
	for _, seed := range seeds {
		series := stepSeries(seed, boundary, 30, 100, 110, 3) // d = 10/3 ≈ 3.3σ
		rep, err := DetectChangePoints("power-fixture", series, ChangePointConfig{Seed: seed ^ 0x85EBCA77})
		if err != nil {
			t.Fatalf("DetectChangePoints: %v", err)
		}
		if rep.Shifted {
			detected++
			if cp := rep.ChangePoints[0]; cp.Index >= boundary-3 && cp.Index <= boundary+3 {
				located++
			}
		}
	}
	rate := float64(detected) / float64(fixtures)
	locRate := float64(located) / float64(fixtures)
	t.Logf("step detection rate = %.1f%% (%d/%d), located-near-step = %.1f%%", rate*100, detected, fixtures, locRate*100)
	if rate < 0.90 {
		t.Errorf("step detection rate %.1f%% below 90%% — the gate would miss real regressions", rate*100)
	}
}

// TestDetectChangePoints_ShortSeriesIsStable: a series too short to carry a MinSegment-wide side on
// each end is a valid "not enough runs yet" answer — stationary, not an error (the gate is called
// as the series grows).
func TestDetectChangePoints_ShortSeriesIsStable(t *testing.T) {
	rep, err := DetectChangePoints("short", []float64{1, 2, 3, 4}, ChangePointConfig{}) // < 2*MinSegment(=5)
	if err != nil {
		t.Fatalf("DetectChangePoints: %v", err)
	}
	if rep.Shifted || len(rep.ChangePoints) != 0 {
		t.Errorf("short series should be stable, got Shifted=%t with %d change point(s)", rep.Shifted, len(rep.ChangePoints))
	}
}

// TestDetectChangePoints_EmptySeriesErrors refuses an empty series rather than returning a vacuous
// report (mirrors RunSequentialGate's empty-stream refusal).
func TestDetectChangePoints_EmptySeriesErrors(t *testing.T) {
	if _, err := DetectChangePoints("empty", nil, ChangePointConfig{}); err == nil {
		t.Error("DetectChangePoints(nil) should error")
	}
}

// TestDetectChangePoints_Deterministic: the same series + config produces byte-identical artifacts,
// so a CI gate's read-back is reproducible (the permutation null is seeded).
func TestDetectChangePoints_Deterministic(t *testing.T) {
	series := stepSeries(7, 25, 25, 50, 70, 4)
	a, err := DetectChangePoints("det", series, ChangePointConfig{})
	if err != nil {
		t.Fatalf("DetectChangePoints: %v", err)
	}
	b, err := DetectChangePoints("det", series, ChangePointConfig{})
	if err != nil {
		t.Fatalf("DetectChangePoints: %v", err)
	}
	if string(a.JSON()) != string(b.JSON()) {
		t.Error("DetectChangePoints not deterministic for a fixed series + config")
	}
}
