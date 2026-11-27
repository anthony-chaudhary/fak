package logitoracle

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/quality"
)

// fak-test:runtime fast est=5ms lane=default
func TestFlipCurveExactTieAndNestedCounts(t *testing.T) {
	flips := []PositionFlip{
		// Exact token match -> not a flip.
		{Position: 0, ServedToken: 10, RefArgmaxToken: 10, LogPRefArgmax: -0.2, LogPServed: -0.2},
		// Different token but exact logp tie -> not a flip.
		{Position: 1, ServedToken: 11, RefArgmaxToken: 10, LogPRefArgmax: -1.0, LogPServed: -1.0},
		// Minor flip 0.3 nats -> counted in TotalFlips, below 0.5 rung.
		{Position: 2, ServedToken: 12, RefArgmaxToken: 10, LogPRefArgmax: -1.0, LogPServed: -1.3},
		// Severity 0.7 nats -> counted in 0.5 rung only.
		{Position: 3, ServedToken: 13, RefArgmaxToken: 10, LogPRefArgmax: -1.0, LogPServed: -1.7},
		// Severity 1.6 nats -> nested in 0.5, 1.0, 1.5 rungs.
		{Position: 4, ServedToken: 14, RefArgmaxToken: 10, LogPRefArgmax: -0.5, LogPServed: -2.1},
	}

	cfg := FlipGateConfig{
		Rungs:      []float64{0.5, 1.0, 1.5, 2.0},
		LambdaUp:   []float64{0.5, 0.5, 0.5, 0.5},
		QMax:       0.6,
		Multiplier: 1.5,
		Delta:      0.04,
	}
	rep, err := EvaluateFlipCurve(flips, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !rep.Pass {
		t.Fatalf("expected pass, got fail: %s", rep.FailureReason)
	}
	if rep.TotalFlips != 3 {
		t.Fatalf("TotalFlips = %d, want 3 (exact tie must not count)", rep.TotalFlips)
	}
	wantObs := []int{2, 1, 1, 0}
	for i, want := range wantObs {
		if rep.Rungs[i].Observed != want {
			t.Fatalf("rung[%d] (%.1f) observed = %d, want %d", i, rep.Rungs[i].RungNats, rep.Rungs[i].Observed, want)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestFlipCurveNonFiniteFails(t *testing.T) {
	flips := []PositionFlip{
		{Position: 0, ServedToken: 1, RefArgmaxToken: 1, LogPRefArgmax: -0.1, LogPServed: -0.1},
		{Position: 1, ServedToken: 2, RefArgmaxToken: 1, LogPRefArgmax: math.NaN(), LogPServed: -0.2},
	}
	cfg := FlipGateConfig{
		Rungs:      []float64{0.5, 1.0},
		LambdaUp:   []float64{0.1, 0.1},
		QMax:       1.0,
		Multiplier: 2.0,
		Delta:      0.05,
	}
	rep, err := EvaluateFlipCurve(flips, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Pass || rep.NonFiniteCount != 1 {
		t.Fatalf("expected fail with NonFiniteCount=1, got pass=%v nonFinite=%d", rep.Pass, rep.NonFiniteCount)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestFlipCurveRateAndRungCeilingFailures(t *testing.T) {
	flips := make([]PositionFlip, 100)
	for i := range flips {
		flips[i] = PositionFlip{Position: i, ServedToken: 1, RefArgmaxToken: 1, LogPRefArgmax: -0.1, LogPServed: -0.1}
	}
	// 10 flips at 0.6 nats -> rate 0.10.
	for i := 0; i < 10; i++ {
		flips[i] = PositionFlip{Position: i, ServedToken: 2, RefArgmaxToken: 1, LogPRefArgmax: -0.1, LogPServed: -0.7}
	}

	// Rate limit failure: flipRate 0.10 > 1.0 * 0.05.
	rateCfg := FlipGateConfig{
		Rungs:      []float64{0.5},
		LambdaUp:   []float64{0.5},
		QMax:       0.05,
		Multiplier: 1.0,
		Delta:      0.05,
	}
	rep, err := EvaluateFlipCurve(flips, rateCfg)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Pass {
		t.Fatalf("expected rate failure, got pass")
	}

	// Rung ceiling failure: lambda_up=0.01 -> mu=1.0, delta=0.01 -> kBase=4, scaled=4, obs=10 > 4.
	rungCfg := FlipGateConfig{
		Rungs:      []float64{0.5},
		LambdaUp:   []float64{0.01},
		QMax:       0.5,
		Multiplier: 1.0,
		Delta:      0.01,
	}
	rep2, err := EvaluateFlipCurve(flips, rungCfg)
	if err != nil {
		t.Fatal(err)
	}
	if rep2.Pass || !rep2.Rungs[0].Exceeded {
		t.Fatalf("expected rung ceiling failure, got %+v", rep2)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestFlipCurvePoissonCeilingAndMultiplicityBridge(t *testing.T) {
	// P(Poisson(1.0) > 4) = 1 - e^-1 * (1 + 1 + 1/2 + 1/6 + 1/24) = 1 - 65/(24*e) ~= 0.00366
	// P(Poisson(1.0) > 3) = 1 - 64/(24*e) ~= 0.01899
	// So for mu=1.0, delta=0.01, ceiling must be 4.
	if got := PoissonCeiling(1.0, 0.01); got != 4 {
		t.Fatalf("PoissonCeiling(1.0, 0.01) = %d, want 4", got)
	}
	// Large mu stability check (would underflow naive exp(-mu) walk).
	if got := PoissonCeiling(1000.0, 0.001); got <= 1000 || got > 1200 {
		t.Fatalf("PoissonCeiling(1000, 0.001) = %d, out of expected range", got)
	}

	flips := []PositionFlip{
		{Position: 0, ServedToken: 1, RefArgmaxToken: 1, LogPRefArgmax: -0.1, LogPServed: -0.1},
		{Position: 1, ServedToken: 2, RefArgmaxToken: 1, LogPRefArgmax: -0.1, LogPServed: -0.8},
	}
	cfg := FlipGateConfig{
		Rungs:      []float64{0.5, 1.0},
		LambdaUp:   []float64{0.5, 0.5},
		QMax:       0.6,
		Multiplier: 1.0,
		Delta:      0.05,
	}
	rep, err := EvaluateFlipCurve(flips, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ev := quality.Evidence{
		CaseID: "flipcurve-test",
		Provenance: quality.EvidenceProvenance{
			Model:     "Qwen3.8-4B",
			Tokenizer: "qwen3.8",
			Engine:    "fak-native",
			Oracle:    "flipcurve",
			Revision:  "rev1",
			Baseline:  "ref",
		},
	}
	comps := rep.ToMetricComparisons("Qwen3.8-4B", "default", quality.TierPR, 0.1, ev)
	dec, err := quality.ControlMultiplicity(quality.MultiplicityPolicy{
		Alpha:      0.05,
		Correction: quality.CorrectionHolm,
		Primary:    []string{"flip_rung_0.50", "flip_rung_1.00"},
	}, comps)
	if err != nil {
		t.Fatal(err)
	}
	if !dec.Pass {
		t.Fatalf("expected multiplicity pass, got blocks=%v refusals=%v", dec.Blocks, dec.Refusals)
	}
}
