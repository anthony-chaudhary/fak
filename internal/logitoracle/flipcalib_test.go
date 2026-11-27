package logitoracle

import (
	"errors"
	"math"
	"testing"
)

func makeCleanServing(tokens int, flipEvery int, flipGap float64) []PositionFlip {
	out := make([]PositionFlip, tokens)
	for i := range out {
		if flipEvery > 0 && i%flipEvery == 0 {
			out[i] = PositionFlip{
				Position:       i,
				ServedToken:    2,
				RefArgmaxToken: 1,
				LogPRefArgmax:  -0.1,
				LogPServed:     -0.1 - flipGap,
			}
		} else {
			out[i] = PositionFlip{
				Position:       i,
				ServedToken:    1,
				RefArgmaxToken: 1,
				LogPRefArgmax:  -0.1,
				LogPServed:     -0.1,
			}
		}
	}
	return out
}

// fak-test:runtime fast est=5ms lane=default
func TestFlipCalibRequiresEightServingsAndValidatesIdentity(t *testing.T) {
	spec := CalibrationSpec{
		WorkloadHash: "wl-qwen38-v1",
		GPUKind:      "H200",
		ExposureMu:   0.01,
	}

	// Fewer than 8 servings must be rejected.
	tooFew := make([][]PositionFlip, 7)
	for i := range tooFew {
		tooFew[i] = makeCleanServing(50, 0, 0)
	}
	if _, err := CalibrateFlipEnvelope(spec, []TrustedEngineServings{{Engine: "fak-ref", Servings: tooFew}}); !errors.Is(err, ErrInvalidCalibration) {
		t.Fatalf("expected ErrInvalidCalibration for 7 servings, got %v", err)
	}

	// 8 clean servings per engine across two engines: most-permissive per rung & worst q_max.
	llamaServings := make([][]PositionFlip, 8)
	fakServings := make([][]PositionFlip, 8)
	for i := 0; i < 8; i++ {
		// llama.cpp: 1 flip of 0.6 nats per 100 tokens -> 0.01 rate, rung 0.5 only.
		llamaServings[i] = makeCleanServing(100, 100, 0.6)
		// fak-ref: 2 flips per 100 tokens (1 at 0.3 nats, 1 at 1.2 nats) -> 0.02 rate, rung 0.5 & 1.0.
		s := makeCleanServing(100, 0, 0)
		s[10] = PositionFlip{Position: 10, ServedToken: 2, RefArgmaxToken: 1, LogPRefArgmax: -0.1, LogPServed: -0.4}
		s[20] = PositionFlip{Position: 20, ServedToken: 2, RefArgmaxToken: 1, LogPRefArgmax: -0.1, LogPServed: -1.3}
		fakServings[i] = s
	}

	env, err := CalibrateFlipEnvelope(spec, []TrustedEngineServings{
		{Engine: "llama.cpp", Servings: llamaServings},
		{Engine: "fak-ref", Servings: fakServings},
	})
	if err != nil {
		t.Fatal(err)
	}

	// q_max must equal worst engine mean flip rate (0.02).
	if math.Abs(env.GateConfig.QMax-0.02) > 1e-12 {
		t.Fatalf("QMax = %v, want 0.02", env.GateConfig.QMax)
	}
	// Exposure target = ceil(ln(1/0.025) / 0.01) = ceil(368.8879) = 369.
	if env.ExposureTarget != 369 {
		t.Fatalf("ExposureTarget = %d, want 369", env.ExposureTarget)
	}
	// Garwood upper count for 0 events at alpha=0.025 is ln(40) ~= 3.688879.
	wantZeroRate := -math.Log(0.025) / 800.0
	if math.Abs(env.GateConfig.LambdaUp[2]-wantZeroRate) > 1e-9 {
		t.Fatalf("rung 1.5 LambdaUp = %v, want %v", env.GateConfig.LambdaUp[2], wantZeroRate)
	}

	// Mismatched workload hash or GPU kind must be refused.
	cand := makeCleanServing(400, 0, 0)
	if _, err := env.Evaluate("other-wl", "H200", cand); !errors.Is(err, ErrCalibrationMismatch) {
		t.Fatalf("expected ErrCalibrationMismatch for workload, got %v", err)
	}
	if _, err := env.Evaluate("wl-qwen38-v1", "L4", cand); !errors.Is(err, ErrCalibrationMismatch) {
		t.Fatalf("expected ErrCalibrationMismatch for GPU, got %v", err)
	}
	// Insufficient exposure must be refused.
	if _, err := env.Evaluate("wl-qwen38-v1", "H200", makeCleanServing(100, 0, 0)); !errors.Is(err, ErrInsufficientExposure) {
		t.Fatalf("expected ErrInsufficientExposure, got %v", err)
	}
	// Matching workload, GPU, and sufficient clean exposure passes.
	rep, err := env.Evaluate("wl-qwen38-v1", "H200", cand)
	if err != nil {
		t.Fatal(err)
	}
	if !rep.Pass {
		t.Fatalf("expected clean candidate pass, got %s", rep.FailureReason)
	}
}
