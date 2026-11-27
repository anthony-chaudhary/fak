package nativeperfslo

import (
	"errors"
	"math"
	"testing"
)

func TestGeomeanSLOSpeedupAndZeroing(t *testing.T) {
	key := BaselineKey{
		WorkloadFingerprint: "multitenant-slo-v1",
		Load:                1.11,
		GPUKind:             "H200",
	}
	budget := SLOBudget{TTFTSeconds: 0.300, TPOTSeconds: 0.050}
	// Baseline exactly at budget: r=1 -> B = 0.
	baseline := BaselineCredit{Key: key, Credit: 0}

	// Candidate 2x faster on both TTFT and TPOT -> r=0.5 -> L = ln(2) -> exp(L-B) = 2.0.
	reqs := []RequestLatency{
		{TTFTSeconds: 0.150, TPOTSeconds: 0.025},
		{TTFTSeconds: 0.100, TPOTSeconds: 0.025}, // bottleneck is TPOT: 0.025/0.050 = 0.5
	}
	res, err := EvaluateGeomeanSpeedup(GeomeanInput{
		Key:                key,
		Budget:             budget,
		Baseline:           baseline,
		Requests:           reqs,
		FaithfulnessPassed: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Zeroed || math.Abs(res.Score-2.0) > 1e-12 || math.Abs(res.UpperBound-2.0) > 1e-12 {
		t.Fatalf("got score=%v upper=%v zeroed=%v, want 2.0", res.Score, res.UpperBound, res.Zeroed)
	}

	// Faithfulness gate failure zeroes Score while preserving UpperBound.
	unfaithful, err := EvaluateGeomeanSpeedup(GeomeanInput{
		Key:                key,
		Budget:             budget,
		Baseline:           baseline,
		Requests:           reqs,
		FaithfulnessPassed: false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !unfaithful.Zeroed || unfaithful.Score != 0 || math.Abs(unfaithful.UpperBound-2.0) > 1e-12 {
		t.Fatalf("expected Score=0 and UpperBound=2.0, got %+v", unfaithful)
	}

	// Any failed request zeroes the run while preserving UpperBound.
	withFail, err := EvaluateGeomeanSpeedup(GeomeanInput{
		Key:                key,
		Budget:             budget,
		Baseline:           baseline,
		Requests:           append(reqs, RequestLatency{TTFTSeconds: 0.1, TPOTSeconds: 0.02, Failed: true}),
		FaithfulnessPassed: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !withFail.Zeroed || withFail.Score != 0 || withFail.FailedRequests != 1 || math.Abs(withFail.UpperBound-2.0) > 1e-12 {
		t.Fatalf("expected failed request to zero score with UpperBound=2.0, got %+v", withFail)
	}
}

func TestGeomeanSLOFloorAndKeyMismatch(t *testing.T) {
	key := BaselineKey{WorkloadFingerprint: "tab-v1", Load: 24.6, GPUKind: "H200"}
	budget := SLOBudget{TTFTSeconds: 0.200, TPOTSeconds: 0.010}

	// Zero latency is floored at MinSLORatio (1e-9), never +Inf/NaN.
	res, err := EvaluateGeomeanSpeedup(GeomeanInput{
		Key:                key,
		Budget:             budget,
		Baseline:           BaselineCredit{Key: key, Credit: 0},
		Requests:           []RequestLatency{{TTFTSeconds: 0, TPOTSeconds: 0}},
		FaithfulnessPassed: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(res.Score) || math.IsInf(res.Score, 0) || math.Abs(res.Score-1e9) > 1e-2 {
		t.Fatalf("expected floored score 1e9, got %v", res.Score)
	}

	// Mismatched fingerprint, load, or GPU kind must be refused.
	for _, badKey := range []BaselineKey{
		{WorkloadFingerprint: "other", Load: 24.6, GPUKind: "H200"},
		{WorkloadFingerprint: "tab-v1", Load: 10.0, GPUKind: "H200"},
		{WorkloadFingerprint: "tab-v1", Load: 24.6, GPUKind: "L4"},
	} {
		_, err := EvaluateGeomeanSpeedup(GeomeanInput{
			Key:                key,
			Budget:             budget,
			Baseline:           BaselineCredit{Key: badKey, Credit: 0},
			Requests:           []RequestLatency{{TTFTSeconds: 0.1, TPOTSeconds: 0.005}},
			FaithfulnessPassed: true,
		})
		if !errors.Is(err, ErrBaselineKeyMismatch) {
			t.Fatalf("expected ErrBaselineKeyMismatch for %+v, got %v", badKey, err)
		}
	}
}
