package engine

import (
	"fmt"
	"testing"
	"time"
)

func TestKVQuantizationPressureLadderBeforeEviction(t *testing.T) {
	now := time.Unix(1000, 0)
	policy := KVQuantizationThresholds{DemotePressure: .90, PromotePressure: .65, AccuracyBudget: .02, MinDwell: time.Minute}
	state := KVQuantizationState{Precision: KVPrecisionFP16, Eligible: true, EstimatedError: .01}

	first := ChooseKVQuantization(now, .96, state, policy)
	if !first.Change || first.To != KVPrecisionINT8 || first.Reason != "pressure-demote" {
		t.Fatalf("FP16 pressure step = %+v", first)
	}
	state.Precision, state.LastTransition = first.To, now

	dwell := ChooseKVQuantization(now.Add(30*time.Second), .97, state, policy)
	if dwell.Change || dwell.Reason != "dwell" {
		t.Fatalf("dwell decision = %+v", dwell)
	}

	second := ChooseKVQuantization(now.Add(time.Minute), .97, state, policy)
	if !second.Change || second.To != KVPrecisionFP8 {
		t.Fatalf("INT8 pressure step = %+v", second)
	}

	bytes, err := KVQuantizedBytes(4096, second.To)
	if err != nil || bytes != 2048 {
		t.Fatalf("quantized bytes = %d, %v", bytes, err)
	}
}

func TestKVQuantizationPromotesOnlyAfterPressureClears(t *testing.T) {
	now := time.Unix(2000, 0)
	policy := KVQuantizationThresholds{DemotePressure: .90, PromotePressure: .65}
	state := KVQuantizationState{Precision: KVPrecisionFP8, Eligible: true}

	band := ChooseKVQuantization(now, .80, state, policy)
	if band.Change || band.Reason != "hysteresis" {
		t.Fatalf("band decision = %+v", band)
	}
	int8 := ChooseKVQuantization(now, .60, state, policy)
	if !int8.Change || int8.To != KVPrecisionINT8 || int8.Reason != "pressure-cleared-promote" {
		t.Fatalf("FP8 promotion = %+v", int8)
	}
	state.Precision = int8.To
	fp16 := ChooseKVQuantization(now, .60, state, policy)
	if !fp16.Change || fp16.To != KVPrecisionFP16 {
		t.Fatalf("INT8 promotion = %+v", fp16)
	}
}

func TestKVQuantizationEligibilityAndAccuracyFences(t *testing.T) {
	now := time.Unix(3000, 0)
	policy := KVQuantizationThresholds{DemotePressure: .90, PromotePressure: .65, AccuracyBudget: .01}
	tests := []struct {
		name   string
		state  KVQuantizationState
		reason string
	}{
		{"ineligible", KVQuantizationState{Precision: KVPrecisionFP16}, "ineligible"},
		{"accuracy", KVQuantizationState{Precision: KVPrecisionFP16, Eligible: true, EstimatedError: .02}, "accuracy-budget"},
		{"unknown", KVQuantizationState{Precision: "q4", Eligible: true}, "unknown-precision"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ChooseKVQuantization(now, .99, tt.state, policy)
			if got.Change || got.Reason != tt.reason {
				t.Fatalf("decision = %+v", got)
			}
		})
	}
}

type kvQuantExecutor struct {
	quantizeErr error
	compressErr error
	quantized   int
	compressed  int
}

func (e *kvQuantExecutor) QuantizeKV(_ string, _, _ KVPrecision) error {
	e.quantized++
	return e.quantizeErr
}
func (e *kvQuantExecutor) CompressColdKV(_ string, _ KVColdCodec) error {
	e.compressed++
	return e.compressErr
}

func TestApplyKVQuantizationColdCompressionAndMetrics(t *testing.T) {
	now := time.Unix(4000, 0)
	exec := &kvQuantExecutor{}
	got := ApplyKVQuantization(now, .95, KVQuantizationSpan{ID: "span-1", FP16Bytes: 4096, LastAccess: now.Add(-time.Hour), State: KVQuantizationState{Precision: KVPrecisionFP16, Eligible: true}}, KVQuantizationOptions{Policy: KVQuantizationThresholds{DemotePressure: .9, PromotePressure: .6}, EnableColdCompression: true, ColdAfter: time.Minute}, exec)
	if got.Fallback || got.Error != "" {
		t.Fatalf("run = %+v", got)
	}
	if got.Candidate.State.Precision != KVPrecisionINT8 || got.Candidate.ColdCodec != KVColdCodecSequence {
		t.Fatalf("candidate = %+v", got.Candidate)
	}
	if got.Metrics.Demotions != 1 || got.Metrics.ColdCompressions != 1 || got.Metrics.BytesBefore != 4096 || got.Metrics.BytesAfter != 2048 {
		t.Fatalf("metrics = %+v", got.Metrics)
	}
	if exec.quantized != 1 || exec.compressed != 1 {
		t.Fatalf("executor calls = %d, %d", exec.quantized, exec.compressed)
	}
}

func TestApplyKVQuantizationFailureFallsBackWithoutStateClaim(t *testing.T) {
	now := time.Unix(5000, 0)
	original := KVQuantizationSpan{ID: "span-2", FP16Bytes: 2048, State: KVQuantizationState{Precision: KVPrecisionFP16, Eligible: true}}
	got := ApplyKVQuantization(now, .99, original, KVQuantizationOptions{}, &kvQuantExecutor{quantizeErr: fmt.Errorf("backend refused conversion")})
	if !got.Fallback || got.Metrics.TransitionFailures != 1 || got.Error != "backend refused conversion" {
		t.Fatalf("fallback = %+v", got)
	}
	if got.Candidate.State.Precision != KVPrecisionFP16 || got.Metrics.BytesAfter != got.Metrics.BytesBefore {
		t.Fatalf("false transition claim = %+v", got)
	}
}

func TestApplyKVQuantizationColdCompressionIsExplicitAndColdOnly(t *testing.T) {
	now := time.Unix(6000, 0)
	exec := &kvQuantExecutor{}
	candidate := KVQuantizationSpan{ID: "span-3", FP16Bytes: 2048, LastAccess: now, State: KVQuantizationState{Precision: KVPrecisionINT8, Eligible: true}}
	got := ApplyKVQuantization(now, .8, candidate, KVQuantizationOptions{EnableColdCompression: true, ColdAfter: time.Minute}, exec)
	if got.Candidate.ColdCodec != KVColdCodecNone || exec.compressed != 0 {
		t.Fatalf("hot span compressed: %+v", got)
	}
	candidate.LastAccess = now.Add(-time.Hour)
	got = ApplyKVQuantization(now, .8, candidate, KVQuantizationOptions{}, exec)
	if got.Candidate.ColdCodec != KVColdCodecNone || exec.compressed != 0 {
		t.Fatalf("disabled codec ran: %+v", got)
	}
}

// TestAuxiliaryIndexKPrecisionIndependenceWitness (#9922):
// First witness: adds a modeled auxiliary cache row and proves its dtype changes memory
// footprint independently of the primary KV dtype.
func TestAuxiliaryIndexKPrecisionIndependenceWitness(t *testing.T) {
	span := KVQuantizationSpan{
		ID:              "span-sparse-mla",
		FP16Bytes:       4096, // Primary GQA KV
		IndexKFP16Bytes: 1024, // Auxiliary sparse IndexK
		State: KVQuantizationState{
			Precision:       KVPrecisionFP16,
			IndexKPrecision: KVPrecisionFP16,
			Eligible:        true,
		},
	}

	// 1. Both at FP16
	main, idxK, total, err := CalculateDualKVBytes(span)
	if err != nil {
		t.Fatalf("CalculateDualKVBytes failed: %v", err)
	}
	if main != 4096 || idxK != 1024 || total != 5120 {
		t.Fatalf("FP16/FP16: main=%d, idxK=%d, total=%d; want 4096/1024/5120", main, idxK, total)
	}

	// 2. Change ONLY IndexK precision to FP8: main KV MUST remain 4096, IndexK drops to 512
	span.State.IndexKPrecision = KVPrecisionFP8
	main2, idxK2, total2, err := CalculateDualKVBytes(span)
	if err != nil {
		t.Fatalf("CalculateDualKVBytes failed: %v", err)
	}
	if main2 != 4096 {
		t.Fatalf("changing IndexK precision mutated main KV bytes: got %d, want 4096", main2)
	}
	if idxK2 != 512 || total2 != 4608 {
		t.Fatalf("FP16/FP8: idxK=%d, total=%d; want 512/4608", idxK2, total2)
	}

	// 3. Change ONLY IndexK precision to INT4: main KV MUST remain 4096, IndexK drops to 256
	span.State.IndexKPrecision = KVPrecisionINT4
	main3, idxK3, total3, err := CalculateDualKVBytes(span)
	if err != nil {
		t.Fatalf("CalculateDualKVBytes failed: %v", err)
	}
	if main3 != 4096 {
		t.Fatalf("changing IndexK precision mutated main KV bytes: got %d, want 4096", main3)
	}
	if idxK3 != 256 || total3 != 4352 {
		t.Fatalf("FP16/INT4: idxK=%d, total=%d; want 256/4352", idxK3, total3)
	}

	// 4. Change main KV precision to INT8 while keeping IndexK at INT4:
	// main KV drops to 2048, IndexK MUST remain 256
	span.State.Precision = KVPrecisionINT8
	main4, idxK4, total4, err := CalculateDualKVBytes(span)
	if err != nil {
		t.Fatalf("CalculateDualKVBytes failed: %v", err)
	}
	if main4 != 2048 {
		t.Fatalf("INT8 main KV bytes = %d, want 2048", main4)
	}
	if idxK4 != 256 {
		t.Fatalf("changing main KV precision mutated IndexK bytes: got %d, want 256", idxK4)
	}
	if total4 != 2304 {
		t.Fatalf("INT8/INT4 total = %d, want 2304", total4)
	}
}
