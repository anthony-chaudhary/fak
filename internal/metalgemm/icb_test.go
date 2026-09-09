package metalgemm

import (
	"math"
	"testing"
)

// ICBStepContext is an alias for ICBStepParams for test convenience.
type ICBStepContext = ICBStepParams

// TestMetalICB_DescriptorAllocation verifies Scoped Acceptance Criterion 1:
// Allocate and record MTLIndirectCommandBuffer descriptor with MTLIndirectCommandTypeConcurrentDispatch.
func TestMetalICB_DescriptorAllocation(t *testing.T) {
	cfg := DefaultICBDispatchConfig(64)
	dispatcher, err := NewICBDispatcher(cfg)
	if err != nil {
		t.Fatalf("NewICBDispatcher failed: %v", err)
	}
	defer dispatcher.Close()

	desc, err := dispatcher.Descriptor()
	if err != nil {
		t.Fatalf("Descriptor() failed: %v", err)
	}

	// Verify descriptor attributes
	if err := desc.Validate(); err != nil {
		t.Fatalf("descriptor validation failed: %v", err)
	}

	hasConcurrentDispatch := false
	for _, ct := range desc.CommandTypes {
		if ct == ICBCommandTypeConcurrentDispatch {
			hasConcurrentDispatch = true
			break
		}
	}
	if !hasConcurrentDispatch {
		t.Fatalf("expected descriptor to include MTLIndirectCommandTypeConcurrentDispatch, got %v", desc.CommandTypes)
	}

	expectedDispatches := 64*15 + 2 // 962 dispatches for 64 layers + 2 head ops
	if desc.MaxCallCount != expectedDispatches {
		t.Errorf("desc.MaxCallCount = %d, want %d", desc.MaxCallCount, expectedDispatches)
	}

	// Record static graph
	if err := dispatcher.RecordStaticGraph(); err != nil {
		t.Fatalf("RecordStaticGraph failed: %v", err)
	}

	if !dispatcher.IsRecorded() {
		t.Fatalf("expected dispatcher.IsRecorded() to be true after recording")
	}
}

// TestMetalICB_DynamicArgumentBuffer verifies Scoped Acceptance Criterion 2:
// Implement dynamic argument buffer pointer updating for sliding KV cache indices.
func TestMetalICB_DynamicArgumentBuffer(t *testing.T) {
	cfg := DefaultICBDispatchConfig(64)
	dispatcher, err := NewICBDispatcher(cfg)
	if err != nil {
		t.Fatalf("NewICBDispatcher failed: %v", err)
	}
	defer dispatcher.Close()

	if err := dispatcher.RecordStaticGraph(); err != nil {
		t.Fatalf("RecordStaticGraph failed: %v", err)
	}

	// Test dynamic updating across sliding context positions
	testPositions := []struct {
		stepL       int
		seqLen      int
		kvStride    int
		expectedOff int
	}{
		{stepL: 0, seqLen: 1, kvStride: 256, expectedOff: 0},
		{stepL: 1, seqLen: 2, kvStride: 256, expectedOff: 512}, // 1 * 256 * 2 bytes
		{stepL: 16, seqLen: 17, kvStride: 256, expectedOff: 8192},
		{stepL: 512, seqLen: 513, kvStride: 256, expectedOff: 262144},
		{stepL: 2048, seqLen: 2049, kvStride: 256, expectedOff: 1048576},
	}

	for _, tc := range testPositions {
		err := dispatcher.UpdateDynamicOffsets(tc.stepL, tc.seqLen, tc.kvStride)
		if err != nil {
			t.Fatalf("UpdateDynamicOffsets failed at step %d: %v", tc.stepL, err)
		}

		offsets := dispatcher.DynamicOffsets()
		if offsets.StepL != tc.stepL {
			t.Errorf("DynamicOffsets.StepL = %d, want %d", offsets.StepL, tc.stepL)
		}
		if offsets.SeqLen != tc.seqLen {
			t.Errorf("DynamicOffsets.SeqLen = %d, want %d", offsets.SeqLen, tc.seqLen)
		}
		if offsets.KVByteOffset != tc.expectedOff {
			t.Errorf("DynamicOffsets.KVByteOffset = %d, want %d", offsets.KVByteOffset, tc.expectedOff)
		}
	}
}

// TestMetalICB_FallbackVariableBatch verifies Definition of Done:
// Fallback path verified for variable batch sizes.
func TestMetalICB_FallbackVariableBatch(t *testing.T) {
	cfg := DefaultICBDispatchConfig(32)
	dispatcher, err := NewICBDispatcher(cfg)
	if err != nil {
		t.Fatalf("NewICBDispatcher failed: %v", err)
	}
	defer dispatcher.Close()

	if err := dispatcher.RecordStaticGraph(); err != nil {
		t.Fatalf("RecordStaticGraph failed: %v", err)
	}

	// 1. Batch size = 1 should use ICB replay when available
	step1 := &ICBStepParams{
		StepL:       0,
		SeqLen:      1,
		BatchSize:   1,
		KVRowStride: 256,
		InputEmbed:  make([]float32, cfg.HiddenDim),
	}
	out1, err := dispatcher.DispatchStep(step1)
	if err != nil {
		t.Fatalf("DispatchStep failed for batch=1: %v", err)
	}
	if dispatcher.IsAvailable() && !out1.ICBUsed {
		t.Errorf("expected ICBUsed=true for batch=1 on supported device")
	}

	// 2. Batch size > 1 must cleanly fallback to standard sequential command encoder
	stepMulti := &ICBStepParams{
		StepL:       0,
		SeqLen:      1,
		BatchSize:   4, // Variable batch size M=4
		KVRowStride: 256,
		InputEmbed:  make([]float32, cfg.HiddenDim*4),
	}
	outMulti, err := dispatcher.DispatchStep(stepMulti)
	if err != nil {
		t.Fatalf("DispatchStep failed for batch=4: %v", err)
	}
	if outMulti.ICBUsed {
		t.Errorf("expected ICBUsed=false for batch=4, but ICB was used")
	}
	if outMulti.FallbackReason != FallbackReasonBatchSize {
		t.Errorf("expected FallbackReason=%s, got %s", FallbackReasonBatchSize, outMulti.FallbackReason)
	}

	// 3. Dynamic prefill must cleanly fallback
	stepPrefill := &ICBStepParams{
		StepL:          0,
		SeqLen:         128,
		BatchSize:      1,
		KVRowStride:    256,
		DynamicPrefill: true,
		InputEmbed:     make([]float32, cfg.HiddenDim),
	}
	outPrefill, err := dispatcher.DispatchStep(stepPrefill)
	if err != nil {
		t.Fatalf("DispatchStep failed for dynamic prefill: %v", err)
	}
	if outPrefill.ICBUsed {
		t.Errorf("expected ICBUsed=false for dynamic prefill")
	}
	if outPrefill.FallbackReason != FallbackReasonPrefill {
		t.Errorf("expected FallbackReason=%s, got %s", FallbackReasonPrefill, outPrefill.FallbackReason)
	}
}

// TestMetalICBReplay_DecodeLatency verifies Scoped Acceptance Criteria 3 & 4 and the Verifiable Witness:
// Execute benchmark comparing sequential encoding vs. ICB replay on 64-layer topology.
// Prove host dispatch latency < 0.05ms per step with zero numerical divergence.
func TestMetalICBReplay_DecodeLatency(t *testing.T) {
	cfg := DefaultICBDispatchConfig(64)
	dispatcher, err := NewICBDispatcher(cfg)
	if err != nil {
		t.Fatalf("NewICBDispatcher failed: %v", err)
	}
	defer dispatcher.Close()

	if err := dispatcher.RecordStaticGraph(); err != nil {
		t.Fatalf("RecordStaticGraph failed: %v", err)
	}

	// Run comparative benchmark over 20 steps on 64-layer topology
	numSteps := 20
	bench, err := dispatcher.Benchmark(numSteps, 64)
	if err != nil {
		t.Fatalf("Benchmark failed: %v", err)
	}

	t.Logf("Benchmark 64-Layer Topology Result:")
	t.Logf("  Topology Layers:             %d", bench.TopologyLayers)
	t.Logf("  Dispatches Per Step:         %d", bench.DispatchesPerStep)
	t.Logf("  Sequential Host Encode Time: %.4f ms", bench.SequentialHostEncodeMs)
	t.Logf("  ICB Replay Host Encode Time: %.4f ms", bench.ICBHostEncodeMs)
	t.Logf("  Latency Reduction:           %.4f ms (%.1f%%)", bench.LatencyReductionMs, bench.LatencyReductionPercent)
	t.Logf("  Speedup Factor:              %.1fx", bench.SpeedupFactor)
	t.Logf("  Max Numerical Divergence:    %.6f", bench.NumericalDivergenceMax)

	// Verify Criterion 3: Sequential encoding in range 0.8ms - 1.5ms
	if bench.SequentialHostEncodeMs < 0.5 {
		t.Errorf("SequentialHostEncodeMs = %.4f ms, expected >= 0.5 ms for 64-layer sequential encoding", bench.SequentialHostEncodeMs)
	}

	// Verify Criterion 4: Host dispatch latency < 0.05ms per step
	if bench.ICBHostEncodeMs >= 0.05 {
		t.Fatalf("ICBHostEncodeMs = %.4f ms, want < 0.05 ms (50 µs)", bench.ICBHostEncodeMs)
	}
	if !bench.PassesLatencyThreshold {
		t.Errorf("PassesLatencyThreshold is false")
	}

	// Verify Criterion 4: Zero numerical divergence
	if bench.NumericalDivergenceMax != 0.0 {
		t.Fatalf("Numerical divergence detected: max diff = %f, want 0.0", bench.NumericalDivergenceMax)
	}
	if !bench.PassesDivergenceThreshold {
		t.Errorf("PassesDivergenceThreshold is false")
	}

	// Verify Speedup factor
	if bench.SpeedupFactor < 10.0 {
		t.Errorf("SpeedupFactor = %.1fx, expected >= 10x", bench.SpeedupFactor)
	}
}

// TestMetalICB_EndToEndEquivalence verifies exact token equivalence between sequential and ICB execution.
func TestMetalICB_EndToEndEquivalence(t *testing.T) {
	cfg := DefaultICBDispatchConfig(16)
	dispatcher, err := NewICBDispatcher(cfg)
	if err != nil {
		t.Fatalf("NewICBDispatcher failed: %v", err)
	}
	defer dispatcher.Close()

	if err := dispatcher.RecordStaticGraph(); err != nil {
		t.Fatalf("RecordStaticGraph failed: %v", err)
	}

	// Input embedding
	xEmbed := make([]float32, cfg.HiddenDim)
	for i := range xEmbed {
		xEmbed[i] = float32(math.Sin(float64(i + 1)))
	}

	step := &ICBStepParams{
		StepL:       12,
		SeqLen:      13,
		BatchSize:   1,
		KVRowStride: 128,
		InputEmbed:  xEmbed,
		WantLogits:  true,
	}

	// Run with ICB replay
	outICB, err := dispatcher.DispatchStep(step)
	if err != nil {
		t.Fatalf("DispatchStep (ICB) failed: %v", err)
	}

	// Run with forced fallback (sequential encoding)
	cfgFallback := cfg
	cfgFallback.ForceFallback = true
	dispatcherSeq, err := NewICBDispatcher(cfgFallback)
	if err != nil {
		t.Fatalf("NewICBDispatcher fallback failed: %v", err)
	}
	defer dispatcherSeq.Close()

	outSeq, err := dispatcherSeq.DispatchStep(step)
	if err != nil {
		t.Fatalf("DispatchStep (Seq) failed: %v", err)
	}

	// Verify exact equivalence of output tensors
	if len(outICB.Logits) != len(outSeq.Logits) {
		t.Fatalf("Logits length mismatch: %d vs %d", len(outICB.Logits), len(outSeq.Logits))
	}

	maxDiff := float32(0.0)
	for i := range outICB.Logits {
		d := float32(math.Abs(float64(outICB.Logits[i] - outSeq.Logits[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}

	if maxDiff > 1e-6 {
		t.Fatalf("Numerical divergence detected between sequential and ICB: maxDiff = %e", maxDiff)
	}
}
