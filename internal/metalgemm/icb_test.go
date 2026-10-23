package metalgemm

import (
	"encoding/json"
	"errors"
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

// TestICBReceiptEvidenceClass checks the standalone diagnostic API, not GPU parity.
// fak-test:runtime fast est=10ms lane=default
func TestICBReceiptEvidenceClass(t *testing.T) {
	cfg := DefaultICBDispatchConfig(2)
	d, err := NewICBDispatcher(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if err := d.RecordStaticGraph(); err != nil {
		t.Fatal(err)
	}
	out, err := d.DispatchStep(&ICBStepParams{StepL: 1, SeqLen: 2, BatchSize: 1, KVRowStride: 8, WantLogits: true})
	var unavailable *ICBUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("expected typed unavailable error, got %v", err)
	}
	if out == nil || out.Receipt.EvidenceClass != ICBEvidenceUnavailable || out.Receipt.UnavailableReason == "" {
		t.Fatalf("missing unavailable receipt: %+v", out)
	}
	if out.ICBUsed || out.Receipt.ICBUsed || out.Receipt.CommandBuffers != 0 || out.Receipt.Encoders != 0 || out.Receipt.ICBDispatches != 0 {
		t.Fatalf("unavailable result claimed GPU execution: %+v", out.Receipt)
	}
	if out.Receipt.HostEncodeMs != 0 || out.Receipt.HostWaitMs != 0 || out.Receipt.GPUMs != 0 || out.Receipt.TotalMs != 0 {
		t.Fatalf("unavailable result fabricated timing: %+v", out.Receipt)
	}
	if len(out.Logits)+len(out.NewK)+len(out.NewV)+len(out.LastPre) != 0 {
		t.Fatal("unavailable result fabricated tensors")
	}
	if out.Receipt.StepL != 1 {
		t.Fatal("step metadata was lost")
	}
	if !d.IsAvailable() && out.FallbackReason != FallbackReasonDeviceIneligible {
		t.Fatal("missing device fallback reason")
	}
	bench, err := d.Benchmark(20, 64)
	if !errors.As(err, &unavailable) {
		t.Fatalf("benchmark expected typed unavailable error, got %v", err)
	}
	if bench == nil || bench.EvidenceClass != ICBEvidenceUnavailable || bench.UnavailableReason == "" || bench.TopologyLayers != 64 || bench.DispatchesPerStep != 962 {
		t.Fatalf("missing benchmark diagnostic metadata: %+v", bench)
	}
	if bench.PassesLatencyThreshold || bench.PassesDivergenceThreshold || bench.SequentialHostEncodeMs != 0 || bench.ICBHostEncodeMs != 0 || bench.SpeedupFactor != 0 || bench.LatencyReductionMs != 0 || bench.LatencyReductionPercent != 0 || bench.NumericalDivergenceMax != 0 {
		t.Fatalf("unavailable benchmark claimed acceptance: %+v", bench)
	}
	// Neither legacy/zero-value nor relabeled standalone data is a hardware witness.
	for _, class := range []ICBEvidenceClass{"", ICBEvidenceSynthetic, ICBEvidenceUnavailable, "measured"} {
		r := out.Receipt
		r.EvidenceClass = class
		r.GPUMs = 3.2
		if r.ValidateMeasuredEvidence() == nil {
			t.Fatalf("accepted %q receipt", class)
		}
		b := *bench
		b.EvidenceClass = class
		b.PassesLatencyThreshold, b.PassesDivergenceThreshold = true, true
		if b.ValidateMeasuredEvidence() == nil {
			t.Fatalf("accepted %q benchmark", class)
		}
	}
	data, err := json.Marshal(out.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ICBDispatchReceipt
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.EvidenceClass != ICBEvidenceUnavailable || decoded.ValidateMeasuredEvidence() == nil {
		t.Fatal("JSON round trip lost evidence rejection")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestMetalICB_FallbackVariableBatch(t *testing.T) {
	for _, tc := range []struct {
		name            string
		batch           int
		prefill, forced bool
		want            ICBDispatchFallbackReason
	}{
		{"batch", 4, false, false, FallbackReasonBatchSize},
		{"prefill", 1, true, false, FallbackReasonPrefill},
		{"forced", 1, false, true, FallbackReasonForced},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultICBDispatchConfig(2)
			cfg.ForceFallback = tc.forced
			d, err := NewICBDispatcher(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			out, err := d.DispatchStep(&ICBStepParams{BatchSize: tc.batch, DynamicPrefill: tc.prefill, KVRowStride: 8})
			var unavailable *ICBUnavailableError
			if !errors.As(err, &unavailable) || out == nil {
				t.Fatalf("expected unavailable output, got %+v, %v", out, err)
			}
			if out.ICBUsed || out.FallbackReason != tc.want || out.Receipt.FallbackReason != tc.want {
				t.Fatalf("incorrect fallback metadata: %+v", out)
			}
		})
	}
}
