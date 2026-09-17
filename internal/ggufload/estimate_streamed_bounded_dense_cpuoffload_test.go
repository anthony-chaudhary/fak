package ggufload

import (
	"strings"
	"testing"
)

// estimate_streamed_bounded_dense_cpuoffload_test.go - fak#13215. The DeepSeek-V4.1 Q2_K serve on a
// physical strix3 reaches EstimateCPUOffloadExpertsStreamedMemoryPlan (denseResident = -1) through
// serveStreamedCPUOffloadPlanForAperture, which pins the historical FULL device dense charge. The
// staged 63.22 GiB dense side is then charged whole as ONE host staging transit and the guard refuses
// FitTooBig against the 48.76 GiB host window, even though the loader would hold the eligible dense
// k-quant side as a bounded host working set (fak#13209). The combined estimator threads BOTH bounds
// through the ONE shared kernel, so a streamed arm that also declares a bounded dense working set
// charges min(denseEligible, denseResident) instead of the whole dense side.
//
// The fixture is the qwen3moe header estimate_dense_streamed_cpuoffload_test.go uses: one ELIGIBLE
// dense Q4_K matmul (blk.0.ffn_down.weight) and one non-eligible device tensor (token_embd.weight,
// F32). It differs from the bounded-dense fixture only in the routed-expert encoding: this exercise
// needs a real HOST-scoped routed blob the streamed fold can pull out of the raw group, so the
// batched expert is Q4_K (routedExpertResidencyEncoding) rather than F32, which the streamed arm
// would refuse by name.

const (
	streamedDenseEligibleBytes = int64(256 * 256 / 256 * 144) // one 256x256 Q4_K matmul -> 36864 B
	streamedRoutedExpertBytes  = int64(256 * 256 * 8 / 256 * 144)
)

// streamedBoundedDenseTestWeightSource is the combined-policy fixture: a MoE header with one eligible
// dense Q4_K matmul, one non-eligible F32 device tensor, and one batched Q4_K routed-expert blob
// (host-scoped, no canonical mapping => non-eligible) so the streamed and bounded-dense folds can
// BOTH be observed in one plan.
func streamedBoundedDenseTestWeightSource(t *testing.T) *WeightSource {
	t.Helper()
	f := &File{
		Metadata: map[string]Value{
			"general.architecture":                      {Type: TypeString, Value: "qwen3moe"},
			"qwen3moe.embedding_length":                 {Type: TypeUint32, Value: uint32(256)},
			"qwen3moe.block_count":                      {Type: TypeUint32, Value: uint32(1)},
			"qwen3moe.attention.head_count":             {Type: TypeUint32, Value: uint32(1)},
			"qwen3moe.feed_forward_length":              {Type: TypeUint32, Value: uint32(256)},
			"qwen3moe.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},
			"qwen3moe.expert_count":                     {Type: TypeUint32, Value: uint32(8)},
			"qwen3moe.expert_used_count":                {Type: TypeUint32, Value: uint32(2)},
			"qwen3moe.expert_feed_forward_length":       {Type: TypeUint32, Value: uint32(256)},
		},
		Tensors: []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256}, Type: TensorF32}, // device dense, 1024 B
			// ffn_down is the identity-normalized Q4_K matmul -> streamed-dense eligible. A wide row
			// (256x256 Q4_K -> 36864 B) so the eligible dense side is unmistakably larger than the
			// declared bound and separable from the F32 remainder.
			{Name: "blk.0.ffn_down.weight", Dims: []uint64{256, 256}, Type: TensorQ4_K},
			// The batched routed experts are non-eligible (no canonical mapping); a Q4_K encoding is
			// admitted raw by routedExpertResidencyEncoding so the streamed fold can charge them.
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 256, 8}, Type: TensorQ4_K},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// TestEstimateCPUOffloadExpertsStreamedBoundedDenseEmitsBothRows is the RED->GREEN witness: the
// combined estimator emits BOTH the streamed routed-expert row (min(hostRouted, streamResident)) and
// the bounded dense row (min(denseEligible, denseResident)), removes the eligible dense side from the
// device charge, and with denseResident=-1 emits NO dense row (the historical streamed arm preserved
// byte-for-byte).
func TestEstimateCPUOffloadExpertsStreamedBoundedDenseEmitsBothRows(t *testing.T) {
	const (
		streamResident = int64(64 << 10) // 64 KiB bounded routed working set, below the routed blob
		denseResident  = int64(8 << 10)  // 8 KiB bounded dense working set
	)
	ws := streamedBoundedDenseTestWeightSource(t)

	// The historical streamed arm (denseResident = -1) charges the WHOLE routed set host-side, keeps
	// the eligible dense side on the DEVICE, and emits NO bounded dense row.
	legacy, err := ws.EstimateCPUOffloadExpertsStreamedMemoryPlan(streamResident)
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsStreamedMemoryPlan: %v", err)
	}
	legacyBy := memoryPlanBytesByDetail(legacy)
	if _, ok := legacyBy["gguf-host-dense-streamed"]; ok {
		t.Fatalf("legacy streamed plan already carries a bounded dense row; the RED premise is broken")
	}
	if got := legacyBy["gguf-host-expert-offload-streamed"]; got != streamResident {
		t.Fatalf("legacy streamed resident row = %d, want %d; plan=%+v", got, streamResident, legacy)
	}
	// The streamed arm (denseResident = -1) keeps the eligible dense side on the DEVICE: the only
	// device rows are that eligible Q4_K and the F32 token_embd.
	if got := legacy.DeviceTotal(); got != streamedDenseEligibleBytes+1024 {
		t.Fatalf("legacy DeviceTotal = %d, want the full eligible dense %d + F32 remainder 1024 = %d (the eligible side is NOT bounded)", got, streamedDenseEligibleBytes, streamedDenseEligibleBytes+1024)
	}

	// GREEN: BOTH policies fold in one plan.
	plan, err := ws.EstimateCPUOffloadExpertsStreamedBoundedDenseMemoryPlan(1, streamResident, denseResident)
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsStreamedBoundedDenseMemoryPlan: %v", err)
	}
	byDetail := memoryPlanBytesByDetail(plan)
	if got := byDetail["gguf-host-expert-offload-streamed"]; got != streamResident {
		t.Fatalf("gguf-host-expert-offload-streamed = %d, want min(hostRouted, %d) = %d; plan=%+v", got, streamResident, streamResident, plan)
	}
	if got := byDetail["gguf-host-dense-streamed"]; got != denseResident {
		t.Fatalf("gguf-host-dense-streamed = %d, want min(denseEligible, %d) = %d; plan=%+v", got, denseResident, denseResident, plan)
	}
	// The eligible dense side moved OUT of the device charge: DeviceTotal is only the F32 remainder.
	if got := plan.DeviceTotal(); got != 1024 {
		t.Fatalf("combined DeviceTotal = %d, want the non-eligible F32 remainder 1024", got)
	}
	if got := plan.HostTotal(); got != streamResident+denseResident {
		t.Fatalf("combined HostTotal = %d, want both bounded rows summed %d", got, streamResident+denseResident)
	}

	// denseResident = -1 emits NO dense row: the historical streamed arm is preserved byte-for-byte.
	streamOnly, err := ws.EstimateCPUOffloadExpertsStreamedMemoryPlan(streamResident)
	if err != nil {
		t.Fatalf("stream-only plan: %v", err)
	}
	if len(streamOnly) != len(legacy) {
		t.Fatalf("stream-only plan has %d rows, want the legacy %d", len(streamOnly), len(legacy))
	}
	for i := range streamOnly {
		if streamOnly[i] != legacy[i] {
			t.Fatalf("stream-only row %d: %+v != legacy %+v; the no-dense-bound path must be byte-identical", i, streamOnly[i], legacy[i])
		}
	}
	if _, ok := memoryPlanBytesByDetail(streamOnly)["gguf-host-dense-streamed"]; ok {
		t.Fatalf("denseResident=-1 emitted a bounded dense row")
	}
}

// TestEstimateCPUOffloadExpertsStreamedBoundedDenseRefusesNegative fails-CLOSED on either negative
// bound: an unrepresentable policy is refused by name, never silently clamped.
func TestEstimateCPUOffloadExpertsStreamedBoundedDenseRefusesNegative(t *testing.T) {
	ws := streamedBoundedDenseTestWeightSource(t)
	_, err := ws.EstimateCPUOffloadExpertsStreamedBoundedDenseMemoryPlan(1, -1, 0)
	if err == nil || !strings.Contains(err.Error(), "streamed expert resident budget -1 is negative") {
		t.Fatalf("negative stream bound error = %v, want the named refusal", err)
	}
	_, err = ws.EstimateCPUOffloadExpertsStreamedBoundedDenseMemoryPlan(1, 0, -1)
	if err == nil || !strings.Contains(err.Error(), "streamed-dense host working set -1 is negative") {
		t.Fatalf("negative dense bound error = %v, want the named refusal", err)
	}
}
