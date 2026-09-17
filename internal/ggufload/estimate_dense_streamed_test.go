package ggufload

import (
	"strings"
	"testing"
)

// estimate_dense_streamed_test.go — fak#13194. The dense sibling of the #13121 routed-expert
// bound. EstimateQ4KLoadMemoryPlan refused ANY streamedDenseQ4K request by name
// (ErrQ4KLoadEstimateUnsupported), so UnifiedHostResidencyPlan's streamed-dense branch fell back
// to EstimateLoadMemoryPlan() — the raw full payload — and the dense side was always charged at
// its FULL on-disk size. With the pinned V4.1 Q2_K artifact the device-scoped dense remainder is
// 63.099 GiB against a 62.425 GiB Halo MemTotal, so the full-charge plan can never pass. This
// leaf adds a bounded dense host working set to the streamed-dense option and charges that bound,
// exactly as the routed-expert streamed policy does, while keeping the genuine-oversize refusal.

// denseStreamedTestWeightSource is a dense Llama whose Q4_K matmul tensor carries a
// parameterized dense payload, so the bounded working set is unmistakably below the full dense
// side. token_embd is F32 (non-Q4_K, never streamed-dense eligible) so the device dense side is
// non-zero and byte-stable across the full and bounded arms.
func denseStreamedTestWeightSource(t *testing.T, denseQ4KBytes uint64) *WeightSource {
	t.Helper()
	f := &File{
		Metadata: map[string]Value{
			"general.architecture":                   {Type: TypeString, Value: "llama"},
			"llama.embedding_length":                 {Type: TypeUint32, Value: uint32(256)},
			"llama.block_count":                      {Type: TypeUint32, Value: uint32(1)},
			"llama.attention.head_count":             {Type: TypeUint32, Value: uint32(1)},
			"llama.attention.head_count_kv":          {Type: TypeUint32, Value: uint32(1)},
			"llama.feed_forward_length":              {Type: TypeUint32, Value: uint32(256)},
			"llama.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},
		},
		Tensors: []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256}, Type: TensorF32}, // device dense, 1024 B
			// ffn_down is the identity-normalized Q4_K matmul -> streamed-dense eligible.
			{Name: "blk.0.ffn_down.weight", Dims: []uint64{denseQ4KBytes / 144 * 256, 1}, Type: TensorQ4_K},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// TestEstimateQ4KStreamedDenseWorkingSet witnesses the leaf's core charge claim: a bounded dense
// working set is charged in place of the full dense side, while the device dense remainder is
// byte-stable. The negative half asserts the fail-closed direction: a negative bound is refused
// by name, and an unbounded streamed-dense request is still refused by name (no silent full
// charge).
func TestEstimateQ4KStreamedDenseWorkingSet(t *testing.T) {
	const deviceDense = int64(256 * 4) // token_embd F32 = 1024 B (non-Q4_K, never streamed-dense)
	const denseFull = int64(256 * 144) // one Q4_K super-block row -> 36864 B of dense Q4_K
	const resident = int64(8 * 1024)   // 8 KiB bounded dense working set

	ws := denseStreamedTestWeightSource(t, uint64(denseFull))

	// Baseline: with no bound declared, the streamed-dense request is refused by name (the
	// historical behavior this leaf supersedes) — the caller cannot silently get a full charge.
	_, err := ws.EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4K(true))
	if err == nil || !strings.Contains(err.Error(), "streaming") {
		t.Fatalf("unbounded streamed-dense estimate error = %v, want a named streaming refusal", err)
	}

	// The fix: a bounded dense working set is charged, not the full dense side.
	plan, err := ws.EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4KWorkingSet(resident))
	if err != nil {
		t.Fatalf("EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4KWorkingSet): %v", err)
	}
	byDetail := memoryPlanBytesByDetail(plan)
	if got := byDetail["gguf-host-dense-streamed"]; got != resident {
		t.Fatalf("bounded dense working set = %d, want %d (not the full dense side %d); plan=%+v", got, resident, denseFull, plan)
	}
	if got := plan.HostTotal(); got != resident {
		t.Fatalf("bounded HostTotal = %d, want %d", got, resident)
	}
	if got := plan.DeviceTotal(); got != deviceDense {
		t.Fatalf("bounded DeviceTotal = %d, want the unchanged device dense side %d", got, deviceDense)
	}

	// Comparison that names the leaf: the host-scoped dense demand collapses from the full dense
	// side to the declared bound. Mirrors the physical 63.099 GiB vs 62.425 GiB case.
	if plan.HostTotal() >= denseFull {
		t.Fatalf("bounded host total %d did not fall below the full dense side %d", plan.HostTotal(), denseFull)
	}

	// Stream-through: a zero working set charges NO dense host residency, keeping the device side.
	through, err := ws.EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4KWorkingSet(0))
	if err != nil {
		t.Fatalf("stream-through dense plan: %v", err)
	}
	if got := through.HostTotal(); got != 0 {
		t.Fatalf("stream-through dense HostTotal = %d, want 0", got)
	}
	if got := through.DeviceTotal(); got != deviceDense {
		t.Fatalf("stream-through dense DeviceTotal = %d, want %d", got, deviceDense)
	}

	// Negative bound: refused by name, never silently clamped into an unrepresentable policy.
	_, err = ws.EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4KWorkingSet(-1))
	if err == nil || !strings.Contains(err.Error(), "negative") {
		t.Fatalf("negative dense working-set error = %v, want a named refusal", err)
	}
}
