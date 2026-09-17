package ggufload

import (
	"bytes"
	"testing"
)

// estimate_dense_streamed_kquant_test.go - fak#13201. The fak#13200 MoE bounded-dense fold
// charged only ELIGIBLE Q4_K tensors into the bounded host working set; the pinned vcruz
// DeepSeek-V4.1 Q2_K artifact's device-scoped dense remainder is Q2_K/Q3_K/Q5_K/Q6_K, so the
// declared bound was silently discarded for the exact artifact it was written for. This leaf
// widens the bounded route (estimate fold AND loader dispatch) to the dense k-quant types the
// artifact actually carries, through the SAME shared eligibility predicate, while preserving the
// fail-closed direction: a tensor that is not eligible (routed-expert blobs, non-k-quant types,
// or a genuinely unfittable remainder) stays at its full raw charge.

// q2kDenseEligibleBytes is one Q2_K super-block row (blockQ2KBytes = 84) -> 84 B for a
// [256,1] tensor whose single reduction block is 256 weights.
const q2kDenseEligibleBytes = int64(blockQ2KBytes)

// moeKQuantStreamedTestWeightSource is the qwen3moe MoE fixture with a NON-Q4_K dense eligible
// matmul (blk.0.ffn_down.weight, Q2_K) plus a non-eligible F32 device tensor and a routed-expert
// blob, so the bounded host working set is separable from the full dense side and the device
// remainder is non-zero.
func moeKQuantStreamedTestWeightSource(t *testing.T, eligibleQ2KBytes int64) *WeightSource {
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
			// ffn_down is the identity-normalized dense matmul -> bounded-dense eligible, Q2_K.
			{Name: "blk.0.ffn_down.weight", Dims: []uint64{uint64(eligibleQ2KBytes / blockQ2KBytes * 256), 1}, Type: TensorQ2_K},
			// The batched routed experts have no canonical mapping; charged raw.
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 256, 8}, Type: TensorQ4_K},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// TestUnifiedHostResidencyPlanMoEBoundedDenseNonQ4KKQuant is the RED->GREEN witness: a NON-Q4_K
// eligible dense k-quant on an MoE checkpoint is folded into the bounded host row, and the
// non-eligible remainder stays fully charged.
func TestUnifiedHostResidencyPlanMoEBoundedDenseNonQ4KKQuant(t *testing.T) {
	const bound = int64(64) // 64 B bounded dense working set (< one Q2_K row of 84 B)
	ws := moeKQuantStreamedTestWeightSource(t, q2kDenseEligibleBytes)

	fallback, err := ws.EstimateLoadMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateLoadMemoryPlan: %v", err)
	}
	if got := fallback.HostTotal(); got == bound {
		t.Fatalf("RED premise broken: the full-payload fallback already charges the bound %d", bound)
	}

	plan, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(bound))
	if err != nil {
		t.Fatalf("UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet): %v", err)
	}
	if got := plan.HostTotal(); got != bound {
		t.Fatalf("bounded host total = %d, want the declared bound %d; plan=%+v", got, bound, plan)
	}
	if got := memoryPlanBytesByDetail(plan)["gguf-host-dense-streamed"]; got != bound {
		t.Fatalf("gguf-host-dense-streamed = %d, want %d", got, bound)
	}

	// The non-eligible remainder stays fully charged: the device total falls by exactly the
	// eligible non-Q4_K dense side that moved to the bounded host row.
	wantDevice := fallback.DeviceTotal() - q2kDenseEligibleBytes
	if got := plan.DeviceTotal(); got != wantDevice {
		t.Fatalf("DeviceTotal = %d, want non-eligible remainder %d (fallback %d - eligible %d)",
			got, wantDevice, fallback.DeviceTotal(), q2kDenseEligibleBytes)
	}
	if got := plan.Total(); got != wantDevice+bound {
		t.Fatalf("plan total = %d, want %d", got, wantDevice+bound)
	}

	// The bound is capped by the actual dense side, never inflated past what is resident.
	capped, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(q2kDenseEligibleBytes * 4))
	if err != nil {
		t.Fatalf("over-bound MoE plan: %v", err)
	}
	if got := capped.HostTotal(); got != q2kDenseEligibleBytes {
		t.Fatalf("capped HostTotal = %d, want the dense side %d", got, q2kDenseEligibleBytes)
	}
}

// TestUnifiedHostResidencyPlanMoEBoundedDenseNonQ4KFailsClosed is the negative half: the
// non-eligible remainder (routed experts + F32 token embeddings) is never discarded.
func TestUnifiedHostResidencyPlanMoEBoundedDenseNonQ4KFailsClosed(t *testing.T) {
	const bound = int64(64)
	ws := moeKQuantStreamedTestWeightSource(t, q2kDenseEligibleBytes)

	plan, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(bound))
	if err != nil {
		t.Fatalf("UnifiedHostResidencyPlan: %v", err)
	}
	wantDevice := int64(256*256*8/256*blockQ4KBytes) + int64(256*4)
	if got := plan.DeviceTotal(); got != wantDevice {
		t.Fatalf("DeviceTotal = %d, want the full non-eligible charge %d", got, wantDevice)
	}
	if plan.Total() <= bound {
		t.Fatalf("plan total %d collapsed to the bound; the non-eligible remainder was discarded", plan.Total())
	}
}

// TestQwen38StreamedDenseQ2KLoaderBuildsWithoutQ2KPayloadRead is the LOADER-RETENTION half of the
// fak#13201 witness. The estimate-side tests above prove the MoE fold CHARGES a non-Q4_K dense
// k-quant into the bounded host row, but a charge is only real if the loader actually RETRACTS the
// payload instead of reading it. This binds the two: the SAME eligible tensor class (Q2_K,
// blk.0.ffn_down.weight -> model.layers.0.mlp.down_proj.weight) is built through the REAL loader
// entry point under WithStreamedDenseQ4K(true), and the reader must see ZERO reads overlapping the
// Q2_K payload while the built model holds the tensor as a checkpoint-backed (lazy) descriptor.
// Modelled on TestQwen38StreamedDenseQ4KLoaderBuildsWithoutQ4KPayloadRead, which is the Q4_K arm of
// the same contract.
func TestQwen38StreamedDenseQ2KLoaderBuildsWithoutQ2KPayloadRead(t *testing.T) {
	const tensorName = "blk.0.ffn_down.weight"
	q2kPayload := make([]byte, blockQ2KBytes)
	f32Payload := make([]byte, 256*4)
	blob := append(append([]byte(nil), q2kPayload...), f32Payload...)
	reader := &rangeReadCounter{r: bytes.NewReader(blob), lo: 0, hi: int64(len(q2kPayload))}
	meta := map[string]Value{
		"general.architecture":                   {Type: TypeString, Value: "qwen2"},
		"general.name":                           {Type: TypeString, Value: "Qwen3.8-27B-Q2_K_M"},
		"qwen2.embedding_length":                 {Type: TypeUint32, Value: uint32(256)},
		"qwen2.block_count":                      {Type: TypeUint32, Value: uint32(0)},
		"qwen2.attention.head_count":             {Type: TypeUint32, Value: uint32(1)},
		"qwen2.attention.head_count_kv":          {Type: TypeUint32, Value: uint32(1)},
		"qwen2.feed_forward_length":              {Type: TypeUint32, Value: uint32(256)},
		"qwen2.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},
	}
	file := &File{Metadata: meta, Tensors: []TensorInfo{
		{Name: tensorName, Type: TensorQ2_K, Dims: []uint64{256, 1}},
		{Name: "blk.0.attn_v.weight", Type: TensorF32, Dims: []uint64{256, 1}, FileOffset: int64(len(q2kPayload))},
	}}
	ws, err := NewWeightSource(file, reader, int64(len(blob)))
	if err != nil {
		t.Fatal(err)
	}
	m, err := ws.QuantModelQ4KProfileOptions(nil, WithStreamedDenseQ4K(true))
	if err != nil {
		t.Fatal(err)
	}
	if got := reader.overlaps.Load(); got != 0 {
		t.Fatalf("streamed model build made %d reads overlapping the Q2_K payload, want zero", got)
	}
	if got := m.Cfg.Name; got != "Qwen3.8-27B-Q2_K_M" {
		t.Fatalf("checkpoint identity = %q", got)
	}
	if !m.KQuantLazy("model.layers.0.mlp.down_proj.weight") {
		t.Fatal("exact-model dense Q2_K tensor was not checkpoint-backed")
	}
}
