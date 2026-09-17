package ggufload

import (
	"testing"
)

// estimate_dense_streamed_moe_test.go - fak#13200. UnifiedHostResidencyPlan's
// streamedDenseBounded branch called EstimateQ4KLoadMemoryPlan, which refuses EVERY MoE
// checkpoint by name (cfg.IsMoE()); the caller then fell back to EstimateLoadMemoryPlan() - the
// raw full-payload charge - and the declared bounded dense working set was SILENTLY discarded.
// This leaf routes the MoE case to a dense-bound helper that charges the eligible dense Q4_K side
// at min(denseEligible, bound) as a host row while leaving every non-eligible tensor (including
// the routed-expert blobs it deliberately does not model) on the raw device charge.
//
// The fixture is a MoE qwen3moe header (the simplest arch in archUsesGGUFBatchedMoEExperts with a
// real expert_count axis), carrying:
//   - one ELIGIBLE dense Q4_K matmul (blk.0.ffn_down.weight -> mlp.down_proj.weight), and
//   - one non-Q4_K device dense tensor (token_embd.weight, F32 -> model.embed_tokens.weight),
// so the bounded host working set is unmistakably below the full dense side and the device
// remainder is non-zero and separable.

const (
	moeDenseEligibleBytes = int64(256 * 144) // one Q4_K super-block row -> 36864 B
	moeDeviceDenseBytes   = int64(256 * 4)   // token_embd F32 = 1024 B
)

// moeStreamedTestWeightSource is a qwen3moe MoE checkpoint: NumExperts>0 (so the dense-family
// Q4_K estimate refuses by name), one eligible dense Q4_K tensor, and one non-Q4_K device tensor.
func moeStreamedTestWeightSource(t *testing.T, eligibleQ4KBytes int64) *WeightSource {
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
			// ffn_down is the identity-normalized Q4_K matmul -> streamed-dense eligible.
			{Name: "blk.0.ffn_down.weight", Dims: []uint64{uint64(eligibleQ4KBytes / 144 * 256), 1}, Type: TensorQ4_K},
			// The batched routed experts are non-eligible (no canonical mapping); charged raw.
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 256, 8}, Type: TensorQ4_K},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// TestUnifiedHostResidencyPlanMoEBoundedDenseWorkingSet is the RED->GREEN witness. Before the fix
// the MoE checkpoint fell back to the raw full-payload plan (the declared bound discarded); after
// it the eligible dense Q4_K side is charged at the declared host working set while the
// non-eligible remainder stays fully charged (fail-closed preserved).
func TestUnifiedHostResidencyPlanMoEBoundedDenseWorkingSet(t *testing.T) {
	const bound = int64(8 * 1024) // 8 KiB bounded dense working set
	ws := moeStreamedTestWeightSource(t, moeDenseEligibleBytes)

	// RED premise: the bounded dense request on an MoE checkpoint does NOT charge the bound; the
	// current base falls back to the raw full-payload plan, which charges the WHOLE dense side
	// (plus the routed experts) to device. Assert the fallback does not produce the bound.
	fallback, err := ws.EstimateLoadMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateLoadMemoryPlan: %v", err)
	}
	if got := fallback.HostTotal(); got == bound {
		t.Fatalf("RED premise broken: the full-payload fallback already charges the bound %d", bound)
	}

	// GREEN: the unified plan charges the bounded dense host working set.
	plan, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(bound))
	if err != nil {
		t.Fatalf("UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet): %v", err)
	}
	if got := plan.HostTotal(); got != bound {
		t.Fatalf("bounded host total = %d, want the declared bound %d; plan=%+v", got, bound, plan)
	}
	byDetail := memoryPlanBytesByDetail(plan)
	if got := byDetail["gguf-host-dense-streamed"]; got != bound {
		t.Fatalf("gguf-host-dense-streamed = %d, want %d; plan=%+v", got, bound, plan)
	}

	// The non-eligible remainder (device dense + routed experts) stays fully charged: the plan
	// total falls by exactly the eligible dense side that moved to the bounded host row.
	wantDevice := fallback.DeviceTotal() - moeDenseEligibleBytes
	if got := plan.DeviceTotal(); got != wantDevice {
		t.Fatalf("DeviceTotal = %d, want non-eligible remainder %d (fallback %d - eligible %d)",
			got, wantDevice, fallback.DeviceTotal(), moeDenseEligibleBytes)
	}
	if got := plan.Total(); got != wantDevice+bound {
		t.Fatalf("plan total = %d, want %d", got, wantDevice+bound)
	}

	// Zero bound = stream-through: no dense host residency charged.
	through, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(0))
	if err != nil {
		t.Fatalf("stream-through MoE plan: %v", err)
	}
	if got := through.HostTotal(); got != 0 {
		t.Fatalf("stream-through MoE HostTotal = %d, want 0", got)
	}
	if got := through.DeviceTotal(); got != wantDevice {
		t.Fatalf("stream-through MoE DeviceTotal = %d, want %d", got, wantDevice)
	}

	// The bound is capped by the actual dense side, never inflated past what is resident.
	capped, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(moeDenseEligibleBytes * 4))
	if err != nil {
		t.Fatalf("over-bound MoE plan: %v", err)
	}
	if got := capped.HostTotal(); got != moeDenseEligibleBytes {
		t.Fatalf("capped HostTotal = %d, want the dense side %d", got, moeDenseEligibleBytes)
	}
}

// TestUnifiedHostResidencyPlanMoEBoundedDenseFailsClosed is the negative half: the non-eligible
// remainder is NEVER discarded, so a checkpoint whose non-eligible side alone exceeds the
// aperture still produces the full charge (the honest preserve of fail-closed).
func TestUnifiedHostResidencyPlanMoEBoundedDenseFailsClosed(t *testing.T) {
	const bound = int64(8 * 1024)
	ws := moeStreamedTestWeightSource(t, moeDenseEligibleBytes)

	plan, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(bound))
	if err != nil {
		t.Fatalf("UnifiedHostResidencyPlan: %v", err)
	}
	// The routed-expert blob (256*256*8 elems / 256 * 144 = 294912 B) and the F32 token_embd
	// (1024 B) must both still be charged: the bounded dense policy never hides them.
	wantDevice := int64(256*256*8/256*144) + moeDeviceDenseBytes
	if got := plan.DeviceTotal(); got != wantDevice {
		t.Fatalf("DeviceTotal = %d, want the full non-eligible charge %d", got, wantDevice)
	}
	if plan.Total() <= bound {
		t.Fatalf("plan total %d collapsed to the bound; the non-eligible remainder was discarded", plan.Total())
	}

	// A negative bound is refused by name, never silently clamped.
	if _, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(-1)); err == nil {
		t.Fatal("negative streamed-dense working set: want a named refusal, got nil")
	}
}

// --- fak#13201: the bounded-dense fold must cover the NON-Q4_K dense k-quants the pinned V4.1
// Q2_K artifact actually carries, not only TensorQ4_K. The fixture below is a qwen3moe MoE whose
// eligible dense matmul is a Q2_K tensor (blk.0.ffn_down.weight -> mlp.down_proj.weight), which
// the pre-fix Q4_K-only predicate left at full raw device charge. Q2_K packs 256 weights per
// 72-byte super-block (blockQ2KBytes), so the eligible dense payload is derived with the same
// tensorPayloadBytes helper the estimator uses rather than a hardcoded byte count.

// moeNonQ4KDenseTestWeightSource is a qwen3moe MoE checkpoint carrying ONE eligible non-Q4_K
// dense k-quant (Q2_K ffn_down) plus a non-eligible routed-expert blob and a non-eligible F32
// embedding. denseTensors is the element count of the Q2_K ffn_down weight (a multiple of qkK).
func moeNonQ4KDenseTestWeightSource(t *testing.T, denseElems uint64) *WeightSource {
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
			// ffn_down is the identity-normalized dense matmul: a NON-Q4_K k-quant, so the
			// pre-fix Q4_K-only gate charged it at full raw device payload.
			{Name: "blk.0.ffn_down.weight", Dims: []uint64{denseElems, 1}, Type: TensorQ2_K},
			// The batched routed experts are non-eligible (no canonical mapping); charged raw.
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 256, 8}, Type: TensorQ4_K},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// TestUnifiedHostResidencyPlanMoEBoundedDenseNonQ4K is the fak#13201 RED->GREEN witness: an
// eligible NON-Q4_K dense k-quant (Q2_K) is folded into the bounded host row exactly as the
// Q4_K case already was, while the non-eligible remainder (the F32 embedding and the
// routed-expert blobs) stays fully charged.
func TestUnifiedHostResidencyPlanMoEBoundedDenseNonQ4K(t *testing.T) {
	const bound = int64(8 * 1024) // 8 KiB bounded dense working set
	// 256 Q2_K super-blocks = 65536 weights -> 256*blockQ2KBytes (18432 B), deliberately LARGER
	// than the bound so the charged row is min(eligible, bound) == bound and the over-bound cap is
	// exercised separately below.
	denseElems := uint64(qkK * 256)
	ws := moeNonQ4KDenseTestWeightSource(t, denseElems)

	// Derive the expected eligible/device bytes from the estimator's own helpers.
	eligible, err := tensorPayloadBytes(TensorInfo{Name: "blk.0.ffn_down.weight", Dims: []uint64{denseElems, 1}, Type: TensorQ2_K})
	if err != nil {
		t.Fatalf("tensorPayloadBytes(eligible): %v", err)
	}
	if want := uint64(256) * blockQ2KBytes; eligible != want {
		t.Fatalf("fixture geometry: eligible Q2_K payload = %d, want %d", eligible, want)
	}
	if eligible <= uint64(bound) {
		t.Fatalf("fixture geometry: eligible %d must exceed the bound %d", eligible, bound)
	}

	// RED premise (a): the un-bounded base charges the WHOLE dense side (Q2_K included) to
	// device at full raw payload, and its host total is not the declared bound.
	base, err := ws.EstimateLoadMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateLoadMemoryPlan: %v", err)
	}
	if got := base.HostTotal(); got == bound {
		t.Fatalf("RED premise broken: base HostTotal already charges the bound %d", bound)
	}
	if base.DeviceTotal() < int64(eligible) {
		t.Fatalf("RED premise broken: base DeviceTotal %d does not include the eligible %d", base.DeviceTotal(), eligible)
	}

	// GREEN: the bounded plan folds the eligible Q2_K dense side into min(eligible, bound).
	plan, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(bound))
	if err != nil {
		t.Fatalf("UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet): %v", err)
	}
	byDetail := memoryPlanBytesByDetail(plan)
	if got := byDetail["gguf-host-dense-streamed"]; got != bound {
		t.Fatalf("gguf-host-dense-streamed = %d, want min(eligible=%d, bound=%d)=%d; plan=%+v",
			got, eligible, bound, bound, plan)
	}
	wantDevice := base.DeviceTotal() - int64(eligible)
	if got := plan.DeviceTotal(); got != wantDevice {
		t.Fatalf("DeviceTotal = %d, want base %d minus eligible %d = %d", got, base.DeviceTotal(), eligible, wantDevice)
	}

	// (c) Zero bound = stream-through: no dense host residency, eligible side still off device.
	through, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(0))
	if err != nil {
		t.Fatalf("stream-through plan: %v", err)
	}
	if got := through.HostTotal(); got != 0 {
		t.Fatalf("stream-through HostTotal = %d, want 0", got)
	}
	if got := through.DeviceTotal(); got != wantDevice {
		t.Fatalf("stream-through DeviceTotal = %d, want %d", got, wantDevice)
	}

	// (d) Over-bound is capped at the eligible side, never inflated.
	capped, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(int64(eligible) * 4))
	if err != nil {
		t.Fatalf("over-bound plan: %v", err)
	}
	if got := capped.HostTotal(); got != int64(eligible) {
		t.Fatalf("capped HostTotal = %d, want the eligible dense side %d", got, eligible)
	}
}

// TestUnifiedHostResidencyPlanMoEBoundedDenseNonQ4KFailsClosed is the fak#13201 negative half:
// a non-eligible / no-canonical-mapping tensor (the routed-expert blob) is NEVER folded into the
// bounded host row, and a negative bound is refused by name.
func TestUnifiedHostResidencyPlanMoEBoundedDenseNonQ4KFailsClosed(t *testing.T) {
	const bound = int64(8 * 1024)
	ws := moeNonQ4KDenseTestWeightSource(t, uint64(qkK*256))

	plan, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(bound))
	if err != nil {
		t.Fatalf("UnifiedHostResidencyPlan: %v", err)
	}
	// The routed-expert blob (256*256*8 elems / 256 * 144 = 294912 B) and the F32 token_embd
	// (1024 B) must still be charged: the bounded dense policy never hides them.
	expertBytes, err := tensorPayloadBytes(TensorInfo{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 256, 8}, Type: TensorQ4_K})
	if err != nil {
		t.Fatalf("tensorPayloadBytes(expert): %v", err)
	}
	wantDevice := int64(expertBytes) + moeDeviceDenseBytes
	if got := plan.DeviceTotal(); got != wantDevice {
		t.Fatalf("DeviceTotal = %d, want the full non-eligible charge %d", got, wantDevice)
	}
	if byDetail := memoryPlanBytesByDetail(plan); byDetail["gguf-host-dense-streamed"] != bound {
		t.Fatalf("gguf-host-dense-streamed = %d, want %d", byDetail["gguf-host-dense-streamed"], bound)
	}
	if plan.Total() <= bound {
		t.Fatalf("plan total %d collapsed to the bound; the non-eligible remainder was discarded", plan.Total())
	}

	// A negative bound is refused by name, never silently clamped.
	if _, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(-1)); err == nil {
		t.Fatal("negative streamed-dense working set: want a named refusal, got nil")
	}
}
