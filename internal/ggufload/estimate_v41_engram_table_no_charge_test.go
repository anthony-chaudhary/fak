package ggufload

import (
	"testing"
)

// estimate_v41_engram_table_no_charge_test.go — fak-private#13290. The V4.1 packed
// Engram table (blk.<L>.engram_embd.weight, Q2_K, [256, ~384M rows] = 32.26 GB on the
// pinned vcruz Q2_K checkpoint) is read row-wise through the bounded
// model.V41EngramRowSource seam; the materializing loader DROPS it from the load
// (computeQ4KTensorWork). Before this fix the byte-accounting estimators did NOT drop
// it, so the streamed serve plan charged ~30 GiB of phantom device dense demand that
// the loader never allocates — the dominant term in the witnessed strix3 plan-total
// refusal (105.4 GiB vs the 62.4 GiB pool).

// v41EngramTableEstimateHeader is a deepseek41 header carrying one packed Engram table
// plus a dense non-engram F32 tensor that MUST stay charged (fail-closed): the drop is
// scoped to the engram table only.
func v41EngramTableEstimateHeader() *File {
	const H = 5120
	return &File{
		Metadata: ds41Meta("deepseek41"),
		Tensors: []TensorInfo{
			// The packed Engram table: [qkK=256, rows] Q2_K. Rows chosen small enough to
			// keep the fixture cheap; the predicate is name+type based, not size based.
			{Name: "blk.1.engram_embd.weight", Dims: []uint64{256, 4096}, Type: TensorQ2_K},
			// A real dense device tensor that is NOT an engram table: stays charged.
			{Name: "blk.0.attn_q_a.weight", Dims: []uint64{1280, H}, Type: TensorQ2_K},
			{Name: "blk.0.ffn_norm.weight", Dims: []uint64{H}, Type: TensorF32},
		},
	}
}

// TestV41EngramTableNotChargedInStreamedPlan is the RED->GREEN witness: adding the
// packed Engram table to an otherwise identical header must change the serve plan by
// NOTHING — the table is dropped from byte-accounting, exactly as the loader drops it.
func TestV41EngramTableNotChargedInStreamedPlan(t *testing.T) {
	withTable := v41EngramTableEstimateHeader()
	without := v41EngramTableEstimateHeader()
	var kept []TensorInfo
	for _, ti := range without.Tensors {
		if ti.Name == "blk.1.engram_embd.weight" {
			continue
		}
		kept = append(kept, ti)
	}
	without.Tensors = kept

	wsA, err := NewWeightSource(withTable, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource(A): %v", err)
	}
	info, ok := wsA.Tensor("blk.1.engram_embd.weight")
	if !ok {
		t.Fatal("fixture is missing blk.1.engram_embd.weight")
	}
	engramBytes, err := tensorPayloadBytes(info)
	if err != nil {
		t.Fatalf("payload: %v", err)
	}
	wsB, err := NewWeightSource(without, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource(B): %v", err)
	}

	planA, err := wsA.EstimateCPUOffloadExpertsStreamedBoundedDenseMemoryPlan(1, 1<<30, 8<<30)
	if err != nil {
		t.Fatalf("plan A: %v", err)
	}
	planB, err := wsB.EstimateCPUOffloadExpertsStreamedBoundedDenseMemoryPlan(1, 1<<30, 8<<30)
	if err != nil {
		t.Fatalf("plan B: %v", err)
	}
	if planA.DeviceTotal() == 0 {
		t.Fatal("device total is zero; fixture is degenerate")
	}
	if planA.DeviceTotal() != planB.DeviceTotal() {
		t.Fatalf("device total changed when the engram table (%d B) was added: %d vs %d",
			engramBytes, planA.DeviceTotal(), planB.DeviceTotal())
	}
	for _, d := range planA {
		if uint64(d.Bytes) == engramBytes {
			t.Fatalf("plan still charges the engram table payload %d as a row: %+v", engramBytes, d)
		}
	}
}

// TestV41EngramTableSkipIsScopedToTheTable is the negative half: the projection-side
// Engram tensors the forward DOES read as f32 (engram_wkv/engram_q/engram_k) must NOT be
// dropped, and neither must a normal dense tensor.
func TestV41EngramTableSkipIsScopedToTheTable(t *testing.T) {
	if !glmMoeDsaSkipGGUFTensorForType("deepseek41", "blk.1.engram_embd.weight") {
		t.Fatal("packed engram table must be dropped from byte-accounting")
	}
	if !glmMoeDsaSkipGGUFTensorForType("deepseek41", "blk.1.engram_table") {
		t.Fatal("legacy engram_table spelling must be dropped")
	}
	if glmMoeDsaSkipGGUFTensorForType("deepseek41", "blk.1.engram_wkv.weight") {
		t.Fatal("projection-side engram_wkv must NOT be dropped (the forward reads it)")
	}
	if glmMoeDsaSkipGGUFTensorForType("deepseek41", "blk.0.attn_q_a.weight") {
		t.Fatal("a normal dense tensor must NOT be dropped")
	}
	if glmMoeDsaSkipGGUFTensorForType("qwen3moe", "blk.1.engram_embd.weight") {
		t.Fatal("the engram-table drop is deepseek41-only")
	}
}
