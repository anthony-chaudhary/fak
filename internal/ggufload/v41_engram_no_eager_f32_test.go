package ggufload

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// v41_engram_no_eager_f32_test.go is the fak#13152 RED->GREEN regression.
//
// The published vcruz305/DeepSeek-V4.1-Flash-GGUF Q2_K checkpoint stores its
// packed Engram tables as blk.<L>.engram_embd.weight of type Q2_K with dims
// [256, ~384M rows] - ONE Q2_K super-block per row. The native forward reads
// those rows through the bounded V41EngramRowSource seam, never as an f32
// matrix. Before the fix the materializing quant loader had no Engram branch,
// so the tensor canonical-mapped to model.engram.<L>.engram_embd.weight, failed
// the expert-only ResidentKQuantEligible gate at quant_q4k_loader.go:1059-1064,
// and fell through to dequantF32Limited - a single ~98.3B-element (366.2 GiB)
// make([]float32, n) that aborts the process with
// "fatal error: runtime: out of memory" on a 62 GiB Halo.
//
// The fix has two arms, both pinned here:
//   (a) the V4.1 packed Engram table is DROPPED from the materializing load
//       (deepseek41EngramTableTensor), exactly like the MTP/vision sidecar, so
//       it is never eager-dequantized at all; and
//   (b) dequantF32IntoLimited fails CLOSED above MaxEagerF32Bytes with a typed
//       error NAMING the tensor, so any residual over-budget leak is a refusal,
//       never an unbounded allocation.

// TestDeepSeek41EngramTableTensorPredicate pins arm (a): the packed Engram
// TABLE is recognized (both the vcruz engram_embd spelling and the ds4
// engram_table spelling), while the projection-side Engram tensors the forward
// DOES consume as f32 weights are deliberately NOT matched.
func TestDeepSeek41EngramTableTensorPredicate(t *testing.T) {
	table := []string{
		"blk.1.engram_embd.weight",
		"blk.14.engram_embd.weight",
		"blk.1.engram_table",
	}
	for _, name := range table {
		if !deepseek41EngramTableTensor(name) {
			t.Errorf("deepseek41EngramTableTensor(%q) = false; the packed Engram table must be recognized so it is never eager-dequantized", name)
		}
	}

	// Projection-side Engram tensors ARE f32 forward weights and must still load.
	projections := []string{
		"blk.1.engram_wkv.weight",
		"blk.1.engram_q.weight",
		"blk.1.engram_k.weight",
		"blk.1.engram_kv.weight",
		"blk.1.engram_q_norm.weight",
		"blk.1.engram_k_norm.weight",
	}
	for _, name := range projections {
		if deepseek41EngramTableTensor(name) {
			t.Errorf("deepseek41EngramTableTensor(%q) = true; projection-side Engram tensors must load as f32, not be dropped", name)
		}
	}

	// Non-Engram and non-deepseek41-namespace names must not match.
	for _, name := range []string{
		"blk.1.attn_q_a.weight",
		"blk.1.ffn_gate_exps.weight",
		"token_embd.weight",
		"engram_embd.weight",
	} {
		if deepseek41EngramTableTensor(name) {
			t.Errorf("deepseek41EngramTableTensor(%q) = true; only blk.<L>.* Engram tables may match", name)
		}
	}
}

// TestDeepSeek41EngramTableDroppedFromQuantLoad pins arm (a) end-to-end at the
// per-tensor work seam: the published-shape Q2_K Engram table must produce an
// EMPTY tensorWork (no f32 payload, no resident payload, no error) - it is
// dropped, never materialized. This is the exact tensor that today OOMs.
func TestDeepSeek41EngramTableDroppedFromQuantLoad(t *testing.T) {
	ws := v41OpenPublishedShards(t, TensorQ2_K)

	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config on the published-shape header: %v", err)
	}
	info, ok := ws.Tensor("blk.1.engram_embd.weight")
	if !ok {
		t.Fatal("fixture is missing blk.1.engram_embd.weight")
	}
	if info.Type != TensorQ2_K {
		t.Fatalf("fixture tensor type = %s, want Q2_K", info.Type)
	}

	loadOpts, err := resolveQ4KLoadOptions(cfg, []Q4KLoadOption{WithDenseQ2KResident(true)})
	if err != nil {
		t.Fatalf("resolveQ4KLoadOptions: %v", err)
	}

	tw := ws.computeQ4KTensorWork(info, cfg, false, nil, loadOpts, 1)
	if tw.err != nil {
		t.Fatalf("computeQ4KTensorWork refused the Engram table with %v; it must be DROPPED, not refused", tw.err)
	}
	if len(tw.pending) != 0 {
		t.Fatalf("Engram table produced %d pending tensor(s); it must materialize nothing", len(tw.pending))
	}
	if tw.acctResident {
		t.Error("Engram table was marked resident; it is served by the row source, not the k-quant store")
	}
}

// TestDequantF32LimitedRefusesOverBudget pins arm (b): a tensor whose eager f32
// payload exceeds the budget must fail closed with a typed error naming the
// tensor, and must NOT reach an unbounded make([]float32, n). The budget is
// lowered for the test so the fixture stays small; the refusal is arithmetic,
// not a gigabyte allocation.
func TestDequantF32LimitedRefusesOverBudget(t *testing.T) {
	orig := MaxEagerF32Bytes
	MaxEagerF32Bytes = 64 // deliberately tiny: refuse anything above 16 float32
	defer func() { MaxEagerF32Bytes = orig }()

	// A Q2_K tensor of qkK (256) weights = one 84-byte super-block, whose f32
	// payload (1024 bytes) is well over the 64-byte test budget.
	block, _ := q2KFixtureBlock()
	info := TensorInfo{Name: "blk.1.engram_embd.weight", Type: TensorQ2_K, Dims: []uint64{qkK}}

	_, err := dequantF32Limited(info, block, 1)
	if err == nil {
		t.Fatal("dequantF32Limited allowed an over-budget eager f32 allocation")
	}
	var over *ErrEagerF32BudgetExceeded
	if !errors.As(err, &over) {
		t.Fatalf("refusal %v is not an *ErrEagerF32BudgetExceeded; a bare runtime OOM is the failure mode this prevents", err)
	}
	if over.Tensor != info.Name {
		t.Errorf("refusal names tensor %q, want %q", over.Tensor, info.Name)
	}
	if over.WantBytes != int64(qkK)*4 {
		t.Errorf("refusal WantBytes = %d, want %d", over.WantBytes, int64(qkK)*4)
	}
}

// TestDequantF32LimitedAllowsSmallTensor is the anti-over-refusal control: a
// genuinely small tensor still dequantizes to f32 under the normal budget, so
// the bound does not break the correct eager-f32 path.
func TestDequantF32LimitedAllowsSmallTensor(t *testing.T) {
	block, want := q2KFixtureBlock()
	info := TensorInfo{Name: "blk.0.attn_norm.weight", Type: TensorQ2_K, Dims: []uint64{qkK}}

	got, err := dequantF32Limited(info, block, 1)
	if err != nil {
		t.Fatalf("dequantF32Limited refused a small in-budget tensor: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("dequantized %d values, want %d", len(got), len(want))
	}
}

var _ = model.Config{}
