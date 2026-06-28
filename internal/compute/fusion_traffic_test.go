package compute

import (
	"math"
	"testing"
)

// fusion_traffic_test.go — the host-witnessable rungs for issue #279 (B-005)'s first
// acceptance bullet, "20%+ reduction in memory traffic." None of these need a CUDA device:
// HBM bytes moved are an exact count of operands, so the fused-vs-unfused reduction is
// computed (and gated) the same on any host. They pin (1) the attention score-matrix
// elimination clears 20% at the long-sequence targets and grows with P, (2) the reduction is
// correctly tiny at P=1 (decode), which is why the bullet is scoped to long sequences, (3)
// MLP activation-fusion saves traffic but is weight-bound (so attention carries the headline),
// (4) the whole-block reduction clears 20%, and (5) the FusionTraffic invariants + the
// acceptance predicate. The remaining bullet — "15%+ throughput gain on long sequences" — is
// a wall-clock measurement deferred to a CUDA node (see FUSED-B005-NOTES.md); nothing here
// fabricates a throughput number.

// llamaGeom is a Llama-3-8B-ish prefill geometry at length P with the given weight dtype —
// 32 query heads over 8 KV heads (GQA), head dim 128, 14336 FFN inner width. The targets
// P=256/512/1024 are the long-sequence regime the issue's acceptance is scoped to.
func llamaGeom(P int, dt Dtype) PrefillGeometry {
	return PrefillGeometry{
		DModel: 4096, NHeads: 32, NKVHeads: 8, HeadDim: 128, DFF: 14336,
		NLayers: 32, Vocab: 128256, P: P, WeightDtype: dt,
	}
}

// TestAttentionTrafficMeets20PctAtLongSeq is the load-bearing rung: flash attention removes
// the materialized P×P score matrix from HBM, and at the long-sequence targets that score
// matrix dwarfs the Q/K/V/O traffic — so the reduction clears the issue's 20% gate. The gate
// is checked through MeetsMemoryTrafficTarget (the same predicate a CUDA node grades its
// MEASURED traffic with), not by re-deriving 20% inline.
func TestAttentionTrafficMeets20PctAtLongSeq(t *testing.T) {
	for _, P := range []int{256, 512, 1024} {
		ft := llamaGeom(P, F32).AttentionHBMTraffic()
		if !MeetsMemoryTrafficTarget(ft, 0.20) {
			t.Fatalf("P=%d: attention HBM reduction %.4f < 0.20 (unfused=%d fused=%d saved=%d)",
				P, ft.ReductionFrac, ft.Unfused, ft.Fused, ft.Saved)
		}
		t.Logf("#279 attention HBM traffic P=%d: unfused=%d fused=%d reduction=%.4f",
			P, ft.Unfused, ft.Fused, ft.ReductionFrac)
	}
}

// TestAttentionTrafficGrowsWithSeqLen pins the structural fact that motivates the
// long-sequence scoping: the score matrix is O(P²) while Q/K/V/O is O(P), so the fraction of
// HBM traffic fusion removes is strictly increasing in P.
func TestAttentionTrafficGrowsWithSeqLen(t *testing.T) {
	var prev float64 = -1
	for _, P := range []int{128, 256, 512, 1024, 2048} {
		ft := llamaGeom(P, F32).AttentionHBMTraffic()
		if ft.ReductionFrac <= prev {
			t.Fatalf("P=%d: reduction %.4f did not increase over previous %.4f (must be monotone in P)",
				P, ft.ReductionFrac, prev)
		}
		prev = ft.ReductionFrac
	}
}

// TestDecodeAttentionTrafficIsSmall is the honesty rung for the long-sequence scoping: at
// P=1 (single-token decode) the "score matrix" is one row, so flash removes almost nothing —
// the saving lives at long sequences, exactly as the acceptance bullet is worded. If this
// ever reported a large decode reduction the model would be overclaiming.
func TestDecodeAttentionTrafficIsSmall(t *testing.T) {
	decode := llamaGeom(1, F32).AttentionHBMTraffic()
	long := llamaGeom(1024, F32).AttentionHBMTraffic()
	if decode.ReductionFrac >= 0.05 {
		t.Fatalf("P=1 decode reduction %.4f unexpectedly large — the win is supposed to be long-sequence-only", decode.ReductionFrac)
	}
	if decode.ReductionFrac >= long.ReductionFrac {
		t.Fatalf("decode reduction %.4f should be far below long-sequence %.4f", decode.ReductionFrac, long.ReductionFrac)
	}
}

// TestMLPTrafficIsWeightBound pins the honest structural finding that MLP activation-fusion,
// while a real saving (the [P,DFF] intermediate stays on-chip), is a SMALLER fraction of MLP
// HBM than attention fusion is of attention HBM — because the FFN weights dominate the MLP's
// HBM. It also checks the dtype direction: quantized (Q8_0) weights shrink the weight term,
// so the SAME activation saving is a LARGER fraction than with dense f32 weights.
func TestMLPTrafficIsWeightBound(t *testing.T) {
	P := 1024
	f32 := llamaGeom(P, F32).MLPHBMTraffic()
	q8 := llamaGeom(P, Q8_0).MLPHBMTraffic()
	if f32.Saved <= 0 || q8.Saved <= 0 {
		t.Fatalf("MLP fusion must save SOME traffic: f32 saved=%d q8 saved=%d", f32.Saved, q8.Saved)
	}
	// same activation round-trip is eliminated in both, so the absolute bytes saved match.
	if f32.Saved != q8.Saved {
		t.Fatalf("MLP saved bytes should not depend on weight dtype: f32=%d q8=%d", f32.Saved, q8.Saved)
	}
	if q8.ReductionFrac <= f32.ReductionFrac {
		t.Fatalf("quantized weights should make the SAME activation saving a larger fraction: q8 %.4f !> f32 %.4f",
			q8.ReductionFrac, f32.ReductionFrac)
	}
	// the weight-bound point: with dense f32 weights, MLP fusion alone does NOT clear 20% —
	// which is why the headline reduction is carried by attention fusion (asserted below).
	attn := llamaGeom(P, F32).AttentionHBMTraffic()
	if f32.ReductionFrac >= attn.ReductionFrac {
		t.Fatalf("MLP fusion %.4f should be a smaller fraction than attention fusion %.4f (MLP is weight-bound)",
			f32.ReductionFrac, attn.ReductionFrac)
	}
	t.Logf("#279 MLP HBM traffic P=%d: f32 reduction=%.4f, q8 reduction=%.4f, attention reduction=%.4f",
		P, f32.ReductionFrac, q8.ReductionFrac, attn.ReductionFrac)
}

// TestBlockTrafficMeets20Pct checks the whole-block claim: attention + MLP fusion together
// remove ≥20% of a transformer block's HBM traffic at a long-sequence target, for both dense
// (f32) and quantized (Q8_0) weights — the attention score-matrix elimination carries the
// block over the gate even when MLP fusion alone (f32) does not.
func TestBlockTrafficMeets20Pct(t *testing.T) {
	for _, dt := range []Dtype{F32, Q8_0} {
		blk := llamaGeom(1024, dt).BlockHBMTraffic()
		if !MeetsMemoryTrafficTarget(blk, 0.20) {
			t.Fatalf("dtype=%v: block HBM reduction %.4f < 0.20 (unfused=%d fused=%d)",
				dt, blk.ReductionFrac, blk.Unfused, blk.Fused)
		}
		t.Logf("#279 block HBM traffic P=1024 dtype=%v: unfused=%d fused=%d reduction=%.4f",
			dt, blk.Unfused, blk.Fused, blk.ReductionFrac)
	}
}

// TestFusionTrafficInvariants locks the bookkeeping so a reduction number can never disagree
// with the byte counts behind it: Saved == Unfused − Fused, the fused path never moves more
// than the unfused one, and ReductionFrac == Saved/Unfused in [0,1). Checked across stages
// and lengths, including the degenerate P=0 geometry (no divide-by-zero).
func TestFusionTrafficInvariants(t *testing.T) {
	for _, P := range []int{0, 1, 256, 1024} {
		g := llamaGeom(P, Q8_0)
		for _, ft := range []FusionTraffic{g.AttentionHBMTraffic(), g.MLPHBMTraffic(), g.BlockHBMTraffic()} {
			if ft.Fused > ft.Unfused {
				t.Fatalf("%s P=%d: fused %d > unfused %d (fusion cannot ADD traffic)", ft.Stage, P, ft.Fused, ft.Unfused)
			}
			if ft.Saved != ft.Unfused-ft.Fused {
				t.Fatalf("%s P=%d: Saved %d != Unfused-Fused %d", ft.Stage, P, ft.Saved, ft.Unfused-ft.Fused)
			}
			want := 0.0
			if ft.Unfused > 0 {
				want = float64(ft.Saved) / float64(ft.Unfused)
			}
			if math.Abs(ft.ReductionFrac-want) > 1e-12 {
				t.Fatalf("%s P=%d: ReductionFrac %.12f != Saved/Unfused %.12f", ft.Stage, P, ft.ReductionFrac, want)
			}
			if ft.ReductionFrac < 0 || ft.ReductionFrac >= 1 {
				t.Fatalf("%s P=%d: ReductionFrac %.6f out of [0,1)", ft.Stage, P, ft.ReductionFrac)
			}
		}
	}
}

// TestMeetsMemoryTrafficTargetPredicate pins the acceptance predicate itself: a non-positive
// target is no valid gate (false), and otherwise it is exactly "reduction ≥ target."
func TestMeetsMemoryTrafficTargetPredicate(t *testing.T) {
	ft := llamaGeom(1024, F32).AttentionHBMTraffic()
	if MeetsMemoryTrafficTarget(ft, 0) || MeetsMemoryTrafficTarget(ft, -0.1) {
		t.Fatalf("non-positive target must be no valid gate (false)")
	}
	for _, target := range []float64{0.20, 0.50, ft.ReductionFrac, ft.ReductionFrac + 1e-9} {
		got := MeetsMemoryTrafficTarget(ft, target)
		want := ft.ReductionFrac >= target
		if got != want {
			t.Fatalf("MeetsMemoryTrafficTarget(%.6f, %.9f)=%v want %v", ft.ReductionFrac, target, got, want)
		}
	}
}
