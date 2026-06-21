package model

import (
	"math"
	"testing"
)

// tensor_parallel_attn_test.go — gates for the second Megatron tensor-parallel block,
// attention. The claim mirrors the FFN's: sharding the heads across ranks leaves each
// rank's attention-output band BIT-EXACT vs the monolith's slice (heads are independent),
// and the row-parallel output projection's AllReduce matches the single-device result
// within the documented round-off. Plus a composed full-layer gate (attention + FFN) — the
// closest standalone proof of a tensor-parallel transformer layer.

// tpAttnRand reuses the package's deterministic LCG so these gates reproduce exactly.
func tpAttnRand(n int, seed uint64) []float32 { return tpRand(n, seed) }

func tpSeq(seq, hidden int, seed uint64) [][]float32 {
	X := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		X[t] = tpRand(hidden, seed+uint64(t)*7+1)
	}
	return X
}

// maxRelAbsRows returns the max RELATIVE drift over coordinates whose reference magnitude
// exceeds a small floor, and the max ABSOLUTE drift over the REMAINING near-zero coordinates
// (the ones the relative test skips). Splitting this way lets a tight relative bound govern
// the large coordinates while the absolute bound still backstops near-zero coordinates a
// gross error could otherwise hide in — without falsely failing on legitimately large
// (unnormalized) outputs, whose absolute drift scales with their magnitude.
func maxRelAbsRows(got, ref [][]float32) (rel, absSmall float64) {
	for t := range got {
		for i := range got[t] {
			d := math.Abs(float64(got[t][i] - ref[t][i]))
			den := math.Abs(float64(ref[t][i]))
			if den > 1e-6 {
				if r := d / den; r > rel {
					rel = r
				}
			} else if d > absSmall {
				absSmall = d
			}
		}
	}
	return rel, absSmall
}

// TestTensorParallelAttentionMatchesMonolith proves the head-sharded attention block
// reproduces single-device attention: the final output matches within the o_proj AllReduce
// round-off, and (the structural claim) each rank's attention-output band is BIT-EXACT vs
// the monolith's matching head slice — heads carry no cross-rank dependency, so only the
// output projection reassociates. Exercises GQA (nKV<nH) and MHA (nKV==nH).
func TestTensorParallelAttentionMatchesMonolith(t *testing.T) {
	cases := []struct {
		hidden, nH, nKV, headDim, seq int
	}{
		{64, 8, 8, 8, 6},   // MHA
		{128, 8, 2, 16, 5}, // GQA grp=4
		{96, 6, 3, 16, 7},  // GQA grp=2
		{32, 4, 1, 8, 4},   // MQA (single kv head) — only 1-rank plan valid
	}
	for _, c := range cases {
		scale := float32(1.0 / math.Sqrt(float64(c.headDim)))
		qW := tpAttnRand(c.nH*c.headDim*c.hidden, uint64(c.hidden*7+c.nH+1))
		kW := tpAttnRand(c.nKV*c.headDim*c.hidden, uint64(c.hidden*11+c.nKV+2))
		vW := tpAttnRand(c.nKV*c.headDim*c.hidden, uint64(c.hidden*13+c.nKV+3))
		oW := tpAttnRand(c.hidden*c.nH*c.headDim, uint64(c.hidden*17+c.nH+4))
		X := tpSeq(c.seq, c.hidden, uint64(c.hidden*19+5))

		mono := referenceAttention(qW, kW, vW, oW, X, c.hidden, c.nH, c.nKV, c.headDim, scale)
		monoBand := referenceAttentionBandSlice(qW, kW, vW, X, c.hidden, c.nH, c.nKV, c.headDim, scale)
		grp := c.nH / c.nKV

		for _, ranks := range []int{1, 2, 3, 4, 8} {
			if ranks > c.nKV {
				continue // can't have more ranks than KV-head groups
			}
			// NOTE: ranks need NOT divide nKV — NewTPPlan tiles near-evenly, so this
			// exercises UNEVEN KV shards (e.g. nKV=3, ranks=2 -> widths 2,1), the exact
			// place a band-offset bug would hide.
			plan, err := NewTPPlan(c.nKV, ranks)
			if err != nil {
				t.Fatalf("NewTPPlan(nKV=%d, ranks=%d): %v", c.nKV, ranks, err)
			}
			// Structural bit-exactness: each rank's attention-output band == monolith slice.
			for _, s := range plan.Shards {
				band := attnBandForShard(s, grp)
				got := attentionOutputBand(qW, kW, vW, X, band, grp, c.hidden, c.headDim, scale)
				for tt := 0; tt < c.seq; tt++ {
					for i := 0; i < (band.QHi-band.QLo)*c.headDim; i++ {
						g := got[tt][i]
						w := monoBand[tt][band.QLo*c.headDim+i]
						if math.Float32bits(g) != math.Float32bits(w) {
							t.Fatalf("attn[%+v] ranks=%d rank %d band pos %d i=%d not bit-identical to monolith (%v != %v)",
								c, ranks, s.Rank, tt, i, g, w)
						}
					}
				}
			}
			// End-to-end: TP attention within the o_proj AllReduce round-off of the monolith.
			out, err := TensorParallelAttention(qW, kW, vW, oW, X, c.hidden, c.nH, c.nKV, c.headDim, scale, plan, LocalCollective{})
			if err != nil {
				t.Fatalf("TensorParallelAttention%+v ranks=%d: %v", c, ranks, err)
			}
			if len(out) != c.seq {
				t.Fatalf("TP attention returned %d positions, want %d", len(out), c.seq)
			}
			// Bit-exact vs the shard-grouped rank-order reference (max|Δ|=0): pins the o_proj
			// AllReduce's reduction order, invisible to the loose vs-reference round-off bound.
			oref, err := TensorParallelAttentionReference(qW, kW, vW, oW, X, c.hidden, c.nH, c.nKV, c.headDim, scale, plan)
			if err != nil {
				t.Fatalf("TensorParallelAttentionReference%+v ranks=%d: %v", c, ranks, err)
			}
			for tt := 0; tt < c.seq; tt++ {
				for i := 0; i < c.hidden; i++ {
					if math.Float32bits(out[tt][i]) != math.Float32bits(oref[tt][i]) {
						t.Fatalf("attn%+v ranks=%d pos %d i=%d: %v != rank-order reference %v (o_proj AllReduce order not pinned)",
							c, ranks, tt, i, out[tt][i], oref[tt][i])
					}
				}
			}
			if ranks == 1 {
				for tt := 0; tt < c.seq; tt++ {
					for i := 0; i < c.hidden; i++ {
						if math.Float32bits(out[tt][i]) != math.Float32bits(mono[tt][i]) {
							t.Fatalf("attn%+v ranks=1 pos %d i=%d not bit-identical to monolith", c, tt, i)
						}
					}
				}
			}
			rel, abs := maxRelAbsRows(out, mono)
			if rel > 1e-4 || abs > 1e-3 {
				t.Fatalf("attn%+v ranks=%d drift rel %.2e abs %.2e exceeds round-off bound", c, ranks, rel, abs)
			}
		}
	}
}

// TestTensorParallelAttentionFailsClosed pins the shape guards: a plan that does not shard
// the KV-head dimension, or a rank count that splits a GQA group, is rejected before any
// matmul.
func TestTensorParallelAttentionFailsClosed(t *testing.T) {
	hidden, nH, nKV, headDim := 64, 8, 4, 8
	scale := float32(0.125)
	qW := tpAttnRand(nH*headDim*hidden, 1)
	kW := tpAttnRand(nKV*headDim*hidden, 2)
	vW := tpAttnRand(nKV*headDim*hidden, 3)
	oW := tpAttnRand(hidden*nH*headDim, 4)
	X := tpSeq(3, hidden, 5)

	// plan.Dim must equal nKV, not nH.
	badDim, _ := NewTPPlan(nH, 2)
	if _, err := TensorParallelAttention(qW, kW, vW, oW, X, hidden, nH, nKV, headDim, scale, badDim, LocalCollective{}); err == nil {
		t.Fatalf("expected error when plan shards nH instead of nKV")
	}
	// nH must be a multiple of nKV.
	good, _ := NewTPPlan(nKV, 2)
	if _, err := TensorParallelAttention(qW, kW, vW, oW, X, hidden, 7 /*nH not mult of nKV*/, nKV, headDim, scale, good, LocalCollective{}); err == nil {
		t.Fatalf("expected error when nH is not a multiple of nKV")
	}
	// Correct call succeeds.
	if _, err := TensorParallelAttention(qW, kW, vW, oW, X, hidden, nH, nKV, headDim, scale, good, LocalCollective{}); err != nil {
		t.Fatalf("valid TP attention errored: %v", err)
	}
}

// TestTensorParallelLayerMatchesMonolith is the composed proof: a full transformer layer
// (attention block + FFN block, each tensor-parallel) reproduces the single-device layer
// within round-off. It threads the attention output as the FFN input (no residual/norm —
// those are elementwise and rank-invariant; the point is the two sharded matmul blocks
// composing). This is the closest standalone witness of a tensor-parallel layer end to end.
func TestTensorParallelLayerMatchesMonolith(t *testing.T) {
	hidden, nH, nKV, headDim, seq := 64, 8, 2, 8, 5
	inter := 256
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	qW := tpAttnRand(nH*headDim*hidden, 21)
	kW := tpAttnRand(nKV*headDim*hidden, 22)
	vW := tpAttnRand(nKV*headDim*hidden, 23)
	oW := tpAttnRand(hidden*nH*headDim, 24)
	gateW := tpAttnRand(inter*hidden, 25)
	upW := tpAttnRand(inter*hidden, 26)
	downW := tpAttnRand(hidden*inter, 27)
	X := tpSeq(seq, hidden, 28)

	// Monolith layer: attention then FFN, position by position.
	monoAttn := referenceAttention(qW, kW, vW, oW, X, hidden, nH, nKV, headDim, scale)
	monoOut := make([][]float32, seq)
	for pos := 0; pos < seq; pos++ {
		g := matRows(gateW, monoAttn[pos], inter, hidden)
		u := matRows(upW, monoAttn[pos], inter, hidden)
		a := make([]float32, inter)
		for i := 0; i < inter; i++ {
			a[i] = silu(g[i]) * u[i]
		}
		monoOut[pos] = matRows(downW, a, hidden, inter)
	}

	for _, ranks := range []int{1, 2} { // nKV=2 -> 1 or 2 ranks (whole groups)
		attnPlan, _ := NewTPPlan(nKV, ranks)
		tpAttn, err := TensorParallelAttention(qW, kW, vW, oW, X, hidden, nH, nKV, headDim, scale, attnPlan, LocalCollective{})
		if err != nil {
			t.Fatalf("TP attention ranks=%d: %v", ranks, err)
		}
		// FFN can shard the intermediate at any rank count up to inter; use the same ranks.
		ffnPlan, _ := NewTPPlan(inter, ranks)
		tpOut := make([][]float32, seq)
		for pos := 0; pos < seq; pos++ {
			o, err := TensorParallelFFN(gateW, upW, downW, tpAttn[pos], hidden, inter, ffnPlan, LocalCollective{})
			if err != nil {
				t.Fatalf("TP FFN ranks=%d pos %d: %v", ranks, pos, err)
			}
			tpOut[pos] = o
		}
		if rel, abs := maxRelAbsRows(tpOut, monoOut); rel > 1e-4 || abs > 1e-3 {
			t.Fatalf("TP layer ranks=%d drift rel %.2e abs %.2e exceeds round-off bound", ranks, rel, abs)
		}
	}
}
