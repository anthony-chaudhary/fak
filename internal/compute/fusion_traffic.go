package compute

// fusion_traffic.go — the host-tractable accounting for issue #279 (B-005), "CUDA fused
// kernels for attention + MLP." The fused ATTENTION kernel itself already shipped (#486:
// k_flash_attention / Caps.FusedAttn replaced the naive one-block-per-head decode kernel on
// the live Attention path, with the op-level cosine-parity witness + the fused-vs-naive
// microbench in cuda_flash_test.go). What #279's acceptance also asks — and what no code in
// this package yet quantified — is the FIRST acceptance bullet:
//
//	[ ] 20%+ reduction in memory traffic
//
// That number is the one piece of #279 a CUDA-less host CAN witness exactly, because the
// HBM bytes a kernel must move are a COUNT of operands, not a measurement — the same honesty
// discipline as prefill.go's PrefillCostModel (exact FLOPs + bytes, no timer, no fabricated
// throughput). This file ships the analytic memory-traffic model for the two fusion points
// the issue names (fused attention, fused MLP), the fused-vs-unfused reduction it yields,
// and the acceptance predicate (MeetsMemoryTrafficTarget). The REMAINING acceptance bullet —
// "15%+ throughput gain on long sequences" — is a WALL-CLOCK measurement on a CUDA device and
// is deferred to a CUDA bench node (see FUSED-B005-NOTES.md); no throughput number is run,
// estimated, or fabricated here.
//
// HBM-traffic vs FLOP-operands — the conflation this model is careful NOT to make.
// attnCost() in prefill.go counts FLOP-OPERANDS (HeadDim floats per (query,key) pair) to
// land its ~0.5 FLOP/byte arithmetic intensity; that is the right denominator for a roofline
// but it is NOT the HBM byte count, because a real matmul streams K once into the compute
// array and reuses it across the systolic tile — the per-multiply operand touch is not a
// distinct HBM read. This file instead counts HBM TRAFFIC the FlashAttention way: Q/K/V/O are
// streamed once each, and the dominant non-fused term is the P×P score/probability matrix
// that standard attention MATERIALIZES to HBM and reads back, which flash never writes. The
// two accountings answer different questions and must not be summed together.

// FusionTraffic is the HBM-traffic comparison for one fusable stage: the bytes a non-fused
// (separate-kernel) lowering moves through HBM versus the bytes the fused kernel moves, the
// bytes saved, and the saved fraction. Every field is an EXACT count of operands the stage
// must read or write — no wall-clock, no device constant — so it is identical on any host.
// ReductionFrac is Saved/Unfused, the quantity #279's "20%+ reduction in memory traffic"
// acceptance bullet gates (see MeetsMemoryTrafficTarget).
type FusionTraffic struct {
	Stage         string  // "attention" | "mlp" | "block"
	Unfused       int64   // HBM bytes the separate-kernel lowering moves
	Fused         int64   // HBM bytes the fused kernel moves
	Saved         int64   // Unfused − Fused (≥ 0 by construction)
	ReductionFrac float64 // Saved / Unfused — the fraction of HBM traffic the fusion removes
}

// makeFusionTraffic builds a FusionTraffic from the two byte counts, deriving Saved and
// ReductionFrac so the three can never disagree. A non-positive Unfused yields a zeroed
// reduction rather than a divide-by-zero (the degenerate P=0 / empty-geometry case).
func makeFusionTraffic(stage string, unfused, fused int64) FusionTraffic {
	saved := unfused - fused
	if saved < 0 {
		saved = 0 // a fused path never moves MORE than the separate kernels it replaces
	}
	return FusionTraffic{
		Stage:         stage,
		Unfused:       unfused,
		Fused:         fused,
		Saved:         saved,
		ReductionFrac: ratio(saved, unfused),
	}
}

// attnCommonBytes is the HBM traffic BOTH the standard and the flash attention pay for one
// layer at this geometry: Q and O over the query heads, K and V over the (GQA-shared) KV
// heads, each streamed once as f32. It is the part fusion does NOT change — only the score
// matrix differs — so it is the denominator's floor.
func (g PrefillGeometry) attnCommonBytes() int64 {
	P := int64(g.P)
	hd := int64(g.HeadDim)
	q := int64(g.NHeads) * P * hd   // Q read
	o := int64(g.NHeads) * P * hd   // O write
	k := int64(g.NKVHeads) * P * hd // K read (KV heads, GQA)
	v := int64(g.NKVHeads) * P * hd // V read
	return 4 * (q + o + k + v)      // f32
}

// attnScoreMatrixPasses is the number of times a non-fused attention TOUCHES the P×P score/
// probability matrix in HBM, counted conservatively: one write (S = QKᵀ) plus one read (the
// softmax+PV consumer) is the irreducible minimum a "materialize the scores, then consume
// them" lowering must pay. The retained naive device kernel (k_attention) actually pays MORE
// — it writes the row, reads it for the max, reads+writes it for the in-place exp, and reads
// it again for the weighted-V pass (2 writes + 3 reads) — so 2 UNDERSTATES the real saving;
// using the floor keeps the reduction claim defensible rather than inflated.
const attnScoreMatrixPasses = 2

// AttentionHBMTraffic models the HBM traffic of one attention layer at geometry g, fused
// (flash / online-softmax, #486) vs unfused (standard, score matrix materialized to HBM).
// Both pay attnCommonBytes (Q/K/V/O streamed once); the unfused path ADDITIONALLY writes and
// reads the causal score/probability matrix — P·(P+1)/2 attended entries per query head,
// attnScoreMatrixPasses times — which the flash kernel never materializes (its running
// (max,sum,acc) lives in registers/shared memory, as the kernel comment in cuda_kernels.cu
// records). So the saved traffic is exactly the score-matrix materialization, and the
// reduction grows with P: at long sequences (the issue's regime) the O(P²) score matrix
// dwarfs the O(P·HeadDim) Q/K/V/O traffic, clearing 20% with room to spare; at P=1 (decode)
// the score "matrix" is a single row and the saving is small — which is the honest reason the
// acceptance bullet is scoped to LONG sequences.
func (g PrefillGeometry) AttentionHBMTraffic() FusionTraffic {
	common := g.attnCommonBytes()
	P := int64(g.P)
	causalEntries := P * (P + 1) / 2 // attended (query,key) pairs over the causal triangle
	scoreBytes := attnScoreMatrixPasses * int64(g.NHeads) * causalEntries * 4
	unfused := common + scoreBytes
	fused := common // flash never writes the score matrix to HBM
	return makeFusionTraffic("attention", unfused, fused)
}

// mlpIntermediatePasses is the number of times a non-fused SwiGLU MLP TOUCHES a [P,DFF]
// intermediate activation in HBM that a fused kernel keeps on-chip, counted conservatively:
// one write (the elementwise SwiGLU result) plus one read (the down-projection consuming it)
// is the irreducible minimum a "materialize the activation between the elementwise op and the
// matmul" lowering pays — exactly the round-trip the Vulkan backend's SwiGLUMatMulAddInPlace
// already fuses away. A fully unfused gate→up→swiglu→down chain materializes more (gate, up,
// and the swiglu output each written and reread); 2 understates the real saving, keeping the
// claim defensible.
const mlpIntermediatePasses = 2

// MLPHBMTraffic models the HBM traffic of one SwiGLU FFN layer at geometry g, fused
// (SwiGLU folded into the down-projection, the CUDA analogue of Vulkan's
// SwiGLUMatMulAddInPlace) vs unfused (the [P,DFF] activation materialized to HBM between the
// up/gate projection and the down projection). Both stream the same FFN weights (gate, up,
// down) and the same X/Y residual-width activations once; the unfused path ADDITIONALLY
// writes and reads the [P,DFF] intermediate, which the fused kernel never spills. Unlike
// attention, the FFN weights dominate the HBM here, so the activation-fusion saving is a
// SMALLER fraction of total MLP traffic — and it shrinks further as the weights grow (dense
// f32) or stays larger when the weights are quantized (Q8_0); this is the honest, structural
// reason the headline "20%+ reduction" is carried by attention fusion, with MLP fusion an
// additive contribution. WeightDtype selects the weight side via the shared weightBytes.
func (g PrefillGeometry) MLPHBMTraffic() FusionTraffic {
	P := int64(g.P)
	D := int64(g.DModel)
	dff := int64(g.DFF)
	// weights: gate[DFF,DModel] + up[DFF,DModel] + down[DModel,DFF], streamed once by both.
	weights := g.weightBytes(g.DFF, g.DModel)*2 + g.weightBytes(g.DModel, g.DFF)
	// residual-width activations both pay: X read (gate+up share it) + Y write.
	xy := 4 * (P*D + P*D)
	common := weights + xy
	interBytes := mlpIntermediatePasses * P * dff * 4 // the [P,DFF] activation round-trip
	unfused := common + interBytes
	fused := common // the SwiGLU intermediate stays on-chip
	return makeFusionTraffic("mlp", unfused, fused)
}

// BlockHBMTraffic sums the attention and MLP fusion traffic for one transformer block — the
// whole-block memory-traffic reduction a stacked fused decoder layer realizes. At long
// sequences the attention score-matrix elimination dominates the sum, so the block-level
// reduction clears the issue's 20% target even though MLP fusion alone (weight-bound) does
// not. The two summands are HBM-traffic counts in the SAME units, so summing them is sound
// (the FLOP-operand intensity in attnCost is a DIFFERENT quantity and is never mixed in here).
func (g PrefillGeometry) BlockHBMTraffic() FusionTraffic {
	a := g.AttentionHBMTraffic()
	m := g.MLPHBMTraffic()
	return makeFusionTraffic("block", a.Unfused+m.Unfused, a.Fused+m.Fused)
}

// MeetsMemoryTrafficTarget is #279's first acceptance bullet in code: a fusion meets the gate
// when it removes at least `target` of the unfused HBM traffic (the issue fixes target = 0.20
// for "20%+ reduction in memory traffic"). A CUDA bench node grades its MEASURED traffic the
// same way rather than re-deriving "20%" by hand, keeping the pass/fail rule in one tested
// place — the analogue of prefill.go's WithinTarget for the throughput gate. A non-positive
// target is treated as no valid gate → false.
func MeetsMemoryTrafficTarget(t FusionTraffic, target float64) bool {
	if target <= 0 {
		return false
	}
	return t.ReductionFrac >= target
}
