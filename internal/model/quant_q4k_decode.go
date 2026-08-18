package model

// quant_q4k_decode.go — the affine-split fast decode path for resident Q4_K batch-1 GEMV.
//
// The exact kernel (quant_amd64_q4k_f32.s) is bit-identical to the scalar reference and is the
// golden GEMV every parity gate anchors to; that discipline forces a 4-lane XMM fold with a
// per-group VEXTRACTF128 and no FMA, so it spends ~9 vector ops per 8 weights. Decode does not
// need bit-identity — only the correct argmax — so this path relaxes it to buy throughput.
//
// The lever is the Q4_K affine form itself. For sub-block s (scale d_s = d·sc, min m_s = min·m):
//
//	y_sb = Σ_s Σ_i (d_s·nib_i - m_s)·x_i = Σ_s [ d_s·(Σ_i nib_i·x_i) - m_s·(Σ_i x_i) ]
//
// Two consequences: (1) the per-weight body collapses to a single FMA (nib·x) — the d-scale and
// the min-subtract move out to per-sub-block granularity (8 per super-block, not 256); and (2)
// Σ_i x_i per sub-block is WEIGHT-ROW-INDEPENDENT, so it is computed once per matmul (q4kDecodeXSum)
// and reused across every one of the `out` output rows. A GEMV re-streams the whole weight matrix
// once, but the activation reduction was being recomputed per row — this hoists it out.
//
// Reassociating the dot changes f32 rounding order, so this is NOT a drop-in for the golden kernel;
// it is selected only for the decode GEMV dispatch and held to a cosine/argmax quality gate
// (TestQ4KDecodeAffineMatchesScalar), never to max|Δ|==0.

// q4kDecodeXSum returns the per-sub-block activation sums for a super-block-aligned x (len nblk*256):
// entry j = Σ x[j*32:(j+1)*32], for j in [0, nblk*8). This is the m_s·Σx term's shared factor,
// computed ONCE per decode matmul and passed to every output-row range (the affine kernel multiplies
// it by the row's per-sub-block min m_s). Sub-block s of super-block b maps to x[(b*8+s)*32:] — the
// same contiguous layout q4kDequantSuperBlock dots against.
func q4kDecodeXSum(x []float32, nblk int) []float32 {
	n := nblk * 8
	xsum := make([]float32, n)
	for j := 0; j < n; j++ {
		xs := x[j*32 : j*32+32]
		var s float32
		for i := 0; i < 32; i++ {
			s += xs[i]
		}
		xsum[j] = s
	}
	return xsum
}
