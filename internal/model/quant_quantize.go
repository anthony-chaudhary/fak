package model

// quant_quantize.go — quantizeRowQ8scalar: the portable, bit-exact reference for activation
// Q8_0 quantization of one row (nblk blocks of 32 floats). It is the single source of truth
// the AVX-512 kernel (quantizeRowAsm512) is held bit-identical to, the non-amd64
// implementation, and the test oracle. The body is the exact per-block math that used to be
// inlined in quantizeBatchPanelInto / quantizeVecQ8: amax over the block, d = amax/127,
// codes = q8round(x/d), and an explicit all-zero codes write when d==0 (so a reused panel
// buffer never leaks a prior call's codes).
//
// Why this exists as a kernel: q8round is a branchy round-half-away-from-zero that the Go
// compiler cannot vectorize, so the activation quantization was a serial ~25 ms slice of Q8
// prefill (FAK_QPROFILE). The AVX-512 twin vectorizes it 16-wide while reproducing q8round
// EXACTLY (truncate-then-inspect-fractional, not the +0.5 trick), so it spends no correctness
// budget — the Q8 logit gate is unchanged.
func quantizeRowQ8scalar(x []float32, q []int8, d []float32, nblk int) {
	for b := 0; b < nblk; b++ {
		blk := x[b*qBlk : b*qBlk+qBlk]
		var amax float32
		for _, v := range blk {
			a := v
			if a < 0 {
				a = -a
			}
			if a > amax {
				amax = a
			}
		}
		dd := amax / 127
		d[b] = dd
		base := b * qBlk
		if dd == 0 {
			for i := 0; i < qBlk; i++ {
				q[base+i] = 0 // reused buffer: clear, don't leak prior codes
			}
			continue
		}
		inv := float32(1.0) / dd
		for i := 0; i < qBlk; i++ {
			q[base+i] = q8round(blk[i] * inv)
		}
	}
}
