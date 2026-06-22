//go:build arm64

package model

// quant_arm64_tile4x2.go — Go side of qgemm8tile4x2NEON (quant_arm64.s), the conservative NR=2
// register-blocked Q8_0 prefill GEMM tile authored for issue #478. It is the smaller-token-block
// companion to the NR=4 qgemm8tileNEON (quant_arm64_gemm.go): the 4×4 tile fills all 16 4-lane
// float accumulators, the maximum the one-accumulator-per-cell deferred-reduction design admits in
// the 32-register NEON file, so a *larger* same-scheme tile is infeasible. The 4×2 tile instead
// trades token-block width (2 instead of 4) for register headroom, while keeping the same
// weight-block-streamed-once amortization (each 32-wide weight block SDOT'd against both tokens,
// the float reduction deferred to the end of the K loop). Both tiles are bit-identical to
// qgemm8cell(...,4), so they share the existing Q8 oracle gate and pin to the lane-4 reference.
//
// Reachable via qGemm8Tile4x2Into for A/B (BenchmarkPrefillTileVsCell478) and pinned by
// TestQGemm8Tile4x2NEONMatchesCell / TestQGemm8Tile4x2IntoMatchesScalar. The SHIPPED default
// prefill path (qGemm8Into) is unchanged — on Apple Silicon the per-cell/2×4 sweep measured fastest
// (see qGemm8Into's doc comment), so neither register tile is the default; they are opt-in levers
// for non-Apple arm64 and for the on-arm64 acceptance measure (tools/run_478_acceptance_on_arm64.sh).

//go:noescape
func qgemm8tile4x2NEON(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)

const qgemmTile42MR = 4 // tile rows
const qgemmTile42NR = 2 // tile tokens

// qGemm8Tile4x2Into is the 4×2 NEON tile dispatcher: the qgemm8tile4x2NEON micro-kernel over the
// full MR×NR tiles, with the row remainder (out%4) and token remainder (P%2) computed by the
// matching scalar reference qgemm8cell(...,4). Output row-major [P, out], every cell bit-identical
// to qgemm8cell(...,4). Same structure as qGemm8TileInto (the NR=4 dispatcher), only the token
// block width differs.
func qGemm8Tile4x2Into(qt *q8Tensor, qp *q8Panel, Y []float32) {
	out, in, nblk, P := qt.out, qt.in, qt.nblk, qp.P
	Pmain := P &^ (qgemmTile42NR - 1) // tokens handled by the NR=2 tile
	nTiles := out / qgemmTile42MR

	tile := func(lo, hi int) {
		for tt := lo; tt < hi; tt++ {
			o := tt * qgemmTile42MR
			for t := 0; t < Pmain; t += qgemmTile42NR {
				qgemm8tile4x2NEON(
					&qt.q[o*in], &qp.q[t*in],
					&qt.d[o*nblk], &qp.d[t*nblk],
					in, nblk, out, &Y[t*out+o],
				)
			}
		}
	}
	if out*in*P < parThreshold {
		tile(0, nTiles)
	} else {
		parFor(nTiles, numWorkers, tile)
	}

	// Remainder rows (out % MR): every token, via the matching scalar reference.
	for o := nTiles * qgemmTile42MR; o < out; o++ {
		qw := qt.q[o*in : o*in+in]
		dw := qt.d[o*nblk : o*nblk+nblk]
		for t := 0; t < P; t++ {
			Y[t*out+o] = qgemm8cell(qw, dw, qp.q[t*in:t*in+in], qp.d[t*nblk:t*nblk+nblk], nblk, 4)
		}
	}
	// Remainder tokens (P % NR): the tiled rows still need these columns.
	for t := Pmain; t < P; t++ {
		qx := qp.q[t*in : t*in+in]
		dx := qp.d[t*nblk : t*nblk+nblk]
		for o := 0; o < nTiles*qgemmTile42MR; o++ {
			Y[t*out+o] = qgemm8cell(qt.q[o*in:o*in+in], qt.d[o*nblk:o*nblk+nblk], qx, dx, nblk, 4)
		}
	}
}
