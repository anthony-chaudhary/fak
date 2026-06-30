//go:build arm64 && !(fakaccel && darwin && cgo)

package model

// quant_arm64_gemm.go — Go side of the arm64 NEON deferred-reduction Q8_0 kernels
// (quant_arm64_gemm.s). qmatrows4NEON is the decode GEMV micro-kernel (4 weight rows × 1
// activation); qgemm8tileNEON is the prefill GEMM micro-kernel (4 rows × 4 tokens). Both are
// bit-identical to qgemm8cell(...,4), so they slot under the same deferred-reduction reference the
// AVX-512 tile kernel uses — the Q8 path's authoritative gate (argmax-exact vs the HF oracle) is
// unchanged.

import "os"

// armUseTile opts the arm64 prefill GEMM into the register-blocked tile kernel (qGemm8TileInto).
// Default off: the per-cell SDOT sweep is faster on Apple Silicon (see qGemm8Into). Provided for
// A/B and for non-Apple arm64 parts. FAK_ARM_TILE=1 enables it.
var armUseTile = os.Getenv("FAK_ARM_TILE") == "1"

//go:noescape
func qmatrows4NEON(qw, qx *int8, dw, dx *float32, in, nblk int, y *float32)

//go:noescape
func qgemm8tileNEON(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)

const qgemmTileMR = 4 // tile rows
const qgemmTileNR = 4 // tile tokens

// qGemm8TileInto is the arm64 register-blocked Q8_0 prefill GEMM: the 4×4 NEON deferred-reduction
// tile kernel over the full MR×NR tiles, with the row remainder (out%4) and token remainder (P%4)
// computed by the matching scalar reference qgemm8cell(...,4). Output row-major [P, out], every
// cell bit-identical to qgemm8cell(...,4). Mirrors the AVX-512 qGemm8Into structure.
func qGemm8TileInto(qt *q8Tensor, qp *q8Panel, Y []float32) {
	out, in, nblk, P := qt.out, qt.in, qt.nblk, qp.P
	Pmain := P &^ (qgemmTileNR - 1) // tokens handled by the NR=4 tile
	nTiles := out / qgemmTileMR

	tile := func(lo, hi int) {
		for tt := lo; tt < hi; tt++ {
			o := tt * qgemmTileMR
			for t := 0; t < Pmain; t += qgemmTileNR {
				qgemm8tileNEON(
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
	for o := nTiles * qgemmTileMR; o < out; o++ {
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
		for o := 0; o < nTiles*qgemmTileMR; o++ {
			Y[t*out+o] = qgemm8cell(qt.q[o*in:o*in+in], qt.d[o*nblk:o*nblk+nblk], qx, dx, nblk, 4)
		}
	}
}

// qMatRows4NEON computes y[lo:hi] for the decode GEMV: four output rows at a time through the NEON
// micro-kernel, the [lo,hi) row remainder (hi-lo not a multiple of 4) through the matching scalar
// reference qgemm8cell(...,4) so the whole range uses the same deferred-reduction order. Reduction
// order is qgemm8cell's, NOT qdot8scalar's — the faster, more-accurate order the AVX-512 GEMV
// already ships (gated by the Q8 oracle tests), so quant decode now matches quant prefill exactly
// (both route through qgemm8cell(...,4)).
func qMatRows4NEON(qt *q8Tensor, qv q8Vec, y []float32, lo, hi int) {
	in, nblk := qt.in, qt.nblk
	o := lo
	for ; o+4 <= hi; o += 4 {
		qmatrows4NEON(&qt.q[o*in], &qv.q[0], &qt.d[o*nblk], &qv.d[0], in, nblk, &y[o])
	}
	for ; o < hi; o++ {
		y[o] = qgemm8cell(qt.q[o*in:o*in+in], qt.d[o*nblk:o*nblk+nblk], qv.q, qv.d, nblk, 4)
	}
}
