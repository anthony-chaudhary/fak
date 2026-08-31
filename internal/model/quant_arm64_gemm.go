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
		parFor(nTiles, currentWorkerCount(), tile)
	}

	// Remainder rows (out % MR): every token, via the matching scalar reference.
	qGemm8CellRect(qt, qp, Y, nTiles*qgemmTileMR, out, 0, P, 4, false)
	// Remainder tokens (P % NR): the tiled rows still need these columns.
	qGemm8CellRect(qt, qp, Y, 0, nTiles*qgemmTileMR, Pmain, P, 4, true)
}
