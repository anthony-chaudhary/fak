package model

import (
	"encoding/binary"
	"math"
)

// quant_kquant_int8_q6k.go — the int8 decode GEMV for resident Q6_K experts, the Q6_K sibling of
// the Q5_K int8 path in quant_kquant_int8.go (#1002, following the #996 Q5_K landing). On the
// GLM-5.2 cpu-offload hybrid the mixed-quant experts include Q6_K rows (e.g. ffn_down_exps), which
// fell to the slow f32 kQuantMatRowsRange dequant→dot while Q5_K and Q4_K already had int8 paths.
// This moves the Q6_K dot into int8 the same way, so every k-quant expert kind shares the lever.
//
// Q6_K is NON-AFFINE — there is no per-sub-block min, only an int8 scale:
//
//	w[j] = d * sc_g * (q6[j] - 32)      (d per super-block; sc_g the int8 scale of group g)
//
// where q6[j] = (ql nibble) | (qh 2-bit << 4) is the 6-bit weight (0..63). With x[j] = dx_b*qx[j]
// the per-group dot is d*sc_g*dx_b*Σ((q6-32)*qx) = d*sc_g*dx_b*(Σ q6*qx − 32*Σ qx), so the two
// integer reductions are I_g = Σ q6*qx and S_g = Σ qx and the float combine applies the −32
// correction once via (I_g − 32*S_g). This is the only difference from the affine Q5_K combine
// (which subtracts min*Σqx); the SHAPE — one integer reduce per group, one float fold — is the same.
//
// Activation-block alignment (the part #1002 flags as the Q6_K subtlety): a 256-weight super-block
// dequants in two 128-wide chunks, and within a chunk the four output positions n+l+{0,32,64,96}
// land in FOUR distinct 32-wide activation blocks AND each 32-wide block splits across two scales
// (is = l/16). So a super-block has 16 (I,S) groups, group g = chunk*8 + is + pos*2, matching the
// dequant's scale index `scOff + is + pos*2` exactly (q6kDequantSuperBlock). Each group's 16 lanes
// share one activation block, so qx/dx index by the weight position the dequant writes.
//
// Range safety (int32, so the reduction is associative and SIMD-order-free): each group sums 16
// lanes, |I_g| <= 16*63*127 ~= 1.28e5 and |S_g| <= 16*127 ~= 2.0e3 — both far inside int32.
//
// Approximate vs the f32 kQuantMatRowsRange (it adds activation quantization), so it rides the same
// greedy + first-token gate as the Q5_K/Q4_K int8 paths, never the f32 bit-exact rung. The f32
// reference is byte-unchanged; this path is gated by the SAME kQuantSDOTEnabled flag (FAK_KQ_INT8).

// q6kGroupsPerBlock is the number of (scale, activation-block) integer-reduction groups a Q6_K
// super-block splits into: 16 int8 scales, each covering 16 lanes within one 32-wide activation
// block. It is the per-super-block stride into the IS/SS reduction buffers.
const q6kGroupsPerBlock = 16

// q6kReduceRowScalar writes the 16 per-group (I_g = Σ q6*qx, S_g = Σ qx) int32 pairs per
// super-block into IS/SS, reconstructing the 6-bit weight from the ql nibble + the qh 2 bits
// EXACTLY as q6kDequantSuperBlock does (chunk n, lane l, position p∈{0..3}: q6 = (ql>>nibble) |
// ((qh>>2p)&3)<<4, then the −32 is deferred to the combine). The integer reduction is the whole
// correctness story; the float fold is q6kCombineRow.
func q6kReduceRowScalar(row []byte, nblk int, qx []int8, IS, SS []int32) {
	for b := 0; b < nblk; b++ {
		blk := row[b*q6kBlockBytes : (b+1)*q6kBlockBytes]
		ql := blk[0 : qkK/2]
		qh := blk[qkK/2 : qkK/2+qkK/4]
		base := b * q6kGroupsPerBlock
		xBase := b * qkK // this super-block's 256 activations live at qx[b*256 .. b*256+255]
		qlOff, qhOff, scOff := 0, 0, 0
		for n := 0; n < qkK; n += 128 {
			// Each group g accumulates 16 lanes (l in [is*16, is*16+16)); a fresh (I,S) per group.
			for is := 0; is < 2; is++ {
				lo := is * 16
				for p := 0; p < 4; p++ {
					g := base + scOff + is + p*2
					var iAcc, sAcc int32
					for l := lo; l < lo+16; l++ {
						var q6 int32
						switch p {
						case 0:
							q6 = int32((ql[qlOff+l+0] & 0x0f) | (((qh[qhOff+l] >> 0) & 3) << 4))
						case 1:
							q6 = int32((ql[qlOff+l+32] & 0x0f) | (((qh[qhOff+l] >> 2) & 3) << 4))
						case 2:
							q6 = int32((ql[qlOff+l+0] >> 4) | (((qh[qhOff+l] >> 4) & 3) << 4))
						default:
							q6 = int32((ql[qlOff+l+32] >> 4) | (((qh[qhOff+l] >> 6) & 3) << 4))
						}
						xv := int32(qx[xBase+n+l+p*32])
						iAcc += q6 * xv
						sAcc += xv
					}
					IS[g] = iAcc
					SS[g] = sAcc
				}
			}
			qlOff += 64
			qhOff += 32
			scOff += 8
		}
	}
}

// q6kCombineRow folds the int32 (I_g, S_g) reductions to the float dot: per group, d*sc_g applies
// the weight scale and (I_g − 32*S_g) applies the q6−32 zero-point shift once, scaled by the
// group's activation block scale dx. The group→activation-block map mirrors q6kReduceRowScalar:
// group g = b*16 + chunk*8 + is + p*2, and its lanes sit in activation block (chunk*128 + p*32)/32.
func q6kCombineRow(row []byte, nblk int, dx []float32, IS, SS []int32) float32 {
	var acc float32
	for b := 0; b < nblk; b++ {
		blk := row[b*q6kBlockBytes : (b+1)*q6kBlockBytes]
		scales := blk[qkK/2+qkK/4 : qkK/2+qkK/4+qkK/16]
		d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[q6kBlockBytes-2:])))
		base := b * q6kGroupsPerBlock
		for n128 := 0; n128 < 2; n128++ {
			scOff := n128 * 8
			actBlkBase := b*8 + n128*4 // 8 activation blocks per super-block, 4 per 128-chunk
			for is := 0; is < 2; is++ {
				for p := 0; p < 4; p++ {
					g := base + scOff + is + p*2
					ws := d * float32(int8(scales[scOff+is+p*2]))
					// Lanes l in [is*16, is*16+16) of position p sit in activation block
					// actBlkBase+p (the +32-strided output position), so dx indexes there.
					dxg := dx[actBlkBase+p]
					acc += (float32(IS[g]) - 32*float32(SS[g])) * ws * dxg
				}
			}
		}
	}
	return acc
}

// q6kMatRowsRangeInt8 is the int8 GEMV over output rows [lo,hi): one shared activation quantize
// (qv), then per row the integer reduce + float combine. Mirrors q5kMatRowsRangeInt8.
func q6kMatRowsRangeInt8(qt *kQuantTensor, qv q8Vec, y []float32, lo, hi int) {
	ngrp := qt.nblk * q6kGroupsPerBlock
	IS := make([]int32, ngrp)
	SS := make([]int32, ngrp)
	qx := qv.q
	dx := qv.d
	rowBytes := qt.rowBytes()
	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes : (o+1)*rowBytes]
		q6kReduceRow(row, qt.nblk, qx, IS, SS)
		y[o] = q6kCombineRow(row, qt.nblk, dx, IS, SS)
	}
}
