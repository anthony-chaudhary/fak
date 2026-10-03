package model

import (
	"encoding/binary"
	"math"
)

// quant_kquant_int8_q2k.go — the int8 decode GEMV for resident Q2_K experts, the Q2_K sibling of
// the Q5_K (quant_kquant_int8.go) and Q6_K (quant_kquant_int8_q6k.go) int8 paths.
//
// Prior-art: llama.cpp ggml_vec_dot_q2_K_q8_K and dequantize_row_q2_K at
// ggml-org/llama.cpp commit 83078fec0db82d6b5a00d9599062c38c39145755 (MIT), source
// ggml/src/ggml-cpu/arch/x86/quants.c (the AVX2 branch) with the byte layout from
// ggml/src/ggml-quants.c dequantize_row_q2_K. Copyright (c) 2023-2026 The ggml authors.
// This is an ADAPT: the integer reduction is rewritten in portable Go over the SAME GGUF bytes —
// no upstream code is copied verbatim — and the float combine is shared Go (q2kCombineRow).
//
// Integer Q2_K x Q8_K reduction adapted from llama.cpp ggml_vec_dot_q2_K_q8_K
// (ggml/src/ggml-cpu/arch/x86/quants.c) and dequantize_row_q2_K (ggml/src/ggml-quants.c) at
// 83078fec0db82d6b5a00d9599062c38c39145755 (MIT).
//
// Q2_K is AFFINE, like Q4_K/Q5_K but with 16-wide sub-blocks (not 32):
//
//	w[j] = d*sc_s*q2[j] - min*m_s        (per 16-weight sub-block s; d,min per super-block)
//
// where q2[j] = (code byte >> 2*j) & 3 is the 2-bit weight (0..3) and d = f16 at blk[80:82],
// min = f16 at blk[82:84]. With x[j] = dx_b*qx[j] the per-sub-block dot is
//
//	d*sc_s*dx_s*Σ(q2*qx) - min*m_s*dx_s*Σ(qx)
//
// so the two integer reductions are I_s = Σ q2*qx and S_s = Σ qx — the same SHAPE as the Q4_K/Q5_K
// reducer, only 16-wide and with the sub-block→activation-block map below. The float combine is
// q2kCombineRow.
//
// Activation-block alignment (the Q2_K subtlety): a Q2_K super-block has 16 sub-blocks of 16
// weights. A q8Vec block is 32 weights wide, so TWO adjacent 16-wide sub-blocks share one
// activation scale. Within super-block b, sub-block s covers relative positions
// (s/8)*128 + ((s%8)/2)*32 + (s%2)*16 .. +15, i.e. activation block (s/8)*4 + (s%8)/2; the
// absolute activation block is b*8 + (s/8)*4 + (s%8)/2 (8 blocks per super-block, 4 per 128-chunk).
//
// Range safety (int32, so the reduction is associative and SIMD-order-free): each 16-wide sub-block
// sums at most |I_s| <= 16*3*127 ~= 6.1e3 and |S_s| <= 16*127 = 2.0e3 — far inside int32.
//
// Approximate vs the f32 kQuantMatRowsRange (it adds activation quantization), so it rides the same
// kQuantSDOTEnabled gate as the Q5_K/Q6_K/CIQ int8 paths, default OFF (FAK_KQ_INT8) until real-weight
// qualification. The f32 kQuantMatRowsRange is byte-unchanged and remains the conservative floor.

// q2kGroupsPerBlock is the number of (scale, activation-block) integer-reduction groups a Q2_K
// super-block splits into: 16 int8 scales, each covering 16 lanes (one sub-block). It is the
// per-super-block stride into the IS/SS reduction buffers.
const q2kGroupsPerBlock = 16

// q2kReduceRow is the arch-dispatch seam for the Q2_K integer reduction; its definition now lives in
// the arch files (quant_kquant_int8_q2k_amd64.go on amd64, quant_kquant_int8_q2k_noamd64.go
// elsewhere), exactly as q5kReduceRow/q6kReduceRow do. The amd64 build routes to the AVX2/VNNI
// kernel; every other arch takes q2kReduceRowScalar. This named indirection keeps the portable
// reference (below) untouched by the hot loop.

// q2kReduceRowScalar is the portable integer-reduction reference: for each of nblk super-blocks it
// writes the 16 per-sub-block (I_s = Σ q2*qx, S_s = Σ qx) int32 pairs into IS/SS. The 2-bit code
// layout mirrors q2kDequantSuperBlockScalar exactly — a 128-weight chunk drives four shift values
// (shift = 0,2,4,6), each over 32 consecutive weights, and within a shift the first 16 weights read
// `q[qi+l]` while the next 16 read `q[qi+16+l]`. Integer addition is associative with no overflow on
// these ranges, so any SIMD lane order produces the same int32 values; the float fold downstream is
// shared Go (q2kCombineRow), so this bit-identity is the whole correctness story for the path.
func q2kReduceRowScalar(row []byte, nblk int, qx []int8, IS, SS []int32) {
	for b := 0; b < nblk; b++ {
		blk := row[b*q2kBlockBytes : (b+1)*q2kBlockBytes]
		q := blk[qkK/16 : qkK/16+qkK/4]
		base := b * q2kGroupsPerBlock
		xBase := b * qkK // this super-block's 256 activations live at qx[b*256 .. b*256+255]
		qi := 0
		for n := 0; n < qkK; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				s0 := base + (n/128)*8 + 2*j
				var i0, s0Sum int32
				for l := 0; l < 16; l++ {
					code := int32((q[qi+l] >> shift) & 3)
					xv := int32(qx[xBase+n+j*32+l])
					i0 += code * xv
					s0Sum += xv
				}
				IS[s0] = i0
				SS[s0] = s0Sum

				s1 := s0 + 1
				var i1, s1Sum int32
				for l := 0; l < 16; l++ {
					code := int32((q[qi+16+l] >> shift) & 3)
					xv := int32(qx[xBase+n+j*32+16+l])
					i1 += code * xv
					s1Sum += xv
				}
				IS[s1] = i1
				SS[s1] = s1Sum
				shift += 2
			}
			qi += 32
		}
	}
}

// q2kCombineRow folds the int32 (I_s, S_s) reductions back to the float dot using the per-sub-block
// (scale, min) and the per-32 activation scale dx — identical affine math to q5kCombineRow but with
// 16-wide sub-blocks and the Q2_K activation-block map. Per sub-block s of super-block b:
//
//	acc += f32(I_s) * (d*(sc&0xf)) * dx[b*8+(s/8)*4+(s%8)/2]
//	acc -= f32(S_s) * (min*(sc>>4)) * dx[b*8+(s/8)*4+(s%8)/2]
//
// accumulated sub-block 0..15 within each super-block and super-block 0..nblk-1 across the row, in
// that fixed order. It is SHARED Go, so the int8 path's float result is fixed by the reductions alone.
func q2kCombineRow(row []byte, nblk int, dx []float32, IS, SS []int32) float32 {
	var acc float32
	for b := 0; b < nblk; b++ {
		blk := row[b*q2kBlockBytes : (b+1)*q2kBlockBytes]
		scales := blk[:qkK/16]
		d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[qkK/16+qkK/4:])))
		min := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[qkK/16+qkK/4+2:])))
		base := b * q2kGroupsPerBlock
		dxBase := b * 8 // 8 activation blocks per super-block
		for s := 0; s < q2kGroupsPerBlock; s++ {
			sc := scales[s]
			ws, wm := d*float32(sc&0x0f), min*float32(sc>>4)
			dxs := dx[dxBase+(s/8)*4+(s%8)/2]
			acc += (float32(IS[base+s]) * ws) * dxs
			acc -= (float32(SS[base+s]) * wm) * dxs
		}
	}
	return acc
}

// q2kMatRowsRangeInt8 is the int8 GEMV over output rows [lo,hi): one shared activation quantize
// (qv), then per row the integer reduce + float combine. Mirrors q5kMatRowsRangeInt8.
func q2kMatRowsRangeInt8(qt *kQuantTensor, qv q8Vec, y []float32, lo, hi int) {
	q2kMatRowsRangeInt8Raw(qt.raw, qt, qv, y, lo, hi)
}

func q2kMatRowsRangeInt8Raw(raw []byte, qt *kQuantTensor, qv q8Vec, y []float32, lo, hi int) {
	if len(raw) == 0 {
		raw = qt.raw
	}
	nsub := qt.nblk * q2kGroupsPerBlock
	IS := make([]int32, nsub)
	SS := make([]int32, nsub)
	qx := qv.q
	dx := qv.d
	rowBytes := qt.rowBytes()
	for o := lo; o < hi; o++ {
		row := raw[o*rowBytes : (o+1)*rowBytes]
		q2kReduceRow(row, qt.nblk, qx, IS, SS)
		y[o] = q2kCombineRow(row, qt.nblk, dx, IS, SS)
	}
}
