package model

import (
	"encoding/binary"
	"math"
)

// q4kSDOTForce is a test hook that overrides the int8 SDOT decode gate: 0 = default (neonDot on
// arm64, i.e. the P2 decode acceleration; false elsewhere), 1 = force-on, -1 = force-off. Tests
// that compare the f32 reduction order against an f32 reference set it to -1 (via setQ4KSDOTForTest)
// so the activation-quantization noise of the int8 path does not muddy a dispatch/wiring check.
// Production code never touches it; the value is 0 at every real decode.
var q4kSDOTForce int

// setQ4KSDOTForTest is the test-only setter for q4kSDOTForce (the runtime gate is otherwise
// decided once at init from FAK_QKERNEL). Pair with t.Cleanup to restore the default.
func setQ4KSDOTForTest(on bool) {
	if on {
		q4kSDOTForce = 1
	} else {
		q4kSDOTForce = -1
	}
}

// quant_q4k_int8.go — the int8 SDOT decode GEMV for resident Q4_K (plan P2). The scalar-f32
// q4kMatRowsRange (quant_q4k.go) dequants every super-block to 256 f32 and dots scalar — that
// is COMPUTE-bound (bit-unpack + 256 f32 FMA per super-block per token), which is why the
// resident-q4k decode sat at ~0.3 tok/s despite streaming only 0.5625 B/weight. This path moves
// the dot into int8: the Q4_K nibbles (0..15) pair with an int8-quantized activation and the
// inner product becomes a pure int8xint8->int32 reduction (SDOT on arm64, the same shape
// llama.cpp's vec_dot_q4_K_q8_K takes), so the kernel spends compute proportional to the
// compact byte stream, not a 256-wide f32 expansion.
//
// Affine dequant. Unlike Q8_0 (pure scale, w = d*q), Q4_K is AFFINE:
//
//	w[j] = d*sc_s*nibble[j] - min*m_s        (per 32-wide sub-block s; d,min per super-block)
//
// so per sub-block the dot is
//
//	dot_s = d*sc_s*dx_s * Σ(nibble*qx) - min*m_s*dx_s * Σ(qx)
//
// with x[j] = dx_s*qx[j]. That is TWO integer reductions per sub-block — I_s = Σ nibble*qx
// (SDOT) and S_s = Σ qx (a lane-sum) for the min term. Q4_K's 8 sub-blocks are 32-wide, which
// is exactly q8Vec's block size, so quantizeVecQ8(x) produces qx/dx whose blocks align 1:1 with
// the Q4_K sub-blocks (qv.d[b*8+s] is sub-block s's activation scale). No re-quantization, no
// padding — the geometries agree by construction.
//
// Bit-identity discipline (the key simplification). The float COMBINE is kept in shared Go
// (q4kCombineRow below) — it is NOT in asm. The arch-dispatched piece is the integer REDUCTION
// only (q4kReduceRow: asm on arm64, scalar elsewhere). Because the combine is the same compiled
// Go code on every arch, the full asm-path dot equals the scalar-int8-path dot EXACTLY whenever
// the integer reductions match — and they always match, because int8xint8->int32 SDOT is
// associative with no overflow here (|I_s| <= 32*15*127 ~= 6.1e4, |S_s| <= 32*127 ~= 4.1e3,
// both far inside int32). So there is no FMA-fusion matching to chase: TestQ4KReduceAsmMatchesScalar
// holds the reducer bit-identical on int32, and the float path is identical by construction.
//
// This int8 path is APPROXIMATE vs the f32 q4kMatRows (it adds activation quantization), so it
// carries the q4_k_m greedy + first-token gate, NOT the f32 bit-exact rung — the same standard
// the Q8 path and llama.cpp's q4_k_m are held to.

// q4kReduceRowScalar is the portable integer-reduction reference: for each of nblk super-blocks
// it writes the 8 per-sub-block (I_s = Σ nibble*qx, S_s = Σ qx) int32 pairs into IS/SS. It is the
// oracle the arm64 SDOT kernel is held bit-identical to (TestQ4KReduceAsmMatchesScalar). Integer
// addition is associative with no overflow on these ranges, so any SIMD lane order produces the
// same int32 values; the float combine downstream is shared Go, so this bit-identity is the whole
// correctness story for the asm path.
//
// Sub-block layout within a super-block (matches q4kDequantSuperBlock): the 128-byte q field is
// 4 chunks of 32 bytes; chunk k encodes sub-blocks 2k (the LOW nibble of each of 32 bytes) and
// 2k+1 (the HIGH nibble). Sub-block s therefore reads getScaleMinK4(s) in the combine and
// activation block (b*8+s) here.
func q4kReduceRowScalar(row []byte, nblk int, qx []int8, IS, SS []int32) {
	for b := 0; b < nblk; b++ {
		blk := row[b*q4kBlockBytes : (b+1)*q4kBlockBytes]
		q := blk[16:q4kBlockBytes]
		base := b * 8
		qi := 0
		for k := 0; k < 4; k++ {
			// Sub-block 2k: 32 low nibbles of q[qi:qi+32], activation qx[(base+2k)*32:].
			qxA := qx[(base+2*k)*32:]
			var IA, SA int32
			for l := 0; l < 32; l++ {
				IA += int32(q[qi+l]&0x0f) * int32(qxA[l])
				SA += int32(qxA[l])
			}
			IS[base+2*k] = IA
			SS[base+2*k] = SA
			// Sub-block 2k+1: 32 high nibbles (>>4), activation qx[(base+2k+1)*32:].
			qxB := qx[(base+2*k+1)*32:]
			var IB, SB int32
			for l := 0; l < 32; l++ {
				IB += int32(q[qi+l]>>4) * int32(qxB[l])
				SB += int32(qxB[l])
			}
			IS[base+2*k+1] = IB
			SS[base+2*k+1] = SB
			qi += 32
		}
	}
}

// q4kCombineRow folds the integer reductions back into a float dot using the Q4_K scales. This is
// SHARED Go — not arch-dispatched — so the asm and scalar int8 paths both run it verbatim and
// produce identical floats (given identical IS/SS). Per sub-block s of super-block b:
//
//	acc += f32(I_s) * (d*sc_s) * dx[b*8+s] - f32(S_s) * (min*m_s) * dx[b*8+s]
//
// accumulated sub-block 0..7 within each super-block and super-block 0..nblk-1 across the row,
// in that fixed order. dx is q8Vec's per-32-block activation scale, so dx[b*8+s] is sub-block s's.
func q4kCombineRow(row []byte, nblk int, dx []float32, IS, SS []int32) float32 {
	var acc float32
	for b := 0; b < nblk; b++ {
		blk := row[b*q4kBlockBytes : (b+1)*q4kBlockBytes]
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(blk[0:])))
		min := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(blk[2:])))
		scales := blk[4 : 4+12]
		base := b * 8
		for s := 0; s < 8; s++ {
			sc, m := getScaleMinK4(s, scales)
			ws, wm := d*float32(sc), min*float32(m)
			dxs := dx[base+s]
			acc += (float32(IS[base+s]) * ws) * dxs
			acc -= (float32(SS[base+s]) * wm) * dxs
		}
	}
	return acc
}

// q4kMatRowsRangeInt8 is the int8 SDOT decode GEMV row loop. qv is the once-quantized activation
// (shared across all rows — this is a GEMV), IS/SS are per-worker int32 scratch sized nblk*8.
// Each row: arch-dispatched integer reduction, then the shared float combine.
func q4kMatRowsRangeInt8(qt *q4kTensor, qv q8Vec, y []float32, lo, hi int) {
	nblk := qt.nblk
	IS := make([]int32, nblk*8)
	SS := make([]int32, nblk*8)
	rowBytes := qt.q4kRowBytes()
	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes:]
		q4kReduceRow(row, nblk, qv.q, IS, SS)
		y[o] = q4kCombineRow(row, nblk, qv.d, IS, SS)
	}
}
