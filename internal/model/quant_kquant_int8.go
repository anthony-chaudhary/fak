package model

import (
	"encoding/binary"
	"math"
	"os"
)

// quant_kquant_int8.go — the int8 decode GEMV for resident Q5_K experts, the K-quant sibling of
// quant_q4k_int8.go. The scalar-f32 kQuantMatRowsRange (quant_kquant.go) dequants every super-block
// to 256 f32 and dots scalar — COMPUTE-bound exactly like the old resident-Q4_K path was. GLM-5.2's
// MoE experts are MIXED quant (e.g. ffn_gate_exps=Q4_K but ffn_down_exps=Q5_K), so on the cpu-offload
// hybrid the Q5_K experts fell to this slow f32-dequant path while the Q4_K ones already had an int8
// SDOT path — a measured ~1000x decode-throughput gap that is the K-quant scalar dequant, not an
// architecture wall. This moves the Q5_K dot into int8, the same decomposition q4kReduceRow uses.
//
// Affine dequant, identical SHAPE to Q4_K (the combine is literally shared):
//
//	w[j] = d*sc_s*q5[j] - min*m_s            (per 32-wide sub-block s; d,min per super-block)
//
// where q5[j] = (ql nibble) | (qh high-bit << 4) is the 5-bit weight (0..31). Per sub-block the dot
// is d*sc_s*dx_s*Σ(q5*qx) - min*m_s*dx_s*Σ(qx) with x[j]=dx_s*qx[j] — two integer reductions,
// I_s = Σ q5*qx and S_s = Σ qx. Q5_K's 8 sub-blocks are 32-wide = q8Vec's block size, so
// quantizeVecQ8(x) blocks align 1:1 (qv.d[b*8+s] is sub-block s's activation scale), no re-quant.
//
// Range safety (no int32 overflow, so the integer reduction is associative and SIMD-order-free like
// the Q4_K reducer): |I_s| <= 32*31*127 ~= 1.26e5, |S_s| <= 32*127 ~= 4.1e3 — both far inside int32.
//
// Approximate vs the f32 kQuantMatRows (it adds activation quantization), so it rides the same
// greedy + first-token gate the Q8 / Q4_K-int8 paths use, never the f32 bit-exact rung. Default OFF
// (kQuantSDOTForce stays the f32 path) until proven; the f32 kQuantMatRowsRange is byte-unchanged.

// kQuantSDOTForce gates the int8 Q5_K decode path: 0 = default (off — keep the f32 reduction),
// 1 = force-on, -1 = force-off. Tests comparing the int8 path to the f32 reference set it on.
var kQuantSDOTForce int

func setKQuantSDOTForTest(on bool) {
	if on {
		kQuantSDOTForce = 1
	} else {
		kQuantSDOTForce = -1
	}
}

// kQuantSDOTDefault is resolved once from FAK_KQ_INT8: "1"/"on" opts the int8 Q5_K decode path in
// for production (the GLM-5.2 mixed-quant-expert lever). Default OFF — the path is APPROXIMATE
// (activation quantization) and unproven on a real model, so the conservative f32 reduction stays
// the default until a real-weights witness clears it. The test force (kQuantSDOTForce) overrides this.
var kQuantSDOTDefault = func() bool {
	switch os.Getenv("FAK_KQ_INT8") {
	case "1", "on", "true":
		return true
	}
	return false
}()

// kQuantSDOTEnabled reports whether the int8 k-quant decode path runs for this weight kind. Both
// kindQ5K (q5kMatRowsRangeInt8, this file) and kindQ6K (q6kMatRowsRangeInt8, quant_kquant_int8_q6k.go)
// are implemented; any other kind keeps the f32 reduction. The test force wins; otherwise the
// FAK_KQ_INT8 env decides.
func kQuantSDOTEnabled(kind kQuantKind) bool {
	if kind != kindQ5K && kind != kindQ6K {
		return false
	}
	if kQuantSDOTForce != 0 {
		return kQuantSDOTForce > 0
	}
	return kQuantSDOTDefault
}

// q5kReduceRowScalar writes the 8 per-sub-block (I_s = Σ q5*qx, S_s = Σ qx) int32 pairs per
// super-block into IS/SS. q5 reconstructs the 5-bit weight from the ql nibble + the qh high bit,
// matching q5kDequantSuperBlock's bit layout exactly (chunk j=64*c covers sub-blocks 2c,2c+1; the
// qh bit masks advance u1<<=2,u2<<=2 each chunk). The integer reduction is the whole correctness
// story — the float combine is the shared kQuantCombineRow below.
func q5kReduceRowScalar(row []byte, nblk int, qx []int8, IS, SS []int32) {
	for b := 0; b < nblk; b++ {
		blk := row[b*q5kBlockBytes : (b+1)*q5kBlockBytes]
		qh := blk[4+12 : 4+12+qkK/8]
		ql := blk[4+12+qkK/8 : q5kBlockBytes]
		base := b * 8
		qi := 0
		s := 0
		u1, u2 := byte(1), byte(2)
		for j := 0; j < qkK; j += 64 {
			// sub-block s (low nibble of 32 bytes + qh bit u1)
			xLo := qx[(base+s)*32:]
			var iLo, sLo int32
			for l := 0; l < 32; l++ {
				var q5 int32 = int32(ql[qi+l] & 0x0f)
				if qh[l]&u1 != 0 {
					q5 += 16
				}
				xv := int32(xLo[l])
				iLo += q5 * xv
				sLo += xv
			}
			IS[base+s] = iLo
			SS[base+s] = sLo
			// sub-block s+1 (high nibble of the same 32 bytes + qh bit u2)
			xHi := qx[(base+s+1)*32:]
			var iHi, sHi int32
			for l := 0; l < 32; l++ {
				var q5 int32 = int32(ql[qi+l] >> 4)
				if qh[l]&u2 != 0 {
					q5 += 16
				}
				xv := int32(xHi[l])
				iHi += q5 * xv
				sHi += xv
			}
			IS[base+s+1] = iHi
			SS[base+s+1] = sHi
			qi += 32
			s += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
}

// kQuantCombineRow folds the int32 (I_s, S_s) reductions back to the float dot using the per-sub-block
// (scale, min) and the activation scale dx — identical affine math to q4kCombineRow (K-quant shares
// the getScaleMinK4 sub-block scale layout). Shared compiled Go, so the int8 path's float result is
// fixed by the integer reductions alone.
func kQuantCombineRow(row []byte, nblk int, dx []float32, IS, SS []int32) float32 {
	var acc float32
	for b := 0; b < nblk; b++ {
		blk := row[b*q5kBlockBytes : (b+1)*q5kBlockBytes]
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

// q5kMatRowsRangeInt8 is the int8 GEMV over output rows [lo,hi): one shared activation quantize (qv),
// then per row the integer reduce + float combine. Mirrors q4kMatRowsRangeInt8.
func q5kMatRowsRangeInt8(qt *kQuantTensor, qv q8Vec, y []float32, lo, hi int) {
	nsub := qt.nblk * 8
	IS := make([]int32, nsub)
	SS := make([]int32, nsub)
	qx := qv.q
	dx := qv.d
	rowBytes := qt.rowBytes()
	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes : (o+1)*rowBytes]
		q5kReduceRowScalar(row, qt.nblk, qx, IS, SS)
		y[o] = kQuantCombineRow(row, qt.nblk, dx, IS, SS)
	}
}
