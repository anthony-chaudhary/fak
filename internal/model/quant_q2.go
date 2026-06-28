package model

import (
	"math"
)

// quant_q2.go — resident int2 (Q2_0-style) weight path: the aggressive-quantization lever
// for memory-constrained serving (issue #275, B-007). It is the 2-bit sibling of the int4
// path in quant_q4.go, built the same way and to the same correctness discipline — a
// self-contained packing format + a CPU decode GEMV, carrying its own round-trip and
// GEMV-vs-reference witnesses. The proven f32/Q8/Q4 paths are byte-for-byte untouched.
//
// Format. A q2Tensor is [out, in] row-major, in = nblk*qBlk2 (qBlk2 = 32). Each 32-wide
// block is one f32 scale `d` followed by 32 signed 2-bit codes packed four-per-byte (low
// code first), the natural 2-bit generalization of the Q4_0 layout in quant_q2.go's int4
// sibling. A code c (0..3) dequantizes to d*(c-2), so codes cover signed values [-2, 1].
// Per resident weight: (4 + 8)/32 = 0.375 bytes — vs Q8_0's ~1.125 that is a 3× footprint
// reduction, and vs int4's 0.625 a further 1.67×.
//
// Scale choice. d = amax/(half-1) with half = 1<<(nbits-1) — the SAME formula the int4
// block uses (amax/7 at nbits=4) evaluated at nbits=2, giving d = amax/1 = amax. The ±amax
// peaks then reconstruct exactly; like Q4_0's unused -8 slot, the most-negative 2-bit code
// (-2) is never produced by quantization of a magnitude-≤amax block, so the effective code
// set is the ternary {-1, 0, +1}·d. That is the honest cost of 2 bits; the round-trip
// witness pins the per-weight error to one quantum and the GEMV witness pins the kernel.
//
// Correctness discipline. Quantization is lossy; this path carries its own honest gate
// (the round-trip unit test + GEMV agreement with the dequantized reference), never the
// f32 bit-exact rungs. dequantQ2Block is the exact inverse of quantizeQ2Block's packing so
// the round trip is bounded only by 2-bit rounding error.

const qBlk2 = 32

// q2Tensor is a resident int2 weight matrix [out, in], in == nblk*qBlk2. Scales and codes
// are separate per-row slices so a row's codes prefetch alongside its scales (the same
// layout discipline as q8Tensor/q4Tensor).
type q2Tensor struct {
	out, in, nblk int
	d             []float32 // out*nblk per-block f32 scales
	q             []byte    // out*nblk*8 packed 2-bit codes (4 per byte, low code first)
}

// quantizeQ2Block stores the 32 weights of src as one (d, 8 packed bytes) block at dst,
// returning the per-block scale. d = amax (so the largest magnitude maps to code ±1); a
// zero block stays zero. Codes are round(w/d)+2 clamped to [0,3] (signed range [-2,1]).
func quantizeQ2Block(dst []byte, src []float32) float32 {
	var amax float32
	for _, v := range src {
		a := v
		if a < 0 {
			a = -a
		}
		if a > amax {
			amax = a
		}
	}
	if amax == 0 {
		for i := 0; i < qBlk2/4; i++ {
			dst[i] = 0
		}
		return 0
	}
	d := amax // = amax/(half-1), half = 1<<(2-1) = 2
	inv := 1.0 / float64(d)
	for i := 0; i < qBlk2/4; i++ {
		var bits byte
		for j := 0; j < 4; j++ {
			// Round-to-nearest (int() truncates toward zero and would bias every code by up
			// to half a quantum). This runs once at load, so a float64 round is fine.
			c := int(math.Round(float64(src[4*i+j])*inv)) + 2
			if c < 0 {
				c = 0
			} else if c > 3 {
				c = 3
			}
			bits |= byte(c) << (2 * j)
		}
		dst[i] = bits
	}
	return d
}

// dequantQ2Block writes the 32 weights of one (d, 8 packed bytes) block into dst.
func dequantQ2Block(dst []float32, d float32, q []byte) {
	for i := 0; i < qBlk2/4; i++ {
		b := q[i]
		dst[4*i+0] = d * float32(int(b&0x3)-2)
		dst[4*i+1] = d * float32(int((b>>2)&0x3)-2)
		dst[4*i+2] = d * float32(int((b>>4)&0x3)-2)
		dst[4*i+3] = d * float32(int((b>>6)&0x3)-2)
	}
}

// quantizeQ2 builds a resident int2 copy of an [out,in] f32 matrix (in must be a multiple
// of qBlk2). Built once at load; not on the hot path. Row-parallel exactly like quantizeQ4.
func quantizeQ2(w []float32, out, in int) *q2Tensor {
	if in%qBlk2 != 0 {
		// Without this guard a non-multiple-of-32 reduction dim would silently drop the
		// tail; fail loudly like the Q8/Q4 lanes do. Every Qwen3.6 reduction dim is a
		// multiple of 32.
		panic("model: int2 reduction dim not a multiple of 32")
	}
	nblk := in / qBlk2
	quarter := qBlk2 / 4
	qt := &q2Tensor{out: out, in: in, nblk: nblk, d: make([]float32, out*nblk), q: make([]byte, out*nblk*quarter)}
	parFor(out, numWorkers, func(lo, hi int) {
		for o := lo; o < hi; o++ {
			for b := 0; b < nblk; b++ {
				off := o*in + b*qBlk2
				qt.d[o*nblk+b] = quantizeQ2Block(qt.q[o*nblk*quarter+b*quarter:], w[off:off+qBlk2])
			}
		}
	})
	return qt
}

// dequantQ2Tensor reconstructs the full [out,in] f32 matrix from the int2 codes. Used by
// the witness (GEMV-vs-reference) and by any consumer that wants the dense weights back;
// it is the whole-tensor inverse of quantizeQ2.
func dequantQ2Tensor(qt *q2Tensor) []float32 {
	w := make([]float32, qt.out*qt.in)
	quarter := qBlk2 / 4
	parFor(qt.out, numWorkers, func(lo, hi int) {
		for o := lo; o < hi; o++ {
			for b := 0; b < qt.nblk; b++ {
				dequantQ2Block(w[o*qt.in+b*qBlk2:], qt.d[o*qt.nblk+b], qt.q[o*qt.nblk*quarter+b*quarter:])
			}
		}
	})
	return w
}

// footprintBytes is the resident size of the int2 tensor: 8 code bytes + one f32 scale per
// 32-wide block. Exposed so the memory-reduction witness can compare it against the Q4/Q8
// footprints without reaching into the unexported fields.
func (qt *q2Tensor) footprintBytes() int {
	return len(qt.q) + 4*len(qt.d)
}

// q2MatRows is the int2 decode GEMV: y[o] = dot(row o, x). Row-parallel exactly like
// q4MatRows/qMatRows — decode is memory-bound, so spreading rows across cores taps
// aggregate bandwidth; with int2 codes each core streams ~3× fewer bytes than the Q8 GEMV.
// The dequant of each block lands in a tiny L1-resident scratch before the fdot, so the
// bandwidth-dominant stream is the 8 code bytes/block (the f32 scale is 1/8 of that).
func q2MatRows(qt *q2Tensor, x []float32) []float32 {
	y := make([]float32, qt.out)
	q2MatRowsInto(qt, x, y)
	return y
}

func q2MatRowsInto(qt *q2Tensor, x, y []float32) {
	y = y[:qt.out]
	if numWorkers <= 1 || qt.out*qt.in < parThreshold {
		q2MatRowsRange(qt, x, y, 0, qt.out)
		return
	}
	parFor(qt.out, numWorkers, func(lo, hi int) { q2MatRowsRange(qt, x, y, lo, hi) })
}

func q2MatRowsRange(qt *q2Tensor, x, y []float32, lo, hi int) {
	blk := make([]float32, qBlk2)
	nblk := qt.nblk
	quarter := qBlk2 / 4
	for o := lo; o < hi; o++ {
		qrow := qt.q[o*nblk*quarter:]
		drow := qt.d[o*nblk:]
		var s0, s1, s2, s3 float32
		for b := 0; b < nblk; b++ {
			dequantQ2Block(blk, drow[b], qrow[b*quarter:])
			xs := x[b*qBlk2:]
			// fdot over the 32-wide block, 8 accumulators (matches the q4MatRowsRange ILP
			// pattern; combined in fixed order so the result is deterministic).
			s0 += blk[0]*xs[0] + blk[1]*xs[1] + blk[2]*xs[2] + blk[3]*xs[3]
			s1 += blk[4]*xs[4] + blk[5]*xs[5] + blk[6]*xs[6] + blk[7]*xs[7]
			s2 += blk[8]*xs[8] + blk[9]*xs[9] + blk[10]*xs[10] + blk[11]*xs[11]
			s3 += blk[12]*xs[12] + blk[13]*xs[13] + blk[14]*xs[14] + blk[15]*xs[15]
			s0 += blk[16]*xs[16] + blk[17]*xs[17] + blk[18]*xs[18] + blk[19]*xs[19]
			s1 += blk[20]*xs[20] + blk[21]*xs[21] + blk[22]*xs[22] + blk[23]*xs[23]
			s2 += blk[24]*xs[24] + blk[25]*xs[25] + blk[26]*xs[26] + blk[27]*xs[27]
			s3 += blk[28]*xs[28] + blk[29]*xs[29] + blk[30]*xs[30] + blk[31]*xs[31]
		}
		y[o] = ((s0 + s1) + (s2 + s3))
	}
}
