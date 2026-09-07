//go:build amd64

package model

import (
	"encoding/binary"
	"math"
)

// quant_amd64_q2k.go — optimized AMD64 dequantization for resident Q2_K superblocks.
// Bypasses scalar loop overhead by unrolling 16-weight groups with precomputed 4-entry
// lookup tables (0-ml, dl-ml, 2*dl-ml, 3*dl-ml) that map directly to registers.

// q2kDequantSuperBlockArch dequantizes one Q2_K super-block on amd64, bit-identical to
// q2kDequantSuperBlock. It returns false when the resolved tier lacks AVX2.
func q2kDequantSuperBlockArch(dst []float32, blk []byte) bool {
	if qtier < tierAVX2 {
		return false
	}
	scales := blk[:qkK/16]
	q := blk[qkK/16 : qkK/16+qkK/4]
	dm := qkK/16 + qkK/4
	d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[dm:])))
	minVal := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[dm+2:])))

	is := 0
	qi := 0
	for n := 0; n < qkK; n += 128 {
		shift := uint(0)
		for j := 0; j < 4; j++ {
			sc0 := scales[is]
			is++
			dl0, ml0 := d*float32(sc0&0x0f), minVal*float32(sc0>>4)
			t0 := [4]float32{0 - ml0, dl0 - ml0, dl0*2 - ml0, dl0*3 - ml0}

			sc1 := scales[is]
			is++
			dl1, ml1 := d*float32(sc1&0x0f), minVal*float32(sc1>>4)
			t1 := [4]float32{0 - ml1, dl1 - ml1, dl1*2 - ml1, dl1*3 - ml1}

			dst0 := dst[n+j*32 : n+j*32+16]
			dst1 := dst[n+j*32+16 : n+j*32+32]
			q0 := q[qi : qi+16]
			q1 := q[qi+16 : qi+32]

			// Direct unrolled table reads
			dst0[0] = t0[(q0[0]>>shift)&3]
			dst0[1] = t0[(q0[1]>>shift)&3]
			dst0[2] = t0[(q0[2]>>shift)&3]
			dst0[3] = t0[(q0[3]>>shift)&3]
			dst0[4] = t0[(q0[4]>>shift)&3]
			dst0[5] = t0[(q0[5]>>shift)&3]
			dst0[6] = t0[(q0[6]>>shift)&3]
			dst0[7] = t0[(q0[7]>>shift)&3]
			dst0[8] = t0[(q0[8]>>shift)&3]
			dst0[9] = t0[(q0[9]>>shift)&3]
			dst0[10] = t0[(q0[10]>>shift)&3]
			dst0[11] = t0[(q0[11]>>shift)&3]
			dst0[12] = t0[(q0[12]>>shift)&3]
			dst0[13] = t0[(q0[13]>>shift)&3]
			dst0[14] = t0[(q0[14]>>shift)&3]
			dst0[15] = t0[(q0[15]>>shift)&3]

			dst1[0] = t1[(q1[0]>>shift)&3]
			dst1[1] = t1[(q1[1]>>shift)&3]
			dst1[2] = t1[(q1[2]>>shift)&3]
			dst1[3] = t1[(q1[3]>>shift)&3]
			dst1[4] = t1[(q1[4]>>shift)&3]
			dst1[5] = t1[(q1[5]>>shift)&3]
			dst1[6] = t1[(q1[6]>>shift)&3]
			dst1[7] = t1[(q1[7]>>shift)&3]
			dst1[8] = t1[(q1[8]>>shift)&3]
			dst1[9] = t1[(q1[9]>>shift)&3]
			dst1[10] = t1[(q1[10]>>shift)&3]
			dst1[11] = t1[(q1[11]>>shift)&3]
			dst1[12] = t1[(q1[12]>>shift)&3]
			dst1[13] = t1[(q1[13]>>shift)&3]
			dst1[14] = t1[(q1[14]>>shift)&3]
			dst1[15] = t1[(q1[15]>>shift)&3]

			shift += 2
		}
		qi += 32
	}
	return true
}
