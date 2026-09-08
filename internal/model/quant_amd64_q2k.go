//go:build amd64

package model

import (
	"encoding/binary"
	"math"
)

// quant_amd64_q2k.go owns the amd64 Q2_K dequantization ladder. Go constructs the
// 16 lookup tables with exactly the scalar operation order; assembly only unpacks
// two-bit indices and permutes those already-rounded float32 values. Keeping all
// floating-point arithmetic here makes the SIMD result bit-identical for signed
// zero, infinities, and NaN payloads as well as ordinary finite values.

//go:noescape
func q2kDequantSuperBlockAsmAVX2(dst *float32, q *byte, tables *float32)

//go:noescape
func q2kDequantSuperBlockAsmAVX512(dst *float32, q *byte, tables *float32)

// q2kDequantSuperBlockArch dequantizes one Q2_K super-block on amd64. It returns
// false when the resolved quant tier lacks AVX2 so the caller can use the scalar
// reference.
func q2kDequantSuperBlockArch(dst []float32, blk []byte) bool {
	if qtier < tierAVX2 {
		return false
	}

	dm := qkK/16 + qkK/4
	d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[dm:])))
	minVal := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[dm+2:])))

	var tables [qkK / 16][4]float32
	for i, sc := range blk[:qkK/16] {
		dl, ml := d*float32(sc&0x0f), minVal*float32(sc>>4)
		tables[i] = [4]float32{
			0 - ml,
			dl - ml,
			dl*2 - ml,
			dl*3 - ml,
		}
	}

	q := &blk[qkK/16]
	if qtier >= tierAVX512 {
		q2kDequantSuperBlockAsmAVX512(&dst[0], q, &tables[0][0])
	} else {
		q2kDequantSuperBlockAsmAVX2(&dst[0], q, &tables[0][0])
	}
	return true
}
