//go:build amd64

package model

import (
	"encoding/binary"
	"math"
)

// Resident Q4/Q5/Q6 float-block port of internal/ggufload's AVX2 decoder at
// 2fb9f3bc303bcfdeaebcaebb4e9cd61f1d187484, under the same Apache-2.0 repository
// license. Model cannot import ggufload (loader -> model cycle). Like the
// existing resident IQ3 port, reuse its exact unpack/multiply instructions but
// use model's resolved CPU/OS tier and original scalar exceptional fallback.
// The ordinary batch loop still decodes each block once across all tokens.
func init() {
	q4kDequantSuperBlockArch = q4kDequantSuperBlockAVX2
	q5kDequantSuperBlockArch = q5kDequantSuperBlockAVX2
	q6kDequantSuperBlockArch = q6kDequantSuperBlockAVX2
}

//go:noescape
func q5kDequantLoAVX2(dst *float32, ql, qh *byte, scale, min float32, highMask uint32)

//go:noescape
func q5kDequantHiAVX2(dst *float32, ql, qh *byte, scale, min float32, highMask uint32)

//go:noescape
func q6kDequantPos0AVX2(dst *float32, ql, qh *byte, scale float32)

//go:noescape
func q6kDequantPos1AVX2(dst *float32, ql, qh *byte, scale float32)

//go:noescape
func q6kDequantPos2AVX2(dst *float32, ql, qh *byte, scale float32)

//go:noescape
func q6kDequantPos3AVX2(dst *float32, ql, qh *byte, scale float32)

func q5kDequantSuperBlockAVX2(dst []float32, blk []byte) bool {
	if qtier < tierAVX2 || len(dst) < qkK || len(blk) < q5kBlockBytes {
		return false
	}
	db, mb := binary.LittleEndian.Uint16(blk), binary.LittleEndian.Uint16(blk[2:])
	// Preserve scalar NaN/Inf payload/order by declining before any output write.
	// Finite f16 values times <=63 scales and <=31 codes cannot overflow f32.
	if db&0x7c00 == 0x7c00 || mb&0x7c00 == 0x7c00 {
		return false
	}
	d, minVal := math.Float32frombits(F16BitsToF32Bits(db)), math.Float32frombits(F16BitsToF32Bits(mb))
	scales, qh, ql := blk[4:16], blk[16:48], blk[48:q5kBlockBytes]
	qi, is := 0, 0
	u1, u2 := byte(1), byte(2)
	for j := 0; j < qkK; j += 64 {
		sc, m := GetScaleMinK4(is, scales)
		d1, m1 := d*float32(sc), minVal*float32(m)
		sc, m = GetScaleMinK4(is+1, scales)
		d2, m2 := d*float32(sc), minVal*float32(m)
		q5kDequantLoAVX2(&dst[j], &ql[qi], &qh[0], d1, m1, uint32(u1)*0x01010101)
		q5kDequantHiAVX2(&dst[j+32], &ql[qi], &qh[0], d2, m2, uint32(u2)*0x01010101)
		qi += 32
		is += 2
		u1 <<= 2
		u2 <<= 2
	}
	return true
}

func q6kDequantSuperBlockAVX2(dst []float32, blk []byte) bool {
	if qtier < tierAVX2 || len(dst) < qkK || len(blk) < q6kBlockBytes {
		return false
	}
	db := binary.LittleEndian.Uint16(blk[q6kBlockBytes-2:])
	if db&0x7c00 == 0x7c00 {
		return false
	}
	d := math.Float32frombits(F16BitsToF32Bits(db))
	ql, qh, scales := blk[:128], blk[128:192], blk[192:208]
	qlOff, qhOff, scOff := 0, 0, 0
	for n := 0; n < qkK; n += 128 {
		for is := 0; is < 2; is++ {
			q6kDequantPos0AVX2(&dst[n+is*16], &ql[qlOff+is*16], &qh[qhOff+is*16], d*float32(int8(scales[scOff+is])))
			q6kDequantPos1AVX2(&dst[n+32+is*16], &ql[qlOff+32+is*16], &qh[qhOff+is*16], d*float32(int8(scales[scOff+is+2])))
			q6kDequantPos2AVX2(&dst[n+64+is*16], &ql[qlOff+is*16], &qh[qhOff+is*16], d*float32(int8(scales[scOff+is+4])))
			q6kDequantPos3AVX2(&dst[n+96+is*16], &ql[qlOff+32+is*16], &qh[qhOff+is*16], d*float32(int8(scales[scOff+is+6])))
		}
		qlOff += 64
		qhOff += 32
		scOff += 8
	}
	return true
}

//go:noescape
func q4kDequantLoAVX2(dst *float32, q *byte, scale, min float32)

//go:noescape
func q4kDequantHiAVX2(dst *float32, q *byte, scale, min float32)

// q4kDequantSuperBlockAVX2 serves the once-per-block float prefill decoder,
// distinct from the existing fused decode-row ASM or opt-in int8 extraction.
func q4kDequantSuperBlockAVX2(dst []float32, blk []byte) bool {
	if qtier < tierAVX2 || len(dst) < qkK || len(blk) < q4kBlockBytes {
		return false
	}
	db, mb := binary.LittleEndian.Uint16(blk), binary.LittleEndian.Uint16(blk[2:])
	if db&0x7c00 == 0x7c00 || mb&0x7c00 == 0x7c00 {
		return false
	}
	d, minVal := math.Float32frombits(F16BitsToF32Bits(db)), math.Float32frombits(F16BitsToF32Bits(mb))
	scales, q := blk[4:16], blk[16:q4kBlockBytes]
	qi, is := 0, 0
	for j := 0; j < qkK; j += 64 {
		sc, m := GetScaleMinK4(is, scales)
		d1, m1 := d*float32(sc), minVal*float32(m)
		sc, m = GetScaleMinK4(is+1, scales)
		d2, m2 := d*float32(sc), minVal*float32(m)
		q4kDequantLoAVX2(&dst[j], &q[qi], d1, m1)
		q4kDequantHiAVX2(&dst[j+32], &q[qi], d2, m2)
		qi += 32
		is += 2
	}
	return true
}
