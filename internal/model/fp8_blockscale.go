package model

import (
	"fmt"
	"math"
)

// FP8 e4m3 128x128 block-scale dequant-at-load — the DeepSeek-V4 / GLM-5.2-FP8
// checkpoint layout. Those checkpoints store each 2-D weight [O,I] as float8_e4m3fn
// bytes plus a companion `weight_scale_inv` tensor holding ONE f32 scale per 128x128
// tile (shape [ceil(O/128), ceil(I/128)]). Dequant is a per-tile multiply:
//
//	dequant[o,i] = e4m3(weight[o,i]) * scaleInv[o/128, i/128]
//
// i.e. the block scale is repeat_interleaved across its 128x128 tile and the ragged
// final row/col tiles are cropped to O,I. This sits beside decodeMXFP4Blocks
// (safetensors.go): MXFP4 is a 1-D 32-element block layout with an exponent-only
// scale; FP8 here is a 2-D 128x128 block layout with a full f32 scale. The two do
// not share a decode path.
//
// v4quant_admit.go already CLASSIFIES F8_E4M3 tensors (the #3019 admission plan) —
// this is the decode those classified tensors were missing (#4360). Only the pure
// decode math lands here; wiring the `_scale_inv` pairing into the safetensors load
// loop and emitting little-endian f32 bytes is the fenced follow-on (see the
// package's load loop around the `_blocks`/`_scales` MXFP4 pairing).

// fp8BlockDim is the DeepSeek/OCP block edge: weight_scale_inv carries one scale per
// fp8BlockDim x fp8BlockDim tile of the weight.
const fp8BlockDim = 128

// fp8E4M3ToF32 decodes one OCP float8_e4m3fn byte to f32. E4M3FN (the variant torch
// and DeepSeek-V4 use for attention/expert/router weights) is 1 sign / 4 exponent
// (bias 7) / 3 mantissa bits with NO infinities: the sole NaN pattern is S.1111.111
// (0x7F / 0xFF), the max finite magnitude is 448, and exponent field 0 is subnormal
// (value = ±2^-6 * man/8, so 0x00/0x80 are ±0). The decode is exact into f32 — every
// e4m3 value is representable — so its golden test asserts bitwise equality.
func fp8E4M3ToF32(b byte) float32 {
	sign := 1.0
	if b&0x80 != 0 {
		sign = -1.0
	}
	exp := int(b>>3) & 0x0f
	man := int(b) & 0x07
	switch {
	case exp == 0x0f && man == 0x07:
		// S.1111.111 — the single NaN pattern (no infinities in e4m3fn).
		return float32(math.NaN())
	case exp == 0:
		// Subnormal: 2^(1-bias) * man/8, bias=7. man==0 gives signed zero.
		return float32(sign * math.Ldexp(float64(man)/8, 1-7))
	default:
		// Normal: 2^(exp-bias) * (1 + man/8), bias=7.
		return float32(sign * math.Ldexp(1+float64(man)/8, exp-7))
	}
}

// decodeFP8BlockScale dequantizes a row-major [O,I] float8_e4m3fn weight stored with
// 128x128 2-D block scales to a flat row-major [O,I] f32 slice. scaleInv holds one f32
// per 128x128 tile in row-major [ceil(O/128), ceil(I/128)] order — exactly the
// `weight_scale_inv` companion tensor a DeepSeek-V4 / GLM-5.2-FP8 checkpoint ships.
// Ragged final tiles (O or I not a multiple of 128) are cropped: the block index is
// o/128, i/128, so short edge tiles simply cover fewer than 128 rows/cols.
//
// Fail-closed: a weight or scale length that disagrees with the shape is an error, not
// a silent mis-load — the same posture v4quant_admit.go takes on an unclassifiable
// tensor. name is used only for error context.
func decodeFP8BlockScale(name string, O, I int, weight []byte, scaleInv []float32) ([]float32, error) {
	if O <= 0 || I <= 0 {
		return nil, fmt.Errorf("fp8 block-scale %s: shape [%d %d] has a non-positive dimension", name, O, I)
	}
	elems, ok := checkedShapeProduct(O, I)
	if !ok {
		return nil, fmt.Errorf("fp8 block-scale %s: shape [%d %d] overflows element count", name, O, I)
	}
	if len(weight) != elems {
		return nil, fmt.Errorf("fp8 block-scale %s: weight has %d bytes, shape [%d %d] implies %d", name, len(weight), O, I, elems)
	}
	sO := (O + fp8BlockDim - 1) / fp8BlockDim
	sI := (I + fp8BlockDim - 1) / fp8BlockDim
	scaleElems, ok := checkedShapeProduct(sO, sI)
	if !ok {
		return nil, fmt.Errorf("fp8 block-scale %s: scale shape [%d %d] overflows element count", name, sO, sI)
	}
	if len(scaleInv) != scaleElems {
		return nil, fmt.Errorf("fp8 block-scale %s: scaleInv has %d entries, shape [%d %d] (blocks of %d) implies %d", name, len(scaleInv), O, I, fp8BlockDim, scaleElems)
	}
	out := make([]float32, elems)
	for o := 0; o < O; o++ {
		scaleRow := (o / fp8BlockDim) * sI
		for i := 0; i < I; i++ {
			out[o*I+i] = fp8E4M3ToF32(weight[o*I+i]) * scaleInv[scaleRow+i/fp8BlockDim]
		}
	}
	return out, nil
}
