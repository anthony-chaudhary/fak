package compute

import (
	"errors"
	"fmt"
	"math"
)

var (
	// ErrKVFP8Geometry identifies an invalid plane shape or length.
	ErrKVFP8Geometry = errors.New("compute: FP8 KV plane geometry refused")
	// ErrKVFP8NonFinite identifies a non-finite input, code, or decoded value.
	ErrKVFP8NonFinite = errors.New("compute: FP8 KV plane non-finite value")
	// ErrKVFP8Scale identifies a nonpositive or non-finite scale.
	ErrKVFP8Scale = errors.New("compute: FP8 KV plane invalid scale")
)

// KVFP8Plane is the host reference image of one attended K or V plane.
// Codes are [Positions][Heads][HeadDim]; Scales are F32 [Positions][Heads].
// It is not a device allocation, KVStore, snapshot format, or activation claim.
type KVFP8Plane struct {
	Positions int
	Heads     int
	HeadDim   int
	Codes     []byte
	Scales    []float32
}

func kvFP8Shape(positions, heads, headDim int) (rows, elems int, err error) {
	const maxNative = int64(1<<31 - 1)
	if positions < 0 || heads <= 0 || headDim <= 0 ||
		int64(positions) > maxNative || int64(heads) > maxNative || int64(headDim) > maxNative {
		return 0, 0, ErrKVFP8Geometry
	}
	// The same bound is used by the native flattened element ABI. Check before
	// multiplying, including on 32-bit hosts.
	if int64(heads) > maxNative/int64(headDim) {
		return 0, 0, ErrKVFP8Geometry
	}
	width := heads * headDim
	if positions > 0 && int64(positions) > maxNative/int64(width) {
		return 0, 0, ErrKVFP8Geometry
	}
	return positions * heads, positions * width, nil
}

func kvFP8Finite(x float32) bool {
	return math.Float32bits(x)&0x7f800000 != 0x7f800000
}

// QuantizeKVFP8Plane is the correctness oracle for dynamic E4M3FN storage.
// Its row reduction/conversion structure is adapted from vLLM
// fa3b9d1377219247060d3c8b49eae199c9e7e154,
// csrc/quantization/w8a8/fp8/common.cu (Apache-2.0).
//
// Fak intentionally does not use vLLM's min scale 1/(448*512): nonzero
// rows use correctly rounded F32 amax/448, replacing only an underflowed
// zero scale with the smallest positive F32. All-zero rows use scale one.
// Division is not replaced with a rounded reciprocal. The existing
// rfqEncodeE4M3 supplies the tested OCP RNE encoder, not the incompatible
// subnormal-grid encoder used by the separate DSV41 reference.
//
// No input is mutated. Refusal returns a zero result, never partial planes.
func QuantizeKVFP8Plane(src []float32, positions, heads, headDim int) (KVFP8Plane, error) {
	rows, elems, err := kvFP8Shape(positions, heads, headDim)
	if err != nil || len(src) != elems {
		return KVFP8Plane{}, ErrKVFP8Geometry
	}
	// Validate the whole input before allocating output. In the device store,
	// the equivalent numerical scan precedes any resident-plane mutation.
	for i, x := range src {
		if !kvFP8Finite(x) {
			return KVFP8Plane{}, fmt.Errorf("%w: input[%d]", ErrKVFP8NonFinite, i)
		}
	}
	out := KVFP8Plane{Positions: positions, Heads: heads, HeadDim: headDim}
	if elems == 0 {
		return out, nil
	}
	out.Codes = make([]byte, elems)
	out.Scales = make([]float32, rows)
	for row := 0; row < rows; row++ {
		base := row * headDim
		var amax float32
		for _, x := range src[base : base+headDim] {
			magnitude := math.Float32frombits(math.Float32bits(x) & 0x7fffffff)
			if magnitude > amax {
				amax = magnitude
			}
		}
		scale := float32(1)
		if amax != 0 {
			scale = float32(amax / float32(448))
			if scale == 0 {
				scale = math.SmallestNonzeroFloat32
			}
		}
		out.Scales[row] = scale
		for d, x := range src[base : base+headDim] {
			code := rfqEncodeE4M3(float32(x / scale))
			// A maliciously supplied scale is handled by the inverse below;
			// also refuse any unexpected overflow in our own encoded image.
			if !kvFP8Finite(float32(DecodeE4M3(code) * scale)) {
				return KVFP8Plane{}, fmt.Errorf("%w: encoded[%d]", ErrKVFP8NonFinite, base+d)
			}
			out.Codes[base+d] = code
		}
	}
	return out, nil
}

// DequantizeKVFP8Plane exactly widens each E4M3 code, then performs one F32
// scale multiply. Scaling through half/half2 would lose range and precision.
// It validates all externally mutable plane fields before publishing output.
func DequantizeKVFP8Plane(plane KVFP8Plane) ([]float32, error) {
	rows, elems, err := kvFP8Shape(plane.Positions, plane.Heads, plane.HeadDim)
	if err != nil || len(plane.Codes) != elems || len(plane.Scales) != rows {
		return nil, ErrKVFP8Geometry
	}
	for row, scale := range plane.Scales {
		if !(scale > 0) || !kvFP8Finite(scale) {
			return nil, fmt.Errorf("%w: row[%d]", ErrKVFP8Scale, row)
		}
	}
	for i, code := range plane.Codes {
		if code&0x7f == 0x7f ||
			!kvFP8Finite(float32(DecodeE4M3(code)*plane.Scales[i/plane.HeadDim])) {
			return nil, fmt.Errorf("%w: decoded[%d]", ErrKVFP8NonFinite, i)
		}
	}
	if elems == 0 {
		return nil, nil
	}
	out := make([]float32, elems)
	for i, code := range plane.Codes {
		out[i] = float32(DecodeE4M3(code) * plane.Scales[i/plane.HeadDim])
	}
	return out, nil
}
