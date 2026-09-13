package compute

import (
	"fmt"
	"math"
)

// fp8_block32.go - the DeepSeek-V4.1 dense FP8 block-scale reference for the compute HAL.
// V4.1 replaces V4-Flash's 128x128 weight_scale_inv tile with a 32x32 edge, so a single
// 2-D [O,I] float8_e4m3fn weight carries ceil(O/32)*ceil(I/32) companion scales. This is
// the compute-side sibling of internal/model's decodeFP8BlockScale (the 128x128 V4-Flash
// path): model imports compute, not the reverse, so the HAL must own its own block decode.
// Two scale dtypes ship in the wild - a widened f32 `weight_scale_inv` and the raw
// F8_E8M0 exponent byte (one octet per tile, value 2^(b-127)) - so both decodes land here
// as separate, fail-closed functions. No global dequantization: GemvFP8Block32E8M0 decodes
// inline per element so a streaming engine never materializes the full f32 matrix.

// FP8Block32Dim is the DeepSeek-V4.1 dense FP8 block edge: one scale accompanies each
// FP8Block32Dim x FP8Block32Dim tile of a 2-D weight.
const FP8Block32Dim = 32

func fp8Block32ScaleShape(O, I int) (sO, sI int) {
	return (O + FP8Block32Dim - 1) / FP8Block32Dim, (I + FP8Block32Dim - 1) / FP8Block32Dim
}

// fp8Block32Product returns a*b with an overflow guard, so a hostile shape cannot wrap
// into a small positive element count and slip past the length checks below.
func fp8Block32Product(a, b int) (int, bool) {
	if a <= 0 || b <= 0 {
		return 0, false
	}
	p := a * b
	if p/b != a || p <= 0 {
		return 0, false
	}
	return p, true
}

// decodeE8M0Scale decodes one raw F8_E8M0 exponent byte as 2^(b-127). Byte 0xff is the
// E8M0 NaN encoding and is rejected; every other byte yields an exactly representable
// power of two, so the only failure mode is the NaN slot.
func decodeE8M0Scale(b byte) (float32, error) {
	if b == 0xff {
		return 0, fmt.Errorf("e8m0 scale byte 0xff is NaN")
	}
	v := float32(math.Ldexp(1, int(b)-127))
	if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
		return 0, fmt.Errorf("e8m0 scale byte %#02x decodes to a non-finite value", b)
	}
	return v, nil
}

// DecodeFP8Block32 dequantizes a row-major [O,I] float8_e4m3fn weight stored with one f32
// scale per FP8Block32Dim x FP8Block32Dim output/input tile. scaleInv holds one f32 per
// tile in row-major [ceil(O/FP8Block32Dim), ceil(I/FP8Block32Dim)] order - the widened
// `weight_scale_inv` companion tensor a DeepSeek-V4.1 FP8 checkpoint ships. Ragged final
// tiles are cropped by integer division (o/32, i/32), so the same layout covers an
// O or I that is not a multiple of the block edge.
//
// Fail-closed: a non-positive dimension, a weight length that disagrees with [O,I], or a
// scale length that disagrees with the block grid is a named error, never a silent
// mis-load. name is used only for error context.
func DecodeFP8Block32(name string, O, I int, weight []byte, scaleInv []float32) ([]float32, error) {
	if O <= 0 || I <= 0 {
		return nil, fmt.Errorf("fp8 block32 %s: shape [%d %d] has a non-positive dimension", name, O, I)
	}
	elems, ok := fp8Block32Product(O, I)
	if !ok {
		return nil, fmt.Errorf("fp8 block32 %s: shape [%d %d] overflows element count", name, O, I)
	}
	if len(weight) != elems {
		return nil, fmt.Errorf("fp8 block32 %s: weight has %d bytes, shape [%d %d] implies %d", name, len(weight), O, I, elems)
	}
	sO, sI := fp8Block32ScaleShape(O, I)
	scaleElems, ok := fp8Block32Product(sO, sI)
	if !ok {
		return nil, fmt.Errorf("fp8 block32 %s: scale shape [%d %d] overflows element count", name, sO, sI)
	}
	if len(scaleInv) != scaleElems {
		return nil, fmt.Errorf("fp8 block32 %s: scaleInv has %d entries, shape [%d %d] (blocks of %d) implies %d", name, len(scaleInv), O, I, FP8Block32Dim, scaleElems)
	}
	out := make([]float32, elems)
	for o := 0; o < O; o++ {
		scaleRow := (o / FP8Block32Dim) * sI
		for i := 0; i < I; i++ {
			out[o*I+i] = DecodeE4M3(weight[o*I+i]) * scaleInv[scaleRow+i/FP8Block32Dim]
		}
	}
	return out, nil
}

// DecodeFP8Block32E8M0 dequantizes the same row-major [O,I] float8_e4m3fn / 32x32-tile
// layout as DecodeFP8Block32, but the scales are raw F8_E8M0 bytes - one exponent octet
// per tile, decoded as 2^(b-127). A byte of 0xff (E8M0 NaN), a weight byte decoding to
// NaN/Inf (the E4M3 NaN slots 0x7f/0xff), or any non-finite product is refused. All
// metadata and lengths are validated BEFORE the output is allocated, so a malformed pair
// fails before any large allocation.
func DecodeFP8Block32E8M0(name string, O, I int, weight, scales []byte) ([]float32, error) {
	if O <= 0 || I <= 0 {
		return nil, fmt.Errorf("fp8 block32 %s: shape [%d %d] has a non-positive dimension", name, O, I)
	}
	elems, ok := fp8Block32Product(O, I)
	if !ok {
		return nil, fmt.Errorf("fp8 block32 %s: shape [%d %d] overflows element count", name, O, I)
	}
	if len(weight) != elems {
		return nil, fmt.Errorf("fp8 block32 %s: weight has %d bytes, shape [%d %d] implies %d", name, len(weight), O, I, elems)
	}
	sO, sI := fp8Block32ScaleShape(O, I)
	scaleElems, ok := fp8Block32Product(sO, sI)
	if !ok {
		return nil, fmt.Errorf("fp8 block32 %s: scale shape [%d %d] overflows element count", name, sO, sI)
	}
	if len(scales) != scaleElems {
		return nil, fmt.Errorf("fp8 block32 %s: scales has %d bytes, shape [%d %d] (blocks of %d) implies %d", name, len(scales), O, I, FP8Block32Dim, scaleElems)
	}
	scaleF := make([]float32, scaleElems)
	for k, b := range scales {
		v, err := decodeE8M0Scale(b)
		if err != nil {
			return nil, fmt.Errorf("fp8 block32 %s: %w", name, err)
		}
		scaleF[k] = v
	}
	for k, b := range weight {
		d := DecodeE4M3(b)
		if math.IsNaN(float64(d)) || math.IsInf(float64(d), 0) {
			return nil, fmt.Errorf("fp8 block32 %s: weight byte %d (%#02x) decodes to a non-finite value", name, k, b)
		}
	}
	out := make([]float32, elems)
	for o := 0; o < O; o++ {
		scaleRow := (o / FP8Block32Dim) * sI
		for i := 0; i < I; i++ {
			out[o*I+i] = DecodeE4M3(weight[o*I+i]) * scaleF[scaleRow+i/FP8Block32Dim]
		}
	}
	return out, nil
}

// GemvFP8Block32E8M0 is a portable scalar GEMV over a V4.1 E8M0 block-scaled weight:
// y[o] = sum_i decode32(weight[o,i], scale[o/32,i/32]) * x[i]. It decodes each weight
// element inline against its tile scale, never materializing the full dequantized matrix,
// so a memory-bound engine can stream the FP8 weight and the fused scale in one pass.
// Fail-closed: len(x) must equal I, and all weight/scale lengths are validated before y
// is allocated.
func GemvFP8Block32E8M0(name string, O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	if O <= 0 || I <= 0 {
		return nil, fmt.Errorf("fp8 block32 %s: shape [%d %d] has a non-positive dimension", name, O, I)
	}
	elems, ok := fp8Block32Product(O, I)
	if !ok {
		return nil, fmt.Errorf("fp8 block32 %s: shape [%d %d] overflows element count", name, O, I)
	}
	if len(weight) != elems {
		return nil, fmt.Errorf("fp8 block32 %s: weight has %d bytes, shape [%d %d] implies %d", name, len(weight), O, I, elems)
	}
	if len(x) != I {
		return nil, fmt.Errorf("fp8 block32 %s: x has %d entries, shape [%d %d] implies %d", name, len(x), O, I, I)
	}
	sO, sI := fp8Block32ScaleShape(O, I)
	scaleElems, ok := fp8Block32Product(sO, sI)
	if !ok {
		return nil, fmt.Errorf("fp8 block32 %s: scale shape [%d %d] overflows element count", name, sO, sI)
	}
	if len(scales) != scaleElems {
		return nil, fmt.Errorf("fp8 block32 %s: scales has %d bytes, shape [%d %d] (blocks of %d) implies %d", name, len(scales), O, I, FP8Block32Dim, scaleElems)
	}
	scaleF := make([]float32, scaleElems)
	for k, b := range scales {
		v, err := decodeE8M0Scale(b)
		if err != nil {
			return nil, fmt.Errorf("fp8 block32 %s: %w", name, err)
		}
		scaleF[k] = v
	}
	for k, b := range weight {
		d := DecodeE4M3(b)
		if math.IsNaN(float64(d)) || math.IsInf(float64(d), 0) {
			return nil, fmt.Errorf("fp8 block32 %s: weight byte %d (%#02x) decodes to a non-finite value", name, k, b)
		}
	}
	y := make([]float32, O)
	for o := 0; o < O; o++ {
		scaleRow := (o / FP8Block32Dim) * sI
		var acc float32
		for i := 0; i < I; i++ {
			w := DecodeE4M3(weight[o*I+i]) * scaleF[scaleRow+i/FP8Block32Dim]
			acc += w * x[i]
		}
		y[o] = acc
	}
	return y, nil
}
