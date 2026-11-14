package ggufload

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// MXFP4 source-to-GGUF repack: the writer half of the seam #489 opened on the
// reader side. A released MXFP4 checkpoint stores each 32-element block as 16
// packed I8 code bytes plus a separate E8M0 scale byte; source byte j holds
// element 2j in its low nibble and element 2j+1 in its high nibble. The GGUF
// block layout (dequantMXFP4) keeps the E8M0 scale byte first, then 16 payload
// bytes where byte j holds element j in its low nibble and element j+16 in its
// high nibble. RepackMXFP4 reorders the codes between those two layouts without
// touching any value: no calibration, no re-quantization. Adapted from
// antirez/ds4 gguf-tools/deepseek41_dspark.py expert_bytes and
// deepseek41_quantize.py to_mxfp4 (MIT, pin
// b3f50180ec2c7324892be1f50bc83555ccb8d5e7).

var (
	// ErrMXFP4MisalignedSource: the packed code length is not a multiple of
	// the 16-byte block stride (16-column alignment violation).
	ErrMXFP4MisalignedSource = errors.New("gguf: mxfp4 repack: packed source is not 16-byte aligned")
	// ErrMXFP4ScaleShapeMismatch: the E8M0 scale count disagrees with the
	// packed source block count.
	ErrMXFP4ScaleShapeMismatch = errors.New("gguf: mxfp4 repack: E8M0 scale shape disagrees with packed source")
	// ErrMXFP4NonfiniteScale: an E8M0 scale byte is 255 (nonfinite).
	ErrMXFP4NonfiniteScale = errors.New("gguf: mxfp4 repack: nonfinite E8M0 scale (byte 255)")
	// ErrMXFP4RepackInvariant: the defensive post-pack nibble check failed.
	// Unreachable for valid inputs; guards the packing arithmetic.
	ErrMXFP4RepackInvariant = errors.New("gguf: mxfp4 repack: packed nibbles disagree with unpacked source")
	// ErrMXFP4ArtifactAssemblyDeferred names the deferred half of the MXFP4
	// seam: the lossless repack primitive ships here, but end-to-end
	// safetensors-source-to-GGUF artifact assembly through the conversion
	// surface is not wired yet.
	ErrMXFP4ArtifactAssemblyDeferred = errors.New("gguf: mxfp4 source-to-GGUF artifact assembly is not wired yet; the lossless repack primitive is ggufload.RepackMXFP4")
)

// RepackMXFP4 converts packed source MXFP4 codes into GGUF MXFP4 block bytes.
// codes holds blocks*16 bytes: two 4-bit E2M1 codes per byte, consecutive
// element order (byte j holds elements 2j and 2j+1). scales holds one E8M0
// byte per 32-element block, in row-major order. The result is
// blocks*blockMXFP4Bytes: each 17-byte GGUF block starts with the unchanged
// E8M0 scale byte, followed by 16 payload bytes carrying elements 0..15 in
// the low nibbles and 16..31 in the high nibbles. Empty input repacks to
// empty output.
func RepackMXFP4(codes, scales []byte) ([]byte, error) {
	if len(codes)%16 != 0 {
		return nil, fmt.Errorf("%w: %d codes", ErrMXFP4MisalignedSource, len(codes))
	}
	blocks := len(codes) / 16
	if len(scales) != blocks {
		return nil, fmt.Errorf("%w: %d scales for %d blocks", ErrMXFP4ScaleShapeMismatch, len(scales), blocks)
	}
	for i, s := range scales {
		if s == 255 {
			return nil, fmt.Errorf("%w: block %d", ErrMXFP4NonfiniteScale, i)
		}
	}
	out := make([]byte, blocks*blockMXFP4Bytes)
	var unpacked [qkMXFP4]byte
	for b := 0; b < blocks; b++ {
		src := codes[b*16 : b*16+16]
		for j := 0; j < qkMXFP4/2; j++ {
			unpacked[2*j] = src[j] & 0x0f
			unpacked[2*j+1] = src[j] >> 4
		}
		dst := out[b*blockMXFP4Bytes:]
		dst[0] = scales[b]
		for j := 0; j < qkMXFP4/2; j++ {
			dst[1+j] = unpacked[j] | unpacked[j+qkMXFP4/2]<<4
		}
		// Defensive invariant mirroring the upstream round-trip assert:
		// repacking must never change a code.
		for j := 0; j < qkMXFP4/2; j++ {
			if dst[1+j]&0x0f != unpacked[j] || dst[1+j]>>4 != unpacked[j+qkMXFP4/2] {
				return nil, fmt.Errorf("%w: block %d", ErrMXFP4RepackInvariant, b)
			}
		}
	}
	return out, nil
}

// SpotCheckExpertIndices returns the upstream validator recipe for byte-exact
// expert spot checks: experts {0, (layer*17+41)%count, count-1}, deduplicated
// in that order (deepseek41_validate_gguf.py check_payload at the pinned
// revision). Invalid arguments return nil.
func SpotCheckExpertIndices(layer, count int) []int {
	if layer < 0 || count <= 0 {
		return nil
	}
	idx := []int{0}
	add := func(i int) {
		for _, x := range idx {
			if x == i {
				return
			}
		}
		idx = append(idx, i)
	}
	add((layer*17 + 41) % count)
	add(count - 1)
	return idx
}

// F16ExactlyRepresentable reports whether every value in values survives an
// IEEE-754 binary16 round trip unchanged - the lossless-converter gate that
// admits a dense tensor to F16 storage; anything else stays F32. Mirrors the
// upstream dense_type check (values.astype(float16).astype(float32) compared
// array-equal at the pinned revision).
func F16ExactlyRepresentable(values []float32) bool {
	for _, v := range values {
		h := f32ToF16Bits(v)
		back := math.Float32frombits(kquantbits.F16BitsToF32Bits(h))
		if back != v {
			return false
		}
	}
	return true
}

// f32ToF16Bits converts a float32 to IEEE-754 binary16 bits using
// round-to-nearest-even, matching numpy float32.astype(float16): overflow
// yields ±Inf, underflow yields ±0 or a subnormal, NaN maps to the quiet form.
func f32ToF16Bits(v float32) uint16 {
	b := math.Float32bits(v)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23) & 0xFF
	frac := b & 0x7FFFFF
	switch {
	case exp == 0xFF: // ±Inf or NaN
		if frac == 0 {
			return sign | 0x7C00
		}
		return sign | 0x7E00
	case exp >= 143: // magnitude >= 2^16 rounds to ±Inf (65536 ties to Inf)
		return sign | 0x7C00
	case exp >= 113: // f16 normal range: biased f16 exponent 1..30
		hexp := uint16(exp - 112)
		hfrac := uint16(frac >> 13)
		rem := frac & 0x1FFF
		if rem > 0x1000 || (rem == 0x1000 && hfrac&1 == 1) {
			hfrac++
			if hfrac == 0x400 {
				return sign | 0x7C00
			}
		}
		return sign | hexp<<10 | hfrac
	default: // zero or f16-subnormal result
		if exp <= 101 { // magnitude < 2^-25 always rounds to ±0
			return sign
		}
		mant := uint32(frac) | 0x800000
		shift := uint(126 - exp) // bits dropped from the 24-bit mantissa
		rounded := mant >> shift
		remBits := mant & ((1 << shift) - 1)
		half := uint32(1) << (shift - 1)
		if remBits > half || (remBits == half && rounded&1 == 1) {
			rounded++
		}
		if rounded > 1023 {
			return sign | 0x3C00 // rounded up into the smallest normal
		}
		return sign | uint16(rounded)
	}
}
