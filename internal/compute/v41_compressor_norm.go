package compute

import (
	"errors"
	"fmt"
	"math"
)

// ErrV41CompressorNormDomain is the closed sentinel for every rejected
// geometry, domain, or operand condition of the dedicated V4.1 compressor
// norm: width/eps outside V41CompressorNormDomain, unequal pooled/gain
// lengths, or a non-finite pooled or gain element.
var ErrV41CompressorNormDomain = errors.New("compute: V4.1 compressor norm input is outside the exact domain")

// V41CompressorNormMaxWidth bounds the serial per-row reduction of the
// dedicated compressor norm kernel. V4.1 binds it at the compressor width
// (512); the bound mirrors the exact RMSNorm kernel so one domain covers both
// device routes.
const V41CompressorNormMaxWidth = V41ExactRMSNormMaxWidth

// V41CompressorNormDomain reports whether width/eps is inside the dedicated
// compressor-norm domain: width a power of two up to the bound, eps a finite
// positive normal binary32.
func V41CompressorNormDomain(width int, eps float32) bool {
	return V41ExactRMSNormDomain(width, eps)
}

// v41CompressorFinite32 reports whether v is a finite binary32 value.
func v41CompressorFinite32(v float32) bool { return math.Float32bits(v)&0x7fffffff < 0x7f800000 }

// v41CompressorBF16Round rounds a finite binary32 value to the nearest BF16
// value, ties to even, and returns its binary32 widening. The increment is
// the softfloat pack rule: half an ulp plus the dropped low bit, so a tie
// lands on the even significand. A finite input never rounds to NaN; past
// the BF16 overflow edge it rounds to infinity.
func v41CompressorBF16Round(v float32) float32 {
	b := math.Float32bits(v)
	b += 0x7fff + uint32((b>>16)&1)
	return math.Float32frombits(b & 0xffff0000)
}

// v41CompressorSqrt64 returns the square root of a non-negative finite x,
// rounded to nearest even, extracted with integer arithmetic only. It adapts
// the bit-by-bit method pinned in the package header: x scales into [1,4)
// with an even power of two, the significand extracts its root one bit at a
// time, and one final round-to-nearest-even consults the remainder as the
// sticky bit. An exact root leaves the remainder zero; a nonzero remainder
// can never sit on a half-ulp, so the even test reduces to the root's low
// bit.
func v41CompressorSqrt64(x float64) float64 {
	switch {
	case x == 0, math.IsNaN(x), math.IsInf(x, 1):
		return x
	case x < 0:
		return math.NaN()
	}
	bits := math.Float64bits(x)
	exp := int(bits >> 52 & 0x7ff)
	sig := bits & (1<<52 - 1)
	if exp == 0 { // subnormal: raise the significand, tracking the scale
		for sig&(1<<52) == 0 {
			sig <<= 1
			exp--
		}
		exp++
	} else {
		sig |= 1 << 52
	}
	exp -= 1023
	if exp&1 == 1 { // odd exponent: scale by two so the scaling power stays even
		sig <<= 1
	}
	exp >>= 1 // exponent of the root
	var root, carried uint64
	n := sig << 1
	for bit := uint64(1) << 53; bit != 0; bit >>= 1 {
		if t := carried + bit; t <= n {
			carried = t + bit
			n -= t
			root += bit
		}
		n <<= 1
	}
	if n != 0 { // not exact: round to nearest even on the extracted low bit
		root += root & 1
	}
	// The root's leading bit always carries into the exponent field, so the
	// field is written one low and the significand added in.
	return math.Float64frombits(root>>1 + uint64(exp-1+1023)<<52)
}

// v41CompressorRecip64 returns the reciprocal of a positive finite x, rounded
// to nearest even, computed as a restoring long division of 2^106 by the
// significand: a 54-bit quotient with one guard bit and the remainder kept as
// the sticky bit, mirroring the softfloat fdiv64 information flow with a
// single round-to-nearest-even at the pack. Subnormal results round the exact
// product (quotient + remainder/significand) once, so the division remainder
// stays in the half-ulp compare.
func v41CompressorRecip64(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return math.NaN()
	case x == 0:
		return math.Inf(1)
	case math.IsInf(x, 1):
		return 0
	}
	bits := math.Float64bits(x)
	exp := int(bits >> 52 & 0x7ff)
	sig := bits & (1<<52 - 1)
	if exp == 0 {
		for sig&(1<<52) == 0 {
			sig <<= 1
			exp--
		}
		exp++
	} else {
		sig |= 1 << 52
	}
	exp -= 1023 // x = sig*2^(exp-52), sig in [2^52, 2^53)
	var q, r uint64
	for i := 106; i >= 0; i-- {
		r <<= 1
		if i == 106 {
			r |= 1
		}
		q <<= 1
		if r >= sig {
			r -= sig
			q |= 1
		}
	}
	// 2^106 = q*sig + r, so 1/x = (q + r/sig)*2^(-54-exp).
	s := q >> 1
	if q&1 != 0 && (r != 0 || s&1 == 1) {
		s++
	}
	e := -1 - exp
	if s == 1<<53 {
		s = 1 << 52
		e++
	}
	if e > 1023 {
		return math.Inf(1)
	}
	if e >= -1022 {
		return math.Float64frombits(uint64(e+1023)<<52 | (s & (1<<52 - 1)))
	}
	// Subnormal result: value = (q + r/sig)*2^(1020-exp); round that exact
	// product once, keeping the division remainder in the half-ulp compare.
	sh := uint(exp - 1020)
	m := q >> sh
	drop := q & (1<<sh - 1)
	num := drop*sig + r
	half := sig << (sh - 1)
	if num > half || (num == half && m&1 == 1) {
		m++
	}
	if m == 1<<52 {
		return math.Float64frombits(1 << 52)
	}
	return math.Float64frombits(m)
}

// validateV41CompressorNorm checks geometry, domain, and operand finiteness
// before any rounding happens. Every rejection wraps
// ErrV41CompressorNormDomain.
func validateV41CompressorNorm(pooled, gain []float32, eps float32) (int, error) {
	width := len(pooled)
	if len(gain) != width || !V41CompressorNormDomain(width, eps) {
		return 0, fmt.Errorf("compute: V4.1 compressor norm width=%d eps=%g is outside the exact domain: %w", width, eps, ErrV41CompressorNormDomain)
	}
	for i, value := range pooled {
		if !v41CompressorFinite32(value) {
			return 0, fmt.Errorf("compute: V4.1 compressor norm pooled element %d is not finite: %w", i, ErrV41CompressorNormDomain)
		}
	}
	for i, value := range gain {
		if !v41CompressorFinite32(value) {
			return 0, fmt.Errorf("compute: V4.1 compressor norm gain element %d is not finite: %w", i, ErrV41CompressorNormDomain)
		}
	}
	return width, nil
}

// V41CompressorNorm computes one dedicated V4.1 compressor norm row: raw
// pooled binary32 values, the BF16 input boundary, the learned RMSNorm, and
// the BF16 output boundary. It is the scalar reference for the device integer
// route (Ref #13668 preparatory lane ds-compressor-device-13668); universal
// backend Supports stays false pending mathematical, compiled-artifact and
// physical evidence.
//
// The rounding order is closed and every step rounds once:
//
//  1. each pooled value rounds to BF16, round-to-nearest-even;
//  2. squares accumulate as a serial binary32 sum in index order;
//  3. the mean is the direct binary32 quotient by the power-of-two width,
//     which is exact: scaling by a power of two changes only the exponent, so
//     no widened-renarrowed soft-float division whose double rounding golang/go
//     issue 48807 flags is ever on the path;
//  4. eps adds in binary32;
//  5. the square root rounds to binary64, the reciprocal rounds to binary64
//     separately, and the reciprocal casts to binary32;
//  6. each lane multiplies (input * reciprocal) * gain in that order;
//  7. each result rounds to BF16.
//
// Step 5 runs the integer mechanisms adapted from golang/go go1.26.0:
// src/math/sqrt.go (sha256
// c7e036d1511bea94992014f9b8c9259e5737c385cdd5135db9092369250d4a94) supplies
// the bit-by-bit significand extraction with one final round-to-nearest-even,
// and src/runtime/softfloat64.go (sha256
// 94f52048c124deff27c7b4c0215eab39700b16cd9a3781e6453f7900a4eb2005) supplies
// the pack/unpack round-to-nearest-even discipline the reciprocal packing
// follows. The reuse verdict is ADAPT: the ideas are re-expressed in
// idiomatic Go, and no upstream code block is copied.
//
// pooled and gain must have the same length inside V41CompressorNormDomain
// and be finite. The result is a fresh caller-owned slice; every element is
// the result of the BF16 output rounding. A serial square sum that overflows
// binary32 flows through as IEEE arithmetic (mean +Inf resolves the row to
// zeros), matching the device route bit for bit.
func V41CompressorNorm(pooled, gain []float32, eps float32) ([]float32, error) {
	width, err := validateV41CompressorNorm(pooled, gain, eps)
	if err != nil {
		return nil, err
	}
	out := make([]float32, width)
	sum := float32(0)
	for i, value := range pooled {
		x := v41CompressorBF16Round(value)
		out[i] = x
		sum += x * x
	}
	mean := sum / float32(width)
	y := mean + eps
	inv := float32(v41CompressorRecip64(v41CompressorSqrt64(float64(y))))
	for i := 0; i < width; i++ {
		v := out[i] * inv
		v *= gain[i]
		out[i] = v41CompressorBF16Round(v)
	}
	return out, nil
}
