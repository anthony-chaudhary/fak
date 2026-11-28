package quantmeta

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// exact.go is the #14494 exact-quantization admission gate: a pure, fail-closed
// check of whether storing a concrete value under a named quantization format
// and reading it back is bit-identical. It answers "does this value survive this
// format?", never "is this format good?" -- #6221's no-universal-winner
// guardrail holds: the check is a pure function of (value, format) and it ranks
// nothing.
//
// The invariant is the upstream strata "exact mode reports failure rather than
// substituting" contract (ro99/strata, Apache-2.0, pinned at
// REFUSED at admission, never silently rounded to the nearest representable
// value.
//
// Topology invariance: the check depends on no device, batch, rank or launch
// identity; it is a pure function of (value, format).

var (
	// ErrNonFiniteValue is returned when a non-finite value is offered to the
	// exactness gate. It is a refusal, not a false pass.
	ErrNonFiniteValue = errors.New("quantmeta: non-finite value cannot be exactness-checked")
	// ErrFormatNotCheckable is returned when the named format cannot be
	// exactness-checked: it is outside this build's vocabulary, or it is known
	// but this build has no round-trip primitive for it. Fail-closed: we cannot
	// verify exactness, so we do not admit it.
	ErrFormatNotCheckable = errors.New("quantmeta: format not checkable for exactness")
)

// RoundTripExact reports whether encoding value under format and decoding it
// back yields value bit-identically. It is a pure function of (value, format):
// no device, batch, rank or launch identity enters the check.
//
// It is fail-closed: a non-finite value, an unknown format, or a known format
// this build cannot check returns an error, never a false "exact" pass. A
// finite value whose storage would alter it returns (false, nil) -- the refusal
// the admission gate acts on.
func RoundTripExact(value float32, format Format) (bool, error) {
	if !format.Known() {
		return false, fmt.Errorf("%w: %q", ErrFormatNotCheckable, format)
	}
	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		return false, fmt.Errorf("%w: %v", ErrNonFiniteValue, value)
	}
	switch format {
	case FormatFP8E4M3:
		return fp8E4M3RoundTrip(value), nil
	case FormatFP4E2M1:
		return fp4E2M1RoundTrip(value), nil
	case FormatMXFP4:
		return mxfp4RoundTrip(value), nil
	default:
		return false, fmt.Errorf("%w: %q", ErrFormatNotCheckable, format)
	}
}

// AdmitExact is the exact-quantization admission gate. It REFUSEs with
// ReasonExactnessViolation when any value in the set would be altered by storage
// under format, and SUPPORTs only when every value round-trips exactly. An empty
// set is vacuously exact and SUPPORTs.
func AdmitExact(values []float32, format Format) (Result, error) {
	if !format.Known() {
		return Result{}, fmt.Errorf("%w: %q", ErrFormatNotCheckable, format)
	}
	for i, v := range values {
		ok, err := RoundTripExact(v, format)
		if err != nil {
			return Result{}, fmt.Errorf("quantmeta: value %d: %w", i, err)
		}
		if !ok {
			return result(OutcomeRefuse, ReasonExactnessViolation, ClaimNone), nil
		}
	}
	return result(OutcomeSupported, ReasonFullyDeclared, ClaimNone), nil
}

// decodeFP8E4M3 converts an 8-bit fp8-e4m3 (e4m3fn) code to float32. The layout
// is 1 sign, 4 exponent (bias 7), 3 mantissa; 0x7F and 0xFF are the NaN codes
// and there is no infinity.
func decodeFP8E4M3(code uint8) float32 {
	sign := code & 0x80
	eField := int(code&0x78) >> 3
	m := int(code & 0x07)
	var mag float64
	switch {
	case eField == 0:
		mag = float64(m) * math.Ldexp(1, -9) // subnormal: m * 2^-9
	case eField == 15 && m == 7:
		return math.Float32frombits(0x7FC00000 | uint32(sign)<<23) // NaN
	default:
		mag = (1 + float64(m)/8) * math.Ldexp(1, eField-7)
	}
	if sign != 0 {
		mag = -mag
	}
	// Every representable magnitude is exactly a float32, so the narrowing is
	// exact and the decode is bit-faithful.
	return float32(mag)
}

// fp8E4M3Magnitudes is the sorted set of magnitudes the fp8-e4m3 layout holds:
// the decodes of codes 0x00..0x7E (code 0x7F is the NaN code). It is the
// "decode" side of the round trip: a value round-trips exactly iff its
// magnitude is one of these.
var fp8E4M3Magnitudes = func() []float32 {
	mags := make([]float32, 0x7F) // 127 entries: codes 0x00..0x7E
	for code := 0; code <= 0x7E; code++ {
		mags[code] = decodeFP8E4M3(uint8(code))
	}
	return mags
}()

// fp8E4M3RoundTrip reports whether value is exactly representable in fp8-e4m3,
// i.e. whether encoding it and decoding it back is bit-identical.
func fp8E4M3RoundTrip(value float32) bool {
	a := math.Abs(float64(value))
	i := sort.Search(len(fp8E4M3Magnitudes), func(j int) bool {
		return float64(fp8E4M3Magnitudes[j]) >= a
	})
	return i < len(fp8E4M3Magnitudes) && float64(fp8E4M3Magnitudes[i]) == a
}

// fp4E2M1Magnitudes is the closed set of magnitudes the fp4-e2m1 layout holds:
// 1 sign, 2 exponent (bias 1), 1 mantissa. The decodes of codes 0..7.
var fp4E2M1Magnitudes = []float32{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

// fp4E2M1RoundTrip reports whether value is exactly representable in fp4-e2m1.
func fp4E2M1RoundTrip(value float32) bool {
	a := math.Abs(float64(value))
	i := sort.Search(len(fp4E2M1Magnitudes), func(j int) bool {
		return float64(fp4E2M1Magnitudes[j]) >= a
	})
	return i < len(fp4E2M1Magnitudes) && float64(fp4E2M1Magnitudes[i]) == a
}

// e2m1NonZeroMagnitudes are the non-zero fp4-e2m1 magnitudes, the mantissa side
// of the mxfp4 product.
var e2m1NonZeroMagnitudes = []float32{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

// mxfp4RoundTrip reports whether value is exactly representable in mxfp4: a
// product of an fp4-e2m1 mantissa and an E8M0 power-of-two scale. A value
// survives mxfp4 storage iff value = m * 2^e for some e2m1 magnitude m and an
// E8M0 exponent e. The scale is a per-block parameter, so the check asks
// whether some valid scale makes the value exact, never which scale is best.
func mxfp4RoundTrip(value float32) bool {
	if value == 0 {
		return true
	}
	a := value
	if a < 0 {
		a = -a
	}
	for _, m := range e2m1NonZeroMagnitudes {
		q := a / m
		if e8m0ScaleValid(q) && m*q == a {
			return true
		}
	}
	return false
}

// e8m0ScaleValid reports whether q is exactly 2^e for an integer e in the E8M0
// exponent range. The range follows the upstream strata convention (the scale
// code is e+127 with code 0xFF reserved), i.e. e in [-127, 127].
func e8m0ScaleValid(q float32) bool {
	if q <= 0 {
		return false
	}
	m, e := math.Frexp(float64(q))
	if m != 0.5 {
		return false // not a power of two
	}
	return e-1 >= -127 && e-1 <= 127
}
