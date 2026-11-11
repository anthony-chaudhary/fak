// silu_mul_fp8_quant.go — fused SiLU-mul + per-token-group FP8 (e4m3fn,
// ue8m0 scale) quantization of expert intermediate activations.
//
// ADAPTED from vllm-project/vllm,
// csrc/libtorch_stable/quantization/activation_kernels.cu:276
// (silu_mul_fp8_quant_deep_gemm_kernel) with the act_and_mul_kernel structure
// (csrc/libtorch_stable/activation_kernels.cu:102) at
// b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd (Apache-2.0; attribution in
// THIRD_PARTY_NOTICES.md): swiglu(gate, up) followed by per-128-group e4m3fn
// quantization whose ue8m0 scale is driven by the group amax, producing the
// packed codes+scales a stage-2 down-projection GEMM would consume.
// Re-expressed here as the CPU library/oracle that pins that contract; the
// CUDA kernel lives in silu_mul_fp8_quant.cu. No upstream code copied.
// Ref oss-port-vllmtop-silu-mul-fp8-quant-20261103.
//
// SEMANTICS (pinned by silu_mul_fp8_quant_test.go):
//   - Row layout follows act_and_mul: token t reads gate = x[t*2H : t*2H+H]
//     and up = x[t*2H+H : t*3H]; outputs are H payload bytes (e4m3fn codes)
//     plus one ue8m0 scale byte per 128-wide group, both row-major.
//   - Group scale: amax = max(|swiglu|) over the group's present elements
//     (the ragged tail counts only its own elements), floored at
//     SiluMulFp8AmaxFloor (upstream's EPS, bf16 raw 0x2E5C widened to f32);
//     exponent = ceil(log2(amax/448)) taken exactly from the f32 bits of
//     amax/448, stored biased by 127 (ue8m0).
//   - Payload: value * 2^-exponent rounded to nearest e4m3fn, ties to even,
//     signed saturation at +/-448 (codes 0x7E/0xFE). The 0x7F/0xFF NaN codes
//     are never emitted on a clean call.
//   - Non-finite refusal: a NaN or infinite gate/up value, or a fused product
//     that overflows f32, refuses the whole call with ErrSiluMulFp8NonFinite;
//     the device path records the first fault index in the status word
//     instead, so a non-finite can never become a plausible byte code.
//
// Deltas from upstream (documented, pinned):
//   - f32 input and intermediates (upstream's deep-gemm path runs bf16); the
//     silu exponential is evaluated in f64 on the host for determinism while
//     the device uses expf (device precision) — qualification must bound it
//     against this oracle.
//   - Ragged tails: H need not be a multiple of 128; the last group covers
//     H%128 elements (upstream requires H % 128 == 0).
//   - The scale exponent is computed exactly in f32 rather than upstream's
//     bf16 log2/ceil pipeline.
//   - Upstream leaves non-finite propagation unguarded; this candidate
//     refuses.

package compute

import (
	"errors"
	"math"
)

const (
	// SiluMulFp8GroupSize is the number of dims sharing one ue8m0 scale.
	SiluMulFp8GroupSize = 128
	// SiluMulFp8Fp8Max is the e4m3fn finite maximum, the saturation bound.
	SiluMulFp8Fp8Max = 448.0
)

var (
	// SiluMulFp8AmaxFloor floors a group amax so an all-zero group still
	// lands on a finite scale; upstream's EPS expressed as bf16 raw 0x2E5C
	// widened to f32.
	SiluMulFp8AmaxFloor = math.Float32frombits(0x2E5C0000)
)

var (
	// ErrSiluMulFp8Empty rejects a zero-token batch call.
	ErrSiluMulFp8Empty = errors.New("compute: silu mul fp8 quant: empty input")

	// ErrSiluMulFp8Shape rejects a row width or buffer length that does not
	// match the declared [tokens, 2*hidden] geometry.
	ErrSiluMulFp8Shape = errors.New("compute: silu mul fp8 quant: shape mismatch")

	// ErrSiluMulFp8NonFinite rejects a gate/up value that is NaN or infinite,
	// or a fused swiglu product that overflows f32, before any quantization.
	ErrSiluMulFp8NonFinite = errors.New("compute: silu mul fp8 quant: non-finite value")
)

// SiluMulFp8NumGroups returns the group count for a row width: full 128-dim
// groups plus one ragged tail group when hidden % 128 != 0.
func SiluMulFp8NumGroups(hidden int) int {
	return (hidden + SiluMulFp8GroupSize - 1) / SiluMulFp8GroupSize
}

// SiluMulFp8QuantRow fuses swiglu and per-group fp8 quantization for one
// token: gate and up must both be exactly hidden wide. It returns H payload
// bytes (e4m3fn codes) and one ue8m0 scale byte per 128-wide group.
func SiluMulFp8QuantRow(gate, up []float32) (codes, scales []byte, err error) {
	if len(gate) == 0 || len(gate) != len(up) {
		return nil, nil, ErrSiluMulFp8Shape
	}
	codes = make([]byte, len(gate))
	scales = make([]byte, SiluMulFp8NumGroups(len(gate)))
	if err := siluMulFp8QuantRowInto(gate, up, codes, scales); err != nil {
		return nil, nil, err
	}
	return codes, scales, nil
}

// SiluMulFp8Quant quantizes tokens rows of a [tokens, 2*hidden] buffer laid
// out act_and_mul style: token t's gate half is x[t*2H : t*2H+H], its up
// half is x[t*2H+H : t*3H]. It returns codes [tokens, H] and scales
// [tokens, groups(H)], both row-major.
func SiluMulFp8Quant(x []float32, tokens, hidden int) (codes, scales []byte, err error) {
	if tokens <= 0 {
		return nil, nil, ErrSiluMulFp8Empty
	}
	if hidden <= 0 || int64(tokens)*int64(2*hidden) != int64(len(x)) {
		return nil, nil, ErrSiluMulFp8Shape
	}
	ng := SiluMulFp8NumGroups(hidden)
	codes = make([]byte, tokens*hidden)
	scales = make([]byte, tokens*ng)
	for t := 0; t < tokens; t++ {
		row := x[t*2*hidden : (t+1)*2*hidden]
		if err := siluMulFp8QuantRowInto(row[:hidden], row[hidden:],
			codes[t*hidden:(t+1)*hidden], scales[t*ng:(t+1)*ng]); err != nil {
			return nil, nil, err
		}
	}
	return codes, scales, nil
}

// siluMulFp8QuantRowInto is the shared row core: it writes H codes and one
// scale byte per group for one token's gate/up halves. A non-finite operand
// or an overflowing fused product refuses the whole row with
// ErrSiluMulFp8NonFinite before any byte is published.
func siluMulFp8QuantRowInto(gate, up []float32, codes, scales []byte) error {
	hidden := len(gate)
	ng := SiluMulFp8NumGroups(hidden)
	if len(up) != hidden || len(codes) != hidden || len(scales) != ng {
		return ErrSiluMulFp8Shape
	}
	var res [SiluMulFp8GroupSize]float32
	for g := 0; g < ng; g++ {
		lo := g * SiluMulFp8GroupSize
		hi := lo + SiluMulFp8GroupSize
		if hi > hidden {
			hi = hidden // ragged tail: count only the present elements
		}
		var amax float32
		for i := lo; i < hi; i++ {
			gr, ur := gate[i], up[i]
			if !siluMulFp8IsFinite(gr) || !siluMulFp8IsFinite(ur) {
				return ErrSiluMulFp8NonFinite
			}
			r := siluMulFp8Silu(gr) * ur
			if !siluMulFp8IsFinite(r) {
				return ErrSiluMulFp8NonFinite
			}
			res[i-lo] = r
			a := r
			if a < 0 {
				a = -a
			}
			if a > amax {
				amax = a
			}
		}
		exp := siluMulFp8GroupExponent(amax)
		scales[g] = siluMulFp8EncodeScale(exp)
		inv := float32(math.Ldexp(1, -exp)) // 2^-exp, exact
		for j := 0; j < hi-lo; j++ {
			codes[lo+j] = siluMulFp8EncodeE4M3(res[j] * inv)
		}
	}
	return nil
}

// siluMulFp8Silu evaluates z/(1+exp(-z)) in f64 with a single final f32
// rounding: the deterministic host form of upstream's silu_kernel. Finite
// inputs only; the caller refuses non-finite operands first.
func siluMulFp8Silu(z float32) float32 {
	return float32(float64(z) / (1 + math.Exp(-float64(z))))
}

// siluMulFp8IsFinite reports whether v is neither NaN nor infinite.
func siluMulFp8IsFinite(v float32) bool {
	return math.Float32bits(v)&0x7f800000 != 0x7f800000
}

// siluMulFp8GroupExponent returns the ue8m0 exponent for one group: the
// smallest integer e with 2^e >= amax/448, amax floored at
// SiluMulFp8AmaxFloor. raw = amax/448 is a positive normal f32 (the floor
// keeps it above 2^-126), so ceil(log2(raw)) resolves from its bits without
// a transcendental: unbiased exponent field, plus one when the mantissa is
// nonzero. This matches the f64 reference ceil(log2(amax/448)) for every
// finite input.
func siluMulFp8GroupExponent(amax float32) int {
	if amax < SiluMulFp8AmaxFloor {
		amax = SiluMulFp8AmaxFloor
	}
	raw := amax / SiluMulFp8Fp8Max
	bits := math.Float32bits(raw)
	exp := int(bits>>23)&0xff - 127
	if bits&0x7fffff != 0 {
		exp++
	}
	return exp
}

// siluMulFp8EncodeScale biases the exponent by 127 and clamps to the uint8
// range. Finite inputs keep the bias inside [84, 247], so the clamp is a
// closed-contract guard, not a reachable path.
func siluMulFp8EncodeScale(exp int) byte {
	enc := exp + 127
	if enc < 0 {
		enc = 0
	}
	if enc > 255 {
		enc = 255
	}
	return byte(enc)
}

// siluMulFp8EncodeE4M3 rounds one already-scaled value to the nearest OCP
// e4m3fn byte: ties to even, signed saturation at +/-448 (0x7E/0xFE). The
// 0x7F/0xFF NaN codes are never emitted. It is the bit-exact host mirror of
// smfq_e4m3_encode in silu_mul_fp8_quant.cu.
func siluMulFp8EncodeE4M3(x float32) byte {
	if x >= SiluMulFp8Fp8Max {
		return 0x7E
	}
	if x <= -SiluMulFp8Fp8Max {
		return 0xFE
	}
	bits := math.Float32bits(x)
	sign := bits >> 31
	mag := bits & 0x7fffffff
	if mag == 0 {
		return byte(sign << 7)
	}
	exp := int(mag>>23)&0xff - 127
	if exp >= -6 {
		frac := mag & 0x7fffff
		m := uint32(frac >> 20)
		rem := uint32(frac & 0xfffff)
		if rem > 0x80000 || (rem == 0x80000 && m&1 != 0) {
			m++
		}
		e := exp + 7
		if m == 8 {
			m = 0
			e++
		}
		if e > 15 {
			return byte(sign<<7 | 0x7E)
		}
		return byte(sign<<7 | uint32(e)<<3 | m)
	}
	// Subnormal target: q = |x| * 512 is exact in f64 and lies in [0, 8).
	q := float64(math.Float32frombits(mag)) * 512
	m := uint32(q)
	r := q - float64(m)
	if r > 0.5 || (r == 0.5 && m&1 != 0) {
		m++
	}
	if m == 8 {
		// Tie at the top of the subnormal band rolls to the smallest
		// normal (code 0x08), not down to the largest subnormal.
		return byte(sign<<7 | 0x08)
	}
	return byte(sign<<7 | m)
}
