// rmsnorm_fp8_quant.go — fused RMSNorm + static-scale FP8 E4M3 activation
// quantization.
//
// ADAPTED from vllm-project/vllm, csrc/layernorm_quant_kernels.cu:23,88,161
// (rms_norm_static_fp8_quant_kernel, fused_add_rms_norm_static_fp8_quant_kernel)
// at ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0): one-pass RMSNorm
// writing OCP e4m3fn codes under a static per-tensor scale, for layers whose
// next op consumes FP8 activations. Re-expressed here as the CPU oracle that
// pins that contract; the CUDA kernel lives in rmsnorm_fp8_quant.cu. No
// upstream code copied. Ref oss-port-vllmtop-rmsnorm-fp8-quant-20261103.
//
// This file is pure Go on purpose: it compiles and is tested in every build,
// while the device kernel ships behind the cuda build tag once the successor
// policy-hook packet wires rmsnorm_fp8_quant.cu into the single libfakcuda
// translation unit. The cgo launch wrapper lands with that wiring; declaring
// the C ABI here would break every -tags cuda link against a libfakcuda.a that
// does not carry the symbols yet.
//
// Semantics mirror the pinned kernel entry by entry:
//   - One row is one token; sumsq accumulates x[i]*x[i] sequentially in index
//     order (upstream's cub BlockReduce order is implementation-dependent;
//     the fixed sequential order is what makes byte-pinning possible).
//   - inv = 1/sqrt(sumsq/width + eps), all binary32 (upstream uses rsqrtf, up
//     to 1.5 ulp error; the exact reciprocal-of-sqrt keeps the oracle
//     bit-pinnable).
//   - Per element: n = (x*inv)*w; code = e4m3(n * scale_inv), ties to even,
//     signed saturation at +/-448, the 0x7F NaN code never emitted. The
//     encoder (rfqEncodeE4M3) follows the OCP e4m3fn grid end to end,
//     matching upstream's hardware cvt.rn.satfinite.e4m3x2.f32 — including
//     the subnormal step 2^-9 and the promotion of a rounded-up top
//     subnormal to the smallest normal 2^-6. It deliberately does NOT reuse
//     dsv41kEncodeE4M3, whose subnormal grid is m*2^-13 and therefore
//     diverges from the OCP format below 2^-6.
//   - Fused variant: res = x + residual is written first, then normalized and
//     quantized from res (upstream re-reads the stored residual; same value).
//
// Deliberate divergence from upstream: a zero-variance row (sumsq == 0 — an
// all-zero row, or one whose elements are too small to contribute to the f32
// squared sum) is refused on the host with ErrRmsNormFp8ZeroVariance instead
// of silently emitting an all-zero code row. The device twin keeps the
// graceful all-zero-code fallback so a bad row mid-batch cannot abort the
// forward pass. The host also refuses non-finite arithmetic intermediates;
// device execution outside that admitted domain remains unqualified.
// Library + oracle only: no serving path changes until an FP8
// activation consumer exists. No GPU or throughput claim.

package compute

import (
	"errors"
	"fmt"
	"math"
)

var (
	// ErrRmsNormFp8Geometry is the closed refusal for inputs that violate
	// operand lengths or overflow the int32 device ABI.
	ErrRmsNormFp8Geometry = errors.New("compute: rmsnorm fp8 quant: geometry refused")

	// ErrRmsNormFp8Params is the closed refusal for a non-finite or out-of-range
	// static scale, its reciprocal, or epsilon.
	ErrRmsNormFp8Params = errors.New("compute: rmsnorm fp8 quant: invalid scale or epsilon")

	// ErrRmsNormFp8NonFinite is the closed refusal for a NaN or infinite value in
	// x, residual, weight or arithmetic intermediates before quantization, so
	// a non-finite can never become a plausible byte code.
	ErrRmsNormFp8NonFinite = errors.New("compute: rmsnorm fp8 quant: non-finite value")

	// ErrRmsNormFp8ZeroVariance is the closed refusal for a row whose f32
	// squared sum is zero; such a row would emit an all-zero code row.
	ErrRmsNormFp8ZeroVariance = errors.New("compute: rmsnorm fp8 quant: zero-variance row")
)

// RmsNormFp8QuantArgs is one fused RMSNorm + static-scale FP8 quantize call.
type RmsNormFp8QuantArgs struct {
	// X is the row-major [rows][width] f32 activation panel.
	X []float32
	// Weight is the [width] f32 RMSNorm gain.
	Weight []float32
	// Residual is an optional [rows][width] f32 residual stream; nil selects
	// the plain variant, non-nil the fused add-first variant.
	Residual []float32
	// Scale is the static per-tensor FP8 scale (finite, > 0, finite reciprocal).
	Scale float32
	// Epsilon is the RMSNorm stabilizer added to the mean square (finite, >= 0).
	Epsilon float32
}

// RmsNormFp8QuantResult is the CPU image of what fcuda_rmsnorm_fp8_quant
// produces on device for one RmsNormFp8QuantArgs.
type RmsNormFp8QuantResult struct {
	// Codes is the [rows][width] OCP e4m3fn activation panel, one byte per dim.
	Codes []byte
	// ResidualOut is the [rows][width] add result when Residual != nil;
	// otherwise nil.
	ResidualOut []float32
}

// rfqValidate closes the admission contract and returns the declared geometry.
func rfqValidate(args RmsNormFp8QuantArgs) (rows, width int, err error) {
	width = len(args.Weight)
	if width <= 0 {
		return 0, 0, fmt.Errorf("%w: width=%d", ErrRmsNormFp8Geometry, width)
	}
	rows = len(args.X) / width
	if rows <= 0 || len(args.X)%width != 0 {
		return 0, 0, fmt.Errorf("%w: x length %d is not a positive multiple of width %d",
			ErrRmsNormFp8Geometry, len(args.X), width)
	}
	if !rfqGeometryOK(rows, width) {
		return 0, 0, fmt.Errorf("%w: rows=%d width=%d overflow the int32 ABI",
			ErrRmsNormFp8Geometry, rows, width)
	}
	if args.Residual != nil && len(args.Residual) != len(args.X) {
		return 0, 0, fmt.Errorf("%w: residual length %d != x length %d",
			ErrRmsNormFp8Geometry, len(args.Residual), len(args.X))
	}
	if !rfqParamOK(args.Scale, args.Epsilon) {
		return 0, 0, fmt.Errorf("%w: scale=%v epsilon=%v",
			ErrRmsNormFp8Params, args.Scale, args.Epsilon)
	}
	for i, v := range args.X {
		if rfqNonFinite(v) {
			return 0, 0, fmt.Errorf("%w: x[%d]", ErrRmsNormFp8NonFinite, i)
		}
	}
	for i, v := range args.Residual {
		if rfqNonFinite(v) {
			return 0, 0, fmt.Errorf("%w: residual[%d]", ErrRmsNormFp8NonFinite, i)
		}
	}
	for i, v := range args.Weight {
		if rfqNonFinite(v) {
			return 0, 0, fmt.Errorf("%w: weight[%d]", ErrRmsNormFp8NonFinite, i)
		}
	}
	return rows, width, nil
}

// rfqGeometryOK matches the CUDA total-element limit without multiplying ints.
func rfqGeometryOK(rows, width int) bool {
	const maxInt32 = int(1<<31 - 1)
	return rows > 0 && width > 0 && width <= maxInt32 && rows <= maxInt32/width
}

// rfqParamOK admits the static scale and epsilon: finite, scale strictly
// positive with a finite reciprocal, epsilon non-negative.
func rfqParamOK(scale, epsilon float32) bool {
	return scale > 0 && math.Float32bits(scale)&0x7f800000 != 0x7f800000 &&
		!rfqNonFinite(1.0/scale) &&
		epsilon >= 0 && math.Float32bits(epsilon)&0x7f800000 != 0x7f800000
}

// rfqNonFinite reports NaN or infinite values before they can turn into
// plausible-looking byte codes.
func rfqNonFinite(v float32) bool { return math.Float32bits(v)&0x7f800000 == 0x7f800000 }

// rfqEncodeE4M3 rounds one finite value to the nearest OCP e4m3fn byte:
// ties to even, signed saturation at +/-448 (codes 0x7E/0xFE). The 0x7F NaN
// code is never emitted. Subnormals follow the OCP grid m*2^-9 (m in 0..7);
// a rounded-up top subnormal promotes to the smallest normal 2^-6 (code
// 0x08), keeping the grid gap-free at 2^-6. It is the bit-exact host mirror
// of rfq_e4m3_encode in rmsnorm_fp8_quant.cu.
func rfqEncodeE4M3(x float32) byte {
	if x >= 448.0 {
		return 0x7E
	}
	if x <= -448.0 {
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
	// Subnormal target: q = |x| * 512 = |x| / 2^-9 is exact in f64 and lies
	// in [0, 8).
	q := float64(math.Float32frombits(mag)) * 512
	m := uint32(q)
	r := q - float64(m)
	if r > 0.5 || (r == 0.5 && m&1 != 0) {
		m++
	}
	if m > 7 {
		return byte(sign<<7 | 0x08) // promote to the smallest normal
	}
	return byte(sign<<7 | m)
}

// RmsNormFp8Quant computes RMSNorm (optionally after adding Residual first)
// and writes E4M3 codes under the static per-tensor scale, mirroring the
// pinned vLLM kernels. It returns either a fully populated result or an error
// — never a partial one.
func RmsNormFp8Quant(args RmsNormFp8QuantArgs) (*RmsNormFp8QuantResult, error) {
	rows, width, err := rfqValidate(args)
	if err != nil {
		return nil, err
	}
	total := rows * width
	scaleInv := 1.0 / args.Scale // f32, mirrors the kernel's inverted scale
	codes := make([]byte, total)
	fused := args.Residual != nil
	var resOut []float32
	if fused {
		resOut = make([]float32, total)
	}
	for r := 0; r < rows; r++ {
		base := r * width
		var sumsq float32
		if fused {
			for i := 0; i < width; i++ {
				s := args.X[base+i] + args.Residual[base+i]
				resOut[base+i] = s
				sumsq += s * s
			}
		} else {
			for i := 0; i < width; i++ {
				v := args.X[base+i]
				sumsq += v * v
			}
		}
		if rfqNonFinite(sumsq) {
			return nil, fmt.Errorf("%w: sum of squares at row %d", ErrRmsNormFp8NonFinite, r)
		}
		if sumsq == 0 {
			return nil, fmt.Errorf("%w: row %d", ErrRmsNormFp8ZeroVariance, r)
		}
		denom := sumsq/float32(width) + args.Epsilon
		if rfqNonFinite(denom) {
			return nil, fmt.Errorf("%w: denominator at row %d", ErrRmsNormFp8NonFinite, r)
		}
		inv := 1.0 / float32(math.Sqrt(float64(denom)))
		if rfqNonFinite(inv) {
			return nil, fmt.Errorf("%w: reciprocal RMS at row %d", ErrRmsNormFp8NonFinite, r)
		}
		src := args.X
		if fused {
			src = resOut
		}
		for i := 0; i < width; i++ {
			n := src[base+i] * inv * args.Weight[i]
			if rfqNonFinite(n) {
				return nil, fmt.Errorf("%w: normalization at row %d column %d", ErrRmsNormFp8NonFinite, r, i)
			}
			q := n * scaleInv
			if rfqNonFinite(q) {
				return nil, fmt.Errorf("%w: scaling at row %d column %d", ErrRmsNormFp8NonFinite, r, i)
			}
			codes[base+i] = rfqEncodeE4M3(q)
		}
	}
	return &RmsNormFp8QuantResult{Codes: codes, ResidualOut: resOut}, nil
}
