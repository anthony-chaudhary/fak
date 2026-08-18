package compute

import (
	"math"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// fp8.go — the always-compiled, hardware-independent FP8 decode reference for the compute HAL
// (issue #4209, "load + dequant an FP8 checkpoint"). compute.go declares the FP8 dtype family
// (E4M3/E5M2) but ships no element decode; this is the correctness-first, CPU, no-GPU half
// #4209's step-2 asks for — the exact byte→float32 numeric semantics a fused device kernel must
// later match. The E4M3 weight decode already lives in internal/model (fp8_blockscale.go, the
// safetensors path), which the compute package cannot import (the model→compute dependency runs
// one way); E5M2 — the wider-range FP8 format used for gradients/activations in mixed-FP8
// checkpoints — is absent repo-wide, so it lands here as the compute-side reference. Per-tensor /
// per-block scale application and GGUF tensor-type wiring stay the device/loader-gated follow-on;
// this file is only the scalar element decode, unit-witnessed on any host.
//
// Why the decode is an exact identity, not re-derived exponent arithmetic: OCP float8_e5m2 is
// 1 sign / 5 exponent (bias 15) / 2 mantissa — bit-for-bit the top 8 bits of IEEE binary16, which
// is 1 sign / 5 exponent (bias 15) / 10 mantissa. Laying an E5M2 byte into the high 8 bits of a
// 16-bit value (b<<8) zero-fills the low 8 mantissa bits, so the binary16 it names carries the
// SAME sign, exponent and leading mantissa bits — i.e. exactly the E5M2 value, across normals,
// subnormals, ±0, ±Inf and NaN (both formats share exponent bias 15 and the all-ones-exponent
// Inf/NaN encoding). So DecodeE5M2(b) is precisely f16(b<<8), reusing the in-package f16 decoder
// (f16bitsToF32) that the Q4_K scale path is already witnessed against.

// DecodeE5M2 decodes one OCP float8_e5m2 (1 sign / 5 exponent, bias 15 / 2 mantissa) byte to a
// float32. Because E5M2 is bit-identical to the top 8 bits of IEEE binary16, the decode is the
// exact identity DecodeE5M2(b) == f16(uint16(b)<<8): it reuses the in-package f16bitsToF32 so
// every byte — including subnormals (exponent 0), ±0, and ±Inf / NaN (exponent 31) — decodes
// identically to the corresponding binary16. No scale is applied here; the per-block / per-tensor
// FP8 scale is the loader's, kept separate so this stays the pure element-decode reference (#4209).
func DecodeE5M2(b byte) float32 {
	return math.Float32frombits(f16bitsToF32(uint16(b) << 8))
}

// DecodeE4M3 decodes one OCP float8_e4m3fn (1 sign / 4 exponent, bias 7 / 3 mantissa) byte
// to a float32. Unlike E5M2, e4m3 is NOT a prefix of any IEEE float (its 4-bit exponent has
// bias 7, binary16's 5-bit exponent has bias 15), so there is no bit-identity shortcut — the
// value is reconstructed arithmetically. The "fn" (finite) variant has NO infinities: the sole
// non-finite encoding is S.1111.111 (0x7F / 0xFF), which is NaN; every other exponent-15 code
// is a normal number, giving a max finite magnitude of 448. Exponent 0 is the subnormal range
// (2^(1-7) * man/8), and man==0 there is signed zero. Every e4m3 value is exactly representable
// in f32, so this is an exact identity, not an approximation — the same numeric contract the
// safetensors weight path pins in internal/model (fp8E4M3ToF32); it is duplicated here rather
// than shared because model imports compute, not the reverse, so the compute HAL must own its
// FP8 element decode to be self-sufficient (#4209 step 2).
func DecodeE4M3(b byte) float32 { return mathx.DecodeE4M3(b) }
