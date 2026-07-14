package compute

import "math"

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
