package compute

import (
	"fmt"
	"math"
)

// shadow.go — the exact-shadow contract: an exactness-critical companion plane for a
// QUANTIZED plane, and the dtype-width ordering that makes "strictly wider" checkable.
//
// The HAL exists to let a plane be stored narrow (q8_0, q4_k, i4, …) while ONE part of
// the computation still needs the pre-quantization bytes reproduced bit-for-bit. The
// canonical case is already encoded in kvprecision.go: the q8 KV tier quantizes the two
// ATTENDED rows (post-RoPE K, V) but MUST keep the pre-RoPE K row F32, because Evict
// re-positions a survivor by a single rotation OF THE PRE-ROPE K and that rotation has
// to be bit-exact. A lossy q8 pre-RoPE K would make evict != never-saw.
//
// A Shadow names that companion: which quantized plane it exact-ifies, its own (strictly
// wider) dtype, and how long it is retained. Shadowed pairs the quantized Tensor with the
// exact Tensor. The contract is deliberately FAIL-CLOSED everywhere: an unspecified
// lifetime, a non-quantized plane, a not-strictly-wider shadow, or a shape mismatch all
// refuse rather than silently proceeding — a shadow that is not exactly right is worse
// than no shadow, because the caller believes it has exactness it does not have.

// ShadowPurpose names why a plane needs an exact companion. It is a string enum so a
// receipt/journal can carry the reason without a lookup table, and so an unknown value
// is visibly not one of the known purposes rather than folding to a numeric zero.
type ShadowPurpose string

const (
	// ShadowPurposeExactEviction is a shadow kept so a mutation on the plane (a
	// re-RoPE/rotation of a survivor) stays bit-exact — the q8 KV pre-RoPE K row.
	ShadowPurposeExactEviction ShadowPurpose = "exact-eviction"
	// ShadowPurposeHotRowOverride is a shadow for a small set of rows read on the
	// hot path where the narrower plane's error is not acceptable.
	ShadowPurposeHotRowOverride ShadowPurpose = "hot-row-override"
	// ShadowPurposeQualityFloor is a shadow that preserves the exact values a
	// quality/bit-identity check is measured against.
	ShadowPurposeQualityFloor ShadowPurpose = "quality-floor"
)

// ShadowLifetime names how long the shadow is retained. It distinguishes the
// storage-cost of the exact companion: a resident shadow pays for every position, a
// per-token shadow only for the live step, and an on-evict shadow only across the
// eviction window it protects.
type ShadowLifetime string

const (
	// ShadowLifetimeResident keeps the shadow for the full residency of the plane.
	ShadowLifetimeResident ShadowLifetime = "resident"
	// ShadowLifetimePerToken keeps the shadow only for the step that produces it.
	ShadowLifetimePerToken ShadowLifetime = "per-token"
	// ShadowLifetimeOnEvict keeps the shadow only while an eviction may rely on it.
	ShadowLifetimeOnEvict ShadowLifetime = "on-evict"
)

// Shadow declares an exactness-critical companion to a quantized plane: WHY it is
// needed, its own dtype (which must be strictly wider than the plane it shadows), and
// HOW LONG it is retained.
type Shadow struct {
	Purpose  ShadowPurpose
	Dtype    Dtype // the shadow's dtype; must be strictly wider than the plane
	Lifetime ShadowLifetime
}

// Shadowed pairs a quantized Tensor with its exact shadow Tensor plus the Shadow spec
// that justifies the pairing. Construct it through NewShadowed so the width/shape/
// lifetime invariants are checked before any caller relies on exactness.
type Shadowed struct {
	Quant  Tensor
	Shadow Tensor
	Spec   Shadow
}

// ---- dtype width ordering --------------------------------------------------------
//
// Dtype.Bytes() is NOT sufficient for a width comparison: it is the per-element BYTE
// cost, and it reports 1 for both q8_0 and i4 (packing/organization is described
// out-of-band by QuantSpec), so Bytes would tie an 8-bit code with a 4-bit one. The
// shadow contract needs a total WIDTH ordering with the real class distinctions, so we
// define an explicit ordinal here. The classes, and the reason each is ranked where it
// is:
//
//	rank 7 — F32    : the reference currency, strictly widest; nothing is wider.
//	rank 6 — F16	: IEEE binary16, 16-bit float (mantissa+exp), strictly wider than
//	           BF16  : 2-byte scalar floats of a >8-bit class.
//	rank 5 — FP8    : 8-bit float; same byte width as q8/i8 but strictly wider in the
//	                  ordering because it carries an exponent (a strictly larger
//	                  representable set is the only reason to accept a shadow of it).
//	rank 4 — Q8_0   : 8-bit class: q8_0 and i8, a full 8-bit code per element.
//	           I8
//	rank 3 — Q6_K   : >4-bit, <8-bit code class: the K-quants whose per-element code
//	           Q5_K   : width sits strictly between a nibble and a full byte (6-bit and
//	                 5-bit codes over a 256-elem super-block).
//	rank 2 — Q4_K   : 4-bit code class: q4_k's nibble codes. A shadow of a q4_k plane
//	           I4     : from q8_0/i8 (or above) is an exactness upgrade; a q8_0 shadow
//	           FP4    : of q4_k is NOT wider by bit width, so the classes are split here.
//	rank 1 — Q2_0   : <=3-bit class: ternary/packed 2-bit and the i-quant 3-/2-bit
//	           Q2_K   : codes. The narrowest band; every wider class exact-ifies one.
//	           Q3_K
//	           IQ3_XXS, IQ3_S, IQ2_XXS
//
// It is a STRICT TOTAL ORDER, and crucially it is strictly DECREASING in real bit
// width as the rank falls: no 4-bit code shares a rank with an 8-bit code, so a
// q8_0 shadow of a q4_k plane is refused rather than admitted as the same "class".
// That strictness is what lets widerThan be a plain >. The default arm returns the
// lowest rank so an UNKNOWN dtype can never be mistaken for a wide one — a shadow of an
// unknown dtype is refused by widerThan unless the plane is itself lower, and the
// constructor further requires the plane to be Quantized().
func (d Dtype) widthRank() int {
	switch d {
	case F32:
		return 7
	case F16, BF16:
		return 6
	case FP8:
		return 5
	case Q8_0, I8:
		return 4
	case Q6_K, Q5_K:
		return 3
	case Q4_K, I4, FP4:
		return 2
	case Q2_0, Q2_K, Q3_K, IQ3_XXS, IQ3_S, IQ2_XXS:
		return 1
	default:
		return 0
	}
}

// widerThan reports whether the shadow's dtype is strictly wider than the plane's, the
// only relation that justifies an exact companion: a shadow at the same width (or
// narrower) cannot promise anything the plane already has, so it is refused.
func (s Shadow) widerThan(plane Dtype) bool {
	return s.Dtype.widthRank() > plane.widthRank()
}

// NewShadowed is the fail-closed constructor for Shadowed. It refuses (returning a
// non-nil error and the zero Shadowed) when ANY invariant of the exact-shadow contract
// is broken:
//
//   - quant must be a QUANTIZED plane — a shadow exists to restore exactness to a
//     lossy plane; exact-ifying an already-exact plane is a caller bug;
//   - shadow.Dtype must be STRICTLY wider than quant.Dtype per widerThan;
//   - the two tensors must have the SAME shape (exactly the same elements);
//   - spec.Lifetime must be specified (no silent default), so a caller cannot forget
//     the retention decision and later pay for a shadow it did not account for.
//
// The returned error always names the offending dtype(s) so the refusal is actionable.
func NewShadowed(quant, shadow Tensor, spec Shadow) (Shadowed, error) {
	if !quant.Dtype.Quantized() {
		return Shadowed{}, fmt.Errorf("compute: shadow: quant plane dtype %s is not quantized; a shadow only exact-ifies a lossy plane", quant.Dtype)
	}
	if !spec.widerThan(quant.Dtype) {
		return Shadowed{}, fmt.Errorf("compute: shadow dtype %s is not strictly wider than quant plane dtype %s", spec.Dtype, quant.Dtype)
	}
	if !sameShape(quant.Shape, shadow.Shape) {
		return Shadowed{}, fmt.Errorf("compute: shadow: shape %v does not cover quant plane shape %v", shadow.Shape, quant.Shape)
	}
	if spec.Lifetime == "" {
		return Shadowed{}, fmt.Errorf("compute: shadow: unspecified lifetime for shadow of quant plane %s (set ShadowLifetime, no silent default)", quant.Dtype)
	}
	return Shadowed{Quant: quant, Shadow: shadow, Spec: spec}, nil
}

// VerifyExact checks the SHADOW tensor against `source`, the PRE-quantization f32 array
// for the covered elements. It passes iff the shadow's host f32 values are BIT-IDENTICAL
// — compared via math.Float32bits, not ==, so a NaN equals a NaN with the same payload
// and -0.0 is distinct from +0.0 — to the first Numel() elements of source, in order.
//
// It deliberately does NOT compare the QUANT plane to source: q8 is lossy by design and
// cannot reproduce source, so that check would always fail (or have to be loosened into
// meaninglessness). The shadow is the thing that promises exactness; this is the witness
// that it delivers.
//
// Fail-closed: a shadow that is not host-addressable (its Buf is not a HostBuffer, i.e.
// a device tensor), a source shorter than the shadow's Numel, or the first mismatching
// element all return an error. The mismatch error names the index and both bit patterns.
func (s Shadowed) VerifyExact(source []float32) error {
	got, ok := hostF32(s.Shadow)
	if !ok {
		return fmt.Errorf("compute: shadow: shadow is not host-addressable")
	}
	n := s.Shadow.Numel()
	if len(source) < n {
		return fmt.Errorf("compute: shadow: source has %d f32 elements, need %d for the shadow plane", len(source), n)
	}
	if len(got) < n {
		return fmt.Errorf("compute: shadow: shadow buffer has %d f32 elements, need %d", len(got), n)
	}
	for i := 0; i < n; i++ {
		want := math.Float32bits(source[i])
		have := math.Float32bits(got[i])
		if want != have {
			return fmt.Errorf("compute: shadow: element %d is not bit-exact: want %#08x (%v), got %#08x (%v)", i, want, source[i], have, got[i])
		}
	}
	return nil
}

// KVExactEvictShadow is the first named instantiation of the shadow contract: the KV
// `mixed` (q8) reserve. The pre-RoPE K row must stay bit-exact because Evict re-positions
// a survivor by a single rotation OF THE PRE-ROPE K (see kvprecision.go and capacity.go's
// pre-RoPE-K note); a lossy pre-RoPE K would make evict != never-saw. So the shadow is an
// F32 companion with an on-evict lifetime.
//
// It refuses (returning an error) when F32 is not strictly wider than quantDtype, so the
// same constructor cannot be asked to shadow a plane F32 cannot exact-ify (e.g. another
// F32, or an unknown dtype).
func KVExactEvictShadow(quantDtype Dtype) (Shadow, error) {
	spec := Shadow{
		Purpose:  ShadowPurposeExactEviction,
		Dtype:    F32,
		Lifetime: ShadowLifetimeOnEvict,
	}
	if !spec.widerThan(quantDtype) {
		return Shadow{}, fmt.Errorf("compute: kv exact-evict shadow: f32 is not strictly wider than quant plane dtype %s", quantDtype)
	}
	return spec, nil
}
