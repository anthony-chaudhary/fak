package compute

import (
	"math"
	"strings"
	"testing"
)

// shadow_test.go — adversarial tests for the shadow contract (ticket: "up-level shadow
// contract — exact companion to a quantized plane").
//
// Authored to the DECLARED INTERFACE only; the implementation (shadow.go) was not read
// before this file was drafted. The contract is BINARY: a shadow is bit-identical to the
// pre-quantization source or it refuses. There is no tolerance, and -0.0 != +0.0 because
// bit-identity is tested via math.Float32bits, not float ==.

// shadowPlaneQ8 builds a real Q8_0 host tensor of the given shape on the CPU reference.
// The codes/scales are arbitrary — only the dtype/shape/width matter to the constructor.
func shadowPlaneQ8(t *testing.T, shape []int) Tensor {
	t.Helper()
	n := 1
	for _, d := range shape {
		n *= d
	}
	block := 32
	if n%block != 0 {
		block = n
	}
	nblk := 1
	if block > 0 {
		nblk = (n + block - 1) / block
	}
	return NewQ8(cpu(), shape, make([]int8, n), make([]float32, nblk), block)
}

// shadowSourceF32 builds an f32 host tensor of the given shape with distinct finite values.
func shadowSourceF32(shape []int) Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i)*0.5 - 3.25
	}
	return NewF32(cpu(), shape, data)
}

// A. Constructor happy path ---------------------------------------------------------

func TestShadowNewShadowedHappyPath(t *testing.T) {
	shape := []int{4, 8}
	quant := shadowPlaneQ8(t, shape)
	shadow := shadowSourceF32(shape)
	spec := Shadow{Purpose: ShadowPurposeExactEviction, Dtype: F32, Lifetime: ShadowLifetimeOnEvict}

	got, err := NewShadowed(quant, shadow, spec)
	if err != nil {
		t.Fatalf("NewShadowed(q8 plane, f32 shadow, on-evict) erred: %v", err)
	}
	if got.Quant.Dtype != Q8_0 {
		t.Fatalf("Shadowed.Quant.Dtype = %v, want q8_0", got.Quant.Dtype)
	}
	if got.Shadow.Dtype != F32 {
		t.Fatalf("Shadowed.Shadow.Dtype = %v, want f32", got.Shadow.Dtype)
	}
	if got.Spec != spec {
		t.Fatalf("Shadowed.Spec = %+v, want %+v (fields must round-trip)", got.Spec, spec)
	}
	if got.Spec.Purpose != ShadowPurposeExactEviction {
		t.Fatalf("Shadowed.Spec.Purpose = %q, want %q", got.Spec.Purpose, ShadowPurposeExactEviction)
	}
	if got.Spec.Lifetime != ShadowLifetimeOnEvict {
		t.Fatalf("Shadowed.Spec.Lifetime = %q, want %q", got.Spec.Lifetime, ShadowLifetimeOnEvict)
	}
}

// B. Refusal — non-quantized plane ---------------------------------------------------

func TestShadowRefusesNonQuantizedPlane(t *testing.T) {
	shape := []int{4, 8}
	spec := Shadow{Purpose: ShadowPurposeExactEviction, Dtype: F32, Lifetime: ShadowLifetimeOnEvict}

	for _, plane := range []Dtype{F32, F16, BF16} {
		// A plane of the same non-quantized dtype, shadowed by a "wider" f32 twin.
		planeT := NewF32(cpu(), shape, make([]float32, 32))
		if plane != F32 {
			// only F32 has a public host constructor here; F16/BF16 are still non-quantized
			// and must refuse BEFORE any width reasoning. Build the tensor directly.
			planeT = makeTensor(cpu(), plane, RowMajor, shape, nil, &hostBuf{f32: make([]float32, 32)})
		}
		_, err := NewShadowed(planeT, shadowSourceF32(shape), spec)
		if err == nil {
			t.Fatalf("NewShadowed accepted a non-quantized %v plane; the companion only exists for a quantized plane", plane)
		}
		if !strings.Contains(err.Error(), plane.String()) {
			t.Fatalf("refusal error for %v plane does not name the plane dtype: %v", plane, err)
		}
	}
}

// C. Refusal — narrower-or-equal shadow ----------------------------------------------

func TestShadowRefusesEqualWidthShadow(t *testing.T) {
	shape := []int{4, 8}
	quant := shadowPlaneQ8(t, shape)
	// A Q8_0 "shadow" of a Q8_0 plane: not strictly wider => refuse.
	shadow := shadowPlaneQ8(t, shape)
	spec := Shadow{Purpose: ShadowPurposeHotRowOverride, Dtype: Q8_0, Lifetime: ShadowLifetimeResident}

	_, err := NewShadowed(quant, shadow, spec)
	if err == nil {
		t.Fatal("NewShadowed accepted an equal-width (q8_0) shadow of a q8_0 plane; strict-wider violated")
	}
	// The refusal should name the dtypes actually involved (q8_0 shadow / q8_0 plane) so an
	// operator can act. No f32 participates in an equal-width refusal.
	if !strings.Contains(err.Error(), Q8_0.String()) {
		t.Fatalf("equal-width refusal does not name the q8_0 dtype: %v", err)
	}
}

func TestShadowRefusesNarrowerShadow(t *testing.T) {
	shape := []int{4, 8}
	quant := shadowPlaneQ8(t, shape)
	// A strictly narrower shadow (i4) of a q8_0 plane.
	narrow := makeTensor(cpu(), I4, RowMajor, shape, &QuantSpec{Block: 32, Axis: 2, Bits: 4, Symmetric: true},
		&hostBuf{i8: make([]int8, 32)})
	spec := Shadow{Purpose: ShadowPurposeHotRowOverride, Dtype: I4, Lifetime: ShadowLifetimeResident}

	_, err := NewShadowed(quant, narrow, spec)
	if err == nil {
		t.Fatal("NewShadowed accepted a strictly narrower (i4) shadow of a q8_0 plane")
	}
	if !strings.Contains(err.Error(), I4.String()) {
		t.Fatalf("narrower refusal does not name the shadow dtype: %v", err)
	}
}

// D. Refusal — shape mismatch --------------------------------------------------------

func TestShadowRefusesShapeMismatch(t *testing.T) {
	quant := shadowPlaneQ8(t, []int{4, 8})
	// Same dtype width (f32 shadow is wider), but a DIFFERENT shape.
	shadow := shadowSourceF32([]int{4, 4})
	spec := Shadow{Purpose: ShadowPurposeQualityFloor, Dtype: F32, Lifetime: ShadowLifetimeResident}

	_, err := NewShadowed(quant, shadow, spec)
	if err == nil {
		t.Fatal("NewShadowed accepted a shadow whose shape differs from the plane's shape")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "shape") {
		t.Fatalf("shape-mismatch refusal does not mention shape: %v", err)
	}
}

// E. Refusal — empty lifetime --------------------------------------------------------

func TestShadowRefusesEmptyLifetime(t *testing.T) {
	shape := []int{4, 8}
	quant := shadowPlaneQ8(t, shape)
	shadow := shadowSourceF32(shape)
	spec := Shadow{Purpose: ShadowPurposeExactEviction, Dtype: F32, Lifetime: ""}

	_, err := NewShadowed(quant, shadow, spec)
	if err == nil {
		t.Fatal("NewShadowed accepted an empty Lifetime; lifetime must be explicit and fail closed")
	}
}

// F. VerifyExact — bit-identity pass + adversarial bit probes ------------------------

func TestShadowVerifyExactBitIdenticalPass(t *testing.T) {
	shape := []int{4, 8}
	src := make([]float32, 32)
	for i := range src {
		src[i] = math.Float32frombits(0x3f000000 + uint32(i)) // distinct finite values
	}
	quant := shadowPlaneQ8(t, shape)
	shadow := NewF32(cpu(), shape, append([]float32(nil), src...))
	s, err := NewShadowed(quant, shadow, Shadow{Purpose: ShadowPurposeExactEviction, Dtype: F32, Lifetime: ShadowLifetimeOnEvict})
	if err != nil {
		t.Fatalf("setup NewShadowed erred: %v", err)
	}
	if err := s.VerifyExact(src); err != nil {
		t.Fatalf("VerifyExact on a bit-identical shadow erred: %v", err)
	}
}

func TestShadowVerifyExactNegativeZeroDiffersFromPositiveZero(t *testing.T) {
	shape := []int{2, 4}
	src := []float32{1, 2, 3, 4, float32(math.Copysign(0, -1)), 6, 7, 8}
	shadowVals := []float32{1, 2, 3, 4, 0, 6, 7, 8} // +0.0 where source is -0.0
	quant := shadowPlaneQ8(t, shape)
	// Build the shadow from the *shadowVals* to make the +0.0 explicit in the buffer.
	shadow := NewF32(cpu(), shape, append([]float32(nil), shadowVals...))
	s, err := NewShadowed(quant, shadow, Shadow{Purpose: ShadowPurposeExactEviction, Dtype: F32, Lifetime: ShadowLifetimeOnEvict})
	if err != nil {
		t.Fatalf("setup NewShadowed erred: %v", err)
	}
	// Sanity: the two bit patterns really do differ.
	if math.Float32bits(src[4]) == math.Float32bits(shadowVals[4]) {
		t.Fatal("test setup broken: -0.0 and +0.0 share a bit pattern?")
	}
	if err := s.VerifyExact(src); err == nil {
		t.Fatal("VerifyExact accepted +0.0 against a -0.0 source; `==` semantics leaked into a bit-identity contract")
	}
}

func TestShadowVerifyExactNaNPayloadIdenticalPasses(t *testing.T) {
	shape := []int{2, 4}
	src := []float32{1, 2, math.Float32frombits(0x7fc00001), 4, 5, 6, 7, 8}
	quant := shadowPlaneQ8(t, shape)
	shadow := NewF32(cpu(), shape, append([]float32(nil), src...))
	s, err := NewShadowed(quant, shadow, Shadow{Purpose: ShadowPurposeExactEviction, Dtype: F32, Lifetime: ShadowLifetimeOnEvict})
	if err != nil {
		t.Fatalf("setup NewShadowed erred: %v", err)
	}
	if err := s.VerifyExact(src); err != nil {
		t.Fatalf("VerifyExact on an identical NaN payload erred (bit-identity must accept it): %v", err)
	}
}

func TestShadowVerifyExactNaNDifferentPayloadFails(t *testing.T) {
	shape := []int{2, 4}
	src := []float32{1, 2, math.Float32frombits(0x7fc00001), 4, 5, 6, 7, 8}
	shadowVals := []float32{1, 2, math.Float32frombits(0x7fc00002), 4, 5, 6, 7, 8}
	quant := shadowPlaneQ8(t, shape)
	shadow := NewF32(cpu(), shape, append([]float32(nil), shadowVals...))
	s, err := NewShadowed(quant, shadow, Shadow{Purpose: ShadowPurposeExactEviction, Dtype: F32, Lifetime: ShadowLifetimeOnEvict})
	if err != nil {
		t.Fatalf("setup NewShadowed erred: %v", err)
	}
	if err := s.VerifyExact(src); err == nil {
		t.Fatal("VerifyExact accepted a NaN with a DIFFERENT bit payload; bit-identity violated")
	}
}

func TestShadowVerifyExactMiddleElementCorruptionNamesIndex(t *testing.T) {
	shape := []int{2, 5}
	src := []float32{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
	const corrupt = 5 // neither first (0) nor last (9)
	shadowVals := append([]float32(nil), src...)
	shadowVals[corrupt] += 0.25
	quant := shadowPlaneQ8(t, shape)
	shadow := NewF32(cpu(), shape, shadowVals)
	s, err := NewShadowed(quant, shadow, Shadow{Purpose: ShadowPurposeExactEviction, Dtype: F32, Lifetime: ShadowLifetimeOnEvict})
	if err != nil {
		t.Fatalf("setup NewShadowed erred: %v", err)
	}
	err = s.VerifyExact(src)
	if err == nil {
		t.Fatal("VerifyExact accepted a shadow corrupted at a middle element")
	}
	// The error should name the offending index so the defect is locatable.
	if !strings.Contains(err.Error(), "5") {
		t.Fatalf("corruption refusal does not name the index 5: %v", err)
	}
}

// J. Binary contract — a 1-ULP difference refuses ------------------------------------

func TestShadowVerifyExactOneULPFails(t *testing.T) {
	shape := []int{2, 4}
	src := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	shadowVals := append([]float32(nil), src...)
	// +1 ULP on one element (next float32 magnitude up).
	shadowVals[3] = math.Float32frombits(math.Float32bits(shadowVals[3]) + 1)
	quant := shadowPlaneQ8(t, shape)
	shadow := NewF32(cpu(), shape, shadowVals)
	s, err := NewShadowed(quant, shadow, Shadow{Purpose: ShadowPurposeExactEviction, Dtype: F32, Lifetime: ShadowLifetimeOnEvict})
	if err != nil {
		t.Fatalf("setup NewShadowed erred: %v", err)
	}
	if math.Float32bits(src[3]) == math.Float32bits(shadowVals[3]) {
		t.Fatal("test setup broken: +1 ULP did not change the bit pattern")
	}
	if err := s.VerifyExact(src); err == nil {
		t.Fatal("VerifyExact accepted a 1-ULP difference; the contract is binary, not approximate")
	}
}

// G. VerifyExact fail-closed — short source ------------------------------------------

func TestShadowVerifyExactShortSourceFails(t *testing.T) {
	shape := []int{4, 8}
	quant := shadowPlaneQ8(t, shape)
	full := make([]float32, 32)
	for i := range full {
		full[i] = float32(i)
	}
	shadow := NewF32(cpu(), shape, append([]float32(nil), full...))
	s, err := NewShadowed(quant, shadow, Shadow{Purpose: ShadowPurposeExactEviction, Dtype: F32, Lifetime: ShadowLifetimeOnEvict})
	if err != nil {
		t.Fatalf("setup NewShadowed erred: %v", err)
	}
	if err := s.VerifyExact(full[:16]); err == nil {
		t.Fatal("VerifyExact accepted a source shorter than the shadow's Numel; a truncated pre-quantization buffer must fail closed")
	}
}

// The non-host-addressable arm (a shadow whose Buf is not a HostBuffer, i.e. a device
// tensor) cannot be built deterministically on this host without a GPU: every device
// backend here requires a live CUDA/Metal/Vulkan device to produce a resident tensor, and
// faking a non-host Buffer would test the fake, not the contract. That arm is
// GP-witnessed elsewhere (device memory residency tests); here we only pin the
// host-addressable short-source path above rather than fabricate a device.

// H. KVExactEvictShadow — the first named instantiation ------------------------------

func TestShadowKVExactEvictShadowQ8(t *testing.T) {
	spec, err := KVExactEvictShadow(Q8_0)
	if err != nil {
		t.Fatalf("KVExactEvictShadow(Q8_0) erred: %v", err)
	}
	if spec.Dtype != F32 {
		t.Fatalf("KVExactEvictShadow(Q8_0).Dtype = %v, want f32 (the pre-quantization companion)", spec.Dtype)
	}
	if spec.Purpose != ShadowPurposeExactEviction {
		t.Fatalf("KVExactEvictShadow(Q8_0).Purpose = %q, want %q", spec.Purpose, ShadowPurposeExactEviction)
	}
	if spec.Lifetime != ShadowLifetimeOnEvict {
		t.Fatalf("KVExactEvictShadow(Q8_0).Lifetime = %q, want %q", spec.Lifetime, ShadowLifetimeOnEvict)
	}
}

func TestShadowKVExactEvictShadowRefusesNonWider(t *testing.T) {
	// F32 is not strictly wider than F32 — no shadow is possible.
	if _, err := KVExactEvictShadow(F32); err == nil {
		t.Fatal("KVExactEvictShadow(F32) must refuse: f32 is not strictly wider than f32")
	}
	// NOTE: F16/BF16 DO get a valid f32 shadow (f32 is strictly wider than a 16-bit float),
	// so they are NOT in this refusal set — the declared contract only refuses where f32 is
	// not strictly wider. Pin that the wider-than-f16 case still yields an f32 companion.
	spec, err := KVExactEvictShadow(F16)
	if err != nil {
		t.Fatalf("KVExactEvictShadow(F16) should succeed (f32 is strictly wider than f16): %v", err)
	}
	if spec.Dtype != F32 {
		t.Fatalf("KVExactEvictShadow(F16).Dtype = %v, want f32", spec.Dtype)
	}
}

// I. KV `mixed` instantiation, end-to-end --------------------------------------------

// TestShadowKVMixedInstantiationExactEvictRow is the shadow-side identity for the q8 KV
// tier: a quantized plane paired with the retained f32 pre-RoPE-K shadow verifies
// bit-identical against the ORIGINAL source, AND the tier's per-token arithmetic is
// unchanged (the one parity assertion; the byte-floor test lives in kvprecision_test.go).
func TestShadowKVMixedInstantiationExactEvictRow(t *testing.T) {
	cfg := KVConfig{NumLayers: 2, NumKVHeads: 4, HeadDim: 8}
	const elemsPerRow = 32

	// The pre-quantization source: one pre-RoPE K row of real f32 values.
	src := make([]float32, elemsPerRow)
	for i := range src {
		src[i] = math.Float32frombits(0x3e800000 + uint32(i)*0x00010000)
	}

	// The attended rows are stored q8_0 (a quantized plane of the same shape); the
	// pre-RoPE K row is retained verbatim as its f32 shadow.
	quant := shadowPlaneQ8(t, []int{1, elemsPerRow})
	shadow := NewF32(cpu(), []int{1, elemsPerRow}, append([]float32(nil), src...))

	spec, err := KVExactEvictShadow(Q8_0)
	if err != nil {
		t.Fatalf("KVExactEvictShadow(Q8_0) erred: %v", err)
	}
	s, err := NewShadowed(quant, shadow, spec)
	if err != nil {
		t.Fatalf("NewShadowed for the KV mixed tier erred: %v", err)
	}
	if err := s.VerifyExact(src); err != nil {
		t.Fatalf("the retained pre-RoPE K shadow is not bit-identical to the f32 source: %v", err)
	}

	// Behavior-preservation parity: the shadow's f32 dtype is the same F32.Bytes() the
	// kRaw term always charged, so the tier arithmetic is unchanged.
	f32Cfg := cfg
	f32Cfg.Precision = KVPrecisionF32
	if got := EstimateKVStoreBytes(f32Cfg, 1); got != 768 {
		t.Fatalf("f32 per-token KV bytes = %d, want 768", got)
	}
	q8Cfg := cfg
	q8Cfg.Precision = KVPrecisionQ8
	if got := EstimateKVStoreBytes(q8Cfg, 1); got != 400 {
		t.Fatalf("q8 (mixed) per-token KV bytes = %d, want 400", got)
	}
}

// ---- K. Strict-decreasing-by-bit-width guard (rank-ordering regressions) ------------
//
// The width ordering must be strictly DECREASING in real bit width as the rank falls, so
// a same-width-class shadow can never sneak in. The earlier tests only probed Q8_0-of-Q8_0
// and I4-of-Q8_0; these pin the DISCRIMINATING cross-class pairs where the pre-fix
// class-ordering (which tied q8_0 and q4_k at "one byte") would have wrongly accepted.
//
// The observer here is an INDEPENDENT rank table (shadowExpectedRank) written from the
// DECLARED contract in the ticket, never the unexported widthRank — so a mutation of
// shadow.go's ranks cannot tautologically satisfy these tests.

// shadowExpectedRank is the expected real-bit-width rank, written independently of
// shadow.go. Higher = strictly wider. F32=7; F16/BF16=6; FP8=5; Q8_0/I8=4;
// Q6_K/Q5_K=3; Q4_K/I4/FP4=2; Q2_0/Q2_K/IQ3_XXS/IQ3_S/IQ2_XXS=1; unknown=0.
func shadowExpectedRank(dt Dtype) int {
	switch dt {
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
	case Q2_0, Q2_K, IQ3_XXS, IQ3_S, IQ2_XXS:
		return 1
	default:
		return 0
	}
}

// shadowQuantPlane builds a QUANTIZED host plane of an arbitrary dtype. Only Dtype and
// Shape participate in the constructor's width/shape checks, so the codes are zeroed and
// the QuantSpec is a plausible per-element descriptor. n is sized to the shape's Numel.
func shadowQuantPlane(dt Dtype, shape []int) Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	q := &QuantSpec{Block: n, Axis: 2, Bits: 8, Symmetric: true}
	if dt == Q4_K || dt == Q5_K || dt == Q6_K || dt == Q2_K || dt == IQ3_XXS || dt == IQ3_S || dt == IQ2_XXS {
		q = &QuantSpec{Block: 256, Axis: 2, Bits: 4, Symmetric: false}
	}
	if dt == I4 || dt == FP4 {
		q = &QuantSpec{Block: n, Axis: 2, Bits: 4, Symmetric: true}
	}
	return makeTensor(cpu(), dt, RowMajor, append([]int(nil), shape...), q, &hostBuf{i8: make([]int8, n)})
}

// shadowAttempt is one (quant plane, shadow dtype) refusal/acceptance observation.
type shadowAttempt struct {
	name  string
	quant Dtype
	shad  Dtype
	want  bool // true => must be accepted
}

// shadowCheckAttempts asserts each case against NewShadowed and reports the exact pair.
func shadowCheckAttempts(t *testing.T, cases []shadowAttempt) {
	t.Helper()
	shape := []int{4, 8}
	spec := Shadow{Purpose: ShadowPurposeHotRowOverride, Lifetime: ShadowLifetimeResident}
	for _, c := range cases {
		plane := shadowQuantPlane(c.quant, shape)
		shadow := shadowSourceF32(shape)
		spec.Dtype = c.shad
		_, err := NewShadowed(plane, shadow, spec)
		got := err == nil
		if got != c.want {
			t.Fatalf("%s: NewShadowed(%s plane, %s shadow) accepted=%v, want accepted=%v (independent ranks: shadow=%d plane=%d): %v",
				c.name, c.quant, c.shad, got, c.want, shadowExpectedRank(c.shad), shadowExpectedRank(c.quant), err)
		}
	}
}

// TestShadowStrictWidthQ8ShadowOfQ4KAccepted pins the half-fixed discriminating case:
// q8_0 (rank 4) IS strictly wider than q4_k (rank 2), so the shadow is ACCEPTED. The
// pre-fix class ordering that tied both to "one byte" would have refused this.
func TestShadowStrictWidthQ8ShadowOfQ4KAccepted(t *testing.T) {
	shadowCheckAttempts(t, []shadowAttempt{
		{name: "q8_0 shadow of q4_k plane", quant: Q4_K, shad: Q8_0, want: true},
	})
}

// TestShadowStrictWidthQ4KShadowOfQ8Refused pins the REVERSE — the case the pre-fix
// class-ordering would have WRONGLY ACCEPTED: q4_k (rank 2) is NOT wider than q8_0
// (rank 4), so a q4_k shadow of a q8_0 plane MUST REFUSE.
func TestShadowStrictWidthQ4KShadowOfQ8Refused(t *testing.T) {
	shadowCheckAttempts(t, []shadowAttempt{
		{name: "q4_k shadow of q8_0 plane", quant: Q8_0, shad: Q4_K, want: false},
	})
}

// TestShadowStrictWidthEqualRankRefusedEitherDirection proves equal-rank pairs never pass
// in EITHER direction: i4 and q4_k share rank 2, so both orderings refuse.
func TestShadowStrictWidthEqualRankRefusedEitherDirection(t *testing.T) {
	shadowCheckAttempts(t, []shadowAttempt{
		{name: "i4 shadow of q4_k plane (equal rank 2)", quant: Q4_K, shad: I4, want: false},
		{name: "q4_k shadow of i4 plane (equal rank 2)", quant: I4, shad: Q4_K, want: false},
	})
}

// TestShadowStrictWidthKQuantAdjacentClasses pins adjacency within the K-quant band:
// q6_k (rank 3) is strictly wider than q4_k (rank 2) => accept; the reverse refuses.
func TestShadowStrictWidthKQuantAdjacentClasses(t *testing.T) {
	shadowCheckAttempts(t, []shadowAttempt{
		{name: "q6_k shadow of q4_k plane", quant: Q4_K, shad: Q6_K, want: true},
		{name: "q4_k shadow of q6_k plane", quant: Q6_K, shad: Q4_K, want: false},
	})
}

// TestShadowStrictWidthFP8BeatsI8ButNotReverse pins the exponent-class rule: fp8 and i8
// share a byte cost, but fp8 (rank 5) carries an exponent, so it is strictly wider than
// i8 (rank 4) and a fp8 shadow of an i8 plane is accepted; the reverse refuses.
func TestShadowStrictWidthFP8BeatsI8ButNotReverse(t *testing.T) {
	shadowCheckAttempts(t, []shadowAttempt{
		{name: "fp8 shadow of i8 plane (exponent class is strictly wider)", quant: I8, shad: FP8, want: true},
		{name: "i8 shadow of fp8 plane (no exponent, not wider)", quant: FP8, shad: I8, want: false},
	})
}

// TestShadowStrictWidthMonotonicity is the anti-tautology guard: over a fixed table of
// Dtype constants it asserts NewShadowed succeeds IFF the shadow's INDEPENDENTLY computed
// rank exceeds the plane's. It recomputes the expectation from shadowExpectedRank, so it
// would fail if either shadow.go's widthRank or this test's mirror table drifted.
func TestShadowStrictWidthMonotonicity(t *testing.T) {
	// The shadow may be ANY dtype; the PLANE must additionally be quantized (the
	// constructor refuses to exact-ify an already-exact plane before any width reasoning),
	// so the expected predicate is `plane.Quantized() && shadowRank > planeRank`.
	shadows := []Dtype{
		F32, F16, BF16, FP8, Q8_0, I8, Q6_K, Q5_K, Q4_K, I4, FP4, Q2_0, Q2_K, IQ3_XXS, IQ3_S, IQ2_XXS,
	}
	planes := []Dtype{
		FP8, Q8_0, I8, Q6_K, Q5_K, Q4_K, I4, FP4, Q2_0, Q2_K, IQ3_XXS, IQ3_S, IQ2_XXS,
	}
	shape := []int{4, 8}
	spec := Shadow{Purpose: ShadowPurposeQualityFloor, Lifetime: ShadowLifetimeResident}
	for _, plane := range planes {
		if !plane.Quantized() {
			t.Fatalf("test table broken: %s is not quantized but is used as a plane", plane)
		}
		for _, shad := range shadows {
			planeT := shadowQuantPlane(plane, shape)
			shadowT := shadowQuantPlane(shad, shape)
			spec.Dtype = shad
			_, err := NewShadowed(planeT, shadowT, spec)
			got := err == nil
			want := plane.Quantized() && shadowExpectedRank(shad) > shadowExpectedRank(plane)
			if got != want {
				t.Fatalf("monotonicity violated: shadow %s (rank %d) of plane %s (rank %d): accepted=%v, want=%v: %v",
					shad, shadowExpectedRank(shad), plane, shadowExpectedRank(plane), got, want, err)
			}
		}
	}
}

// TestShadowStrictWidthKVExactEvictGuard pins KVExactEvictShadow against the narrowed
// order: q2_k (rank 1) is strictly below f32, so it yields a valid F32 companion, while
// f32 is not strictly above f32 so it refuses (re-asserted cheaply here).
func TestShadowStrictWidthKVExactEvictGuard(t *testing.T) {
	spec, err := KVExactEvictShadow(Q2_K)
	if err != nil {
		t.Fatalf("KVExactEvictShadow(Q2_K) should succeed (f32 is widest): %v", err)
	}
	if spec.Dtype != F32 {
		t.Fatalf("KVExactEvictShadow(Q2_K).Dtype = %v, want f32", spec.Dtype)
	}
	if shadowExpectedRank(spec.Dtype) <= shadowExpectedRank(Q2_K) {
		t.Fatalf("f32 rank %d is not strictly wider than q2_k rank %d", shadowExpectedRank(spec.Dtype), shadowExpectedRank(Q2_K))
	}
	if _, err := KVExactEvictShadow(F32); err == nil {
		t.Fatal("KVExactEvictShadow(F32) must refuse: f32 is not strictly wider than f32")
	}
}
