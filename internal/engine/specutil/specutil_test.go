package specutil

import (
	"math"
	"testing"
)

func TestCompareLogprobs_BitExactPass(t *testing.T) {
	specOn := []TokenLogprob{
		{TokenID: 1, Logprob: -0.5},
		{TokenID: 2, Logprob: -1.25},
		{TokenID: 3, Logprob: 0},
	}
	specOff := []TokenLogprob{
		{TokenID: 1, Logprob: -0.5},
		{TokenID: 2, Logprob: -1.25},
		{TokenID: 3, Logprob: 0},
	}
	got := CompareLogprobs(specOn, specOff)
	if !got.Pass {
		t.Fatalf("Pass = false, want true for identical sequences")
	}
	if got.DivergenceIndex != -1 {
		t.Fatalf("DivergenceIndex = %d, want -1", got.DivergenceIndex)
	}
	if got.SpecOnLength != 3 || got.SpecOffLength != 3 {
		t.Fatalf("lengths = %d/%d, want 3/3", got.SpecOnLength, got.SpecOffLength)
	}
}

func TestCompareLogprobs_TokenIDDivergence(t *testing.T) {
	specOn := []TokenLogprob{
		{TokenID: 1, Logprob: -0.5},
		{TokenID: 2, Logprob: -1.25},
		{TokenID: 3, Logprob: 0},
	}
	specOff := []TokenLogprob{
		{TokenID: 1, Logprob: -0.5},
		{TokenID: 99, Logprob: -1.25}, // token ID differs at index 1
		{TokenID: 3, Logprob: 0},
	}
	got := CompareLogprobs(specOn, specOff)
	if got.Pass {
		t.Fatal("Pass = true, want false for token ID divergence")
	}
	if got.DivergenceIndex != 1 {
		t.Fatalf("DivergenceIndex = %d, want 1", got.DivergenceIndex)
	}
}

func TestCompareLogprobs_LogprobDivergence(t *testing.T) {
	// Two adjacent float64 values: a single ULP apart. Bit-exact comparison
	// must fail even though the values are numerically close.
	a := math.Nextafter(0.5, 1.0) // one ULP above 0.5
	b := 0.5
	specOn := []TokenLogprob{
		{TokenID: 1, Logprob: -0.5},
		{TokenID: 2, Logprob: a},
	}
	specOff := []TokenLogprob{
		{TokenID: 1, Logprob: -0.5},
		{TokenID: 2, Logprob: b},
	}
	got := CompareLogprobs(specOn, specOff)
	if got.Pass {
		t.Fatal("Pass = true, want false for a single-ULP logprob divergence (bit-exact, no tolerance)")
	}
	if got.DivergenceIndex != 1 {
		t.Fatalf("DivergenceIndex = %d, want 1", got.DivergenceIndex)
	}
}

func TestCompareLogprobs_SignedZeroDivergence(t *testing.T) {
	// +0.0 and -0.0 are ==-equal but have different bit patterns. A bit-exact
	// witness must reject the sign difference.
	specOn := []TokenLogprob{{TokenID: 1, Logprob: 0.0}}
	specOff := []TokenLogprob{{TokenID: 1, Logprob: math.Copysign(0.0, -1.0)}}
	got := CompareLogprobs(specOn, specOff)
	if got.Pass {
		t.Fatal("Pass = true for +0.0 vs -0.0; a bit-exact witness must reject the sign difference")
	}
	if got.DivergenceIndex != 0 {
		t.Fatalf("DivergenceIndex = %d, want 0", got.DivergenceIndex)
	}
}

func TestCompareLogprobs_LengthMismatch(t *testing.T) {
	specOn := []TokenLogprob{
		{TokenID: 1, Logprob: -0.5},
		{TokenID: 2, Logprob: -1.25},
	}
	specOff := []TokenLogprob{
		{TokenID: 1, Logprob: -0.5},
	}
	got := CompareLogprobs(specOn, specOff)
	if got.Pass {
		t.Fatal("Pass = true, want false for length mismatch")
	}
	// The shared prefix is identical; the divergence is the extra element in
	// specOn, at index 1 (the length of the shorter sequence).
	if got.DivergenceIndex != 1 {
		t.Fatalf("DivergenceIndex = %d, want 1", got.DivergenceIndex)
	}
	if got.SpecOnLength != 2 || got.SpecOffLength != 1 {
		t.Fatalf("lengths = %d/%d, want 2/1", got.SpecOnLength, got.SpecOffLength)
	}
}

func TestMatrix_RecordAndGet(t *testing.T) {
	m := NewMatrix()
	q := Qualification{
		Key:           QualificationKey{Backend: "vulkan", Model: "qwen38:27b-q4"},
		Qualified:     true,
		Parity:        ParityResult{Pass: true, DivergenceIndex: -1},
		Date:          "2026-11-15T00:00:00Z",
		EngineSHA:     "mock",
		SampleCount:   5,
		EvidenceClass: "MOCK",
	}
	m.Record(q)

	got, ok := m.Get("vulkan", "qwen38:27b-q4")
	if !ok {
		t.Fatal("Get returned ok=false for a recorded pair")
	}
	if got.Key != q.Key || got.Qualified != q.Qualified || got.EvidenceClass != q.EvidenceClass {
		t.Fatalf("Get = %+v, want %+v", got, q)
	}
	// A MOCK entry is a placeholder: it must NOT qualify the pair for
	// default-on (the "no non-qualified pairs" rule).
	if m.IsQualified("vulkan", "qwen38:27b-q4") {
		t.Fatal("IsQualified = true for a MOCK pair; only HW-WITNESSED pairs qualify for default-on")
	}
	// A live-silicon (HW-WITNESSED) qualified pair DOES qualify.
	m.Record(Qualification{
		Key:           QualificationKey{Backend: "vulkan", Model: "qwen38:27b-q4-hw"},
		Qualified:     true,
		Parity:        ParityResult{Pass: true, DivergenceIndex: -1},
		EvidenceClass: "HW-WITNESSED",
	})
	if !m.IsQualified("vulkan", "qwen38:27b-q4-hw") {
		t.Fatal("IsQualified = false for a recorded HW-WITNESSED qualified pair")
	}
}

func TestMatrix_IsQualified_Unqualified(t *testing.T) {
	m := NewMatrix()
	if m.IsQualified("vulkan", "qwen38:27b-q4") {
		t.Fatal("IsQualified = true for an unrecorded pair on an empty matrix")
	}
	// Record a pair that is NOT qualified; IsQualified must still be false.
	m.Record(Qualification{
		Key:       QualificationKey{Backend: "cuda", Model: "qwen38:27b-q4"},
		Qualified: false,
	})
	if m.IsQualified("cuda", "qwen38:27b-q4") {
		t.Fatal("IsQualified = true for a recorded-but-unqualified pair")
	}
}

func TestDefaultMatrix_HasMockPair(t *testing.T) {
	m := DefaultMatrix()
	if m == nil {
		t.Fatal("DefaultMatrix returned nil")
	}
	// The mock-qualified pair must be present in the matrix.
	q, ok := m.Get("vulkan", "qwen38:27b-q4")
	if !ok {
		t.Fatal("DefaultMatrix: Get returned ok=false for the mock pair")
	}
	if q.EvidenceClass != "MOCK" {
		t.Fatalf("mock pair EvidenceClass = %q, want MOCK", q.EvidenceClass)
	}
	if !q.Parity.Pass {
		t.Fatal("mock pair Parity.Pass = false, want true")
	}
	if q.SampleCount != 5 {
		t.Fatalf("mock pair SampleCount = %d, want 5", q.SampleCount)
	}
	// The mock pair is a placeholder: it must NOT qualify for default-on.
	// Default-on applies only to HW-WITNESSED pairs (the "no non-qualified
	// pairs" rule).
	if m.IsQualified("vulkan", "qwen38:27b-q4") {
		t.Fatal("DefaultMatrix: mock pair qualifies for default-on; only HW-WITNESSED pairs may")
	}
	// DefaultMatrix must be a stable package-level matrix: repeated calls
	// return the same instance.
	if DefaultMatrix() != m {
		t.Fatal("DefaultMatrix returned a different instance on a second call")
	}
}
