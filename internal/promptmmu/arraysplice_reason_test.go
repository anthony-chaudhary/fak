package promptmmu

import "testing"

// arraysplice_reason_test.go — the closed set ArraySplicePointsWithReason names, and the
// proof that adding it did not move ArraySplicePoints' existing contract by a single bit
// (#5442). The consumer-facing acceptance lives in internal/syspromptmmu, driven through the
// two real callers; this file only pins the vocabulary and the compatibility.

// arraySpliceReasonCases pairs one input shape with the single reason it must name.
var arraySpliceReasonCases = []struct {
	name   string
	raw    []byte
	want   string
	offset bool // true iff offsets are expected (reason == ArrayOffsetsResolved)
}{
	{"anchored", []byte(`{"system":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}}]}`), ArrayOffsetsResolved, true},
	{"empty-body", nil, ArrayEmptyBody, false},
	{"not-an-object", []byte(`["a"]`), ArrayNotJSONObject, false},
	{"key-absent", []byte(`{"model":"x"}`), ArrayKeyAbsent, false},
	{"bare-string", []byte(`{"system":"a plain string"}`), ArrayValueNotArray, false},
	{"json-null", []byte(`{"system":null}`), ArrayValueNotArray, false},
	{"object-value", []byte(`{"system":{"type":"text"}}`), ArrayValueNotArray, false},
	{"empty-array", []byte(`{"system":[]}`), ArrayNoElements, false},
	{"no-cache-control", []byte(`{"system":[{"type":"text","text":"foo"}]}`), ArrayNoBreakpoint, false},
}

// TestArraySplicePointsWithReasonNamesEachArm asserts every arm reports its OWN reason —
// specifically that a value which is not an array (a bare string, JSON null, an object) is
// NOT reported as a decode failure, and that the no-breakpoint idle stays its own bucket.
func TestArraySplicePointsWithReasonNamesEachArm(t *testing.T) {
	for _, tc := range arraySpliceReasonCases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, _, reason := ArraySplicePointsWithReason(tc.raw, "system")
			if reason != tc.want {
				t.Errorf("reason = %q, want %q", reason, tc.want)
			}
			// Only a malformed body or an array-shaped value that would not decode is a bug
			// signal. Every other miss is the ordinary shape of an unspliceable request.
			wantStructural := tc.want == ArrayNotJSONObject || tc.want == ArrayUndecodable
			if got := ArrayReasonIsStructural(reason); got != wantStructural {
				t.Errorf("ArrayReasonIsStructural(%q) = %v, want %v", reason, got, wantStructural)
			}
		})
	}
}

// TestArraySplicePointsContractUnmoved proves the bare-ok form is bit-identical to what it
// was before the reason was threaded through it: ok is true exactly when the reason is
// ArrayOffsetsResolved, and the three offsets on the success path are the same values.
// Callers outside this package (internal/agent, internal/gateway) still hold the ok form.
func TestArraySplicePointsContractUnmoved(t *testing.T) {
	for _, tc := range arraySpliceReasonCases {
		t.Run(tc.name, func(t *testing.T) {
			bi, pe, le, ok := ArraySplicePoints(tc.raw, "system")
			if ok != tc.offset {
				t.Fatalf("ok = %v, want %v", ok, tc.offset)
			}
			rbi, rpe, rle, reason := ArraySplicePointsWithReason(tc.raw, "system")
			if ok != (reason == ArrayOffsetsResolved) {
				t.Errorf("ok=%v disagrees with reason=%q", ok, reason)
			}
			if bi != rbi || pe != rpe || le != rle {
				t.Errorf("offsets diverge: ok-form (%d,%d,%d) vs reason-form (%d,%d,%d)", bi, pe, le, rbi, rpe, rle)
			}
			if !ok && (bi != 0 || pe != 0 || le != 0) {
				t.Errorf("a failed call must return zero offsets, got (%d,%d,%d)", bi, pe, le)
			}
		})
	}
}
