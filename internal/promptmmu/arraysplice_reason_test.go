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

// arraySpliceVocabulary is the closed set as DECLARED, name paired with wire token. It is the
// one list the test below reads, so a member added to the const block without being added here
// is caught by the coverage check rather than silently going unasserted.
var arraySpliceVocabulary = []struct{ name, token string }{
	{"ArrayOffsetsResolved", ArrayOffsetsResolved},
	{"ArrayNotJSONObject", ArrayNotJSONObject},
	{"ArrayUndecodable", ArrayUndecodable},
	{"ArrayEmptyBody", ArrayEmptyBody},
	{"ArrayKeyAbsent", ArrayKeyAbsent},
	{"ArrayValueNotArray", ArrayValueNotArray},
	{"ArrayNoElements", ArrayNoElements},
	{"ArrayNoBreakpoint", ArrayNoBreakpoint},
}

// TestArraySpliceReasonsArePairwiseDistinct is the anti-alias guard the Array* set was missing
// while its Skip* twin has carried one (system_structural_reason_test.go asserts
// SkipUndecodableSystem != SkipNoSystem). Every other assertion in this package compares an
// OBSERVED reason to the constant it expects, so two constants collapsed onto one wire token
// ship green: the malformed arm and the no-anchor arm would report the same string,
// ArrayReasonIsStructural would answer the same for both, and every existing check would still
// hold. That is #5442's indistinguishable-result defect reintroduced one layer down — in the
// vocabulary instead of in the control flow — where nothing here could see it.
//
// It also pins that every declared member is exercised by arraySpliceReasonCases, which is
// #5442's third acceptance bullet (declaration sites move together). ArrayUndecodable is the
// ONE deliberate exemption, stated rather than omitted: it is unreachable from this seam by
// construction, because arrRaw is a verbatim slice of an already-validated document, so no
// input can drive decodeArrayElements to fail here. Its reachable twin is witnessed a layer up
// in internal/syspromptmmu and internal/gateway.
func TestArraySpliceReasonsArePairwiseDistinct(t *testing.T) {
	byToken := make(map[string]string, len(arraySpliceVocabulary))
	for _, m := range arraySpliceVocabulary {
		if prev, dup := byToken[m.token]; dup {
			t.Errorf("%s and %s share the wire token %q — the two outcomes are indistinguishable "+
				"to any caller that routes on the reason", prev, m.name, m.token)
			continue
		}
		byToken[m.token] = m.name
	}

	exercised := make(map[string]bool, len(arraySpliceReasonCases))
	for _, tc := range arraySpliceReasonCases {
		exercised[tc.want] = true
	}
	for _, m := range arraySpliceVocabulary {
		if m.token == ArrayUndecodable {
			continue // unreachable from this seam; see the doc comment above.
		}
		if !exercised[m.token] {
			t.Errorf("%s (%q) is declared but no arraySpliceReasonCases entry produces it", m.name, m.token)
		}
	}
}

// TestMalformedIsDistinguishableFromNoCandidate is #5442's headline pair asserted on the two
// OBSERVED reasons against EACH OTHER, not against expected constants: a body that could not be
// parsed at all and a well-formed body that simply carries no cache_control anchor must not
// produce one result. The malformed fixture is a truncated document — genuinely unparseable,
// unlike the table's `["a"]`, which is valid JSON that merely is not an object.
func TestMalformedIsDistinguishableFromNoCandidate(t *testing.T) {
	malformedRaw := []byte(`{"system":[{"type":"text",`)
	noCandidateRaw := []byte(`{"system":[{"type":"text","text":"foo"}]}`)

	mBI, mPE, mLE, mReason := ArraySplicePointsWithReason(malformedRaw, "system")
	_, _, _, nReason := ArraySplicePointsWithReason(noCandidateRaw, "system")

	if mReason == nReason {
		t.Fatalf("malformed JSON and a candidate-free body collapsed into one reason %q — a caller "+
			"cannot route a parse failure away from the benign idle", mReason)
	}
	if !ArrayReasonIsStructural(mReason) {
		t.Errorf("ArrayReasonIsStructural(%q) = false; a body that would not parse is a bug signal", mReason)
	}
	if ArrayReasonIsStructural(nReason) {
		t.Errorf("ArrayReasonIsStructural(%q) = true; a missing cache_control anchor is the ordinary "+
			"shape of a request that is not spliceable", nReason)
	}
	// Fail-safe is unchanged by the split: the malformed arm must not start handing out byte
	// offsets derived from a body that never parsed.
	if mBI != 0 || mPE != 0 || mLE != 0 {
		t.Errorf("malformed body returned offsets (%d,%d,%d), want all zero", mBI, mPE, mLE)
	}
	// And the bare-ok form still answers false for BOTH — the distinction is additive, never a
	// change to the contract the remaining ok-form callers hold.
	if _, _, _, ok := ArraySplicePoints(malformedRaw, "system"); ok {
		t.Error("malformed body: ArraySplicePoints ok = true, want false")
	}
	if _, _, _, ok := ArraySplicePoints(noCandidateRaw, "system"); ok {
		t.Error("candidate-free body: ArraySplicePoints ok = true, want false")
	}
}
