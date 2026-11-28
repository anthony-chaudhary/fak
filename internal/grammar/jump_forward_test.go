package grammar

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestMatcherFindJumpForwardTokens(t *testing.T) {
	pieces := []string{
		`{"name":"paint","arguments":{"color":"`, // 0
		`r`, `e`, `d`, `b`, `l`, `u`,             // 1..6
		`"}}`, // 7
	}
	dec := make(byteTokenDecoder, len(pieces))
	for i := range pieces {
		dec[i] = []byte(pieces[i])
	}
	mask, err := Compile([]ToolSpec{{
		Name:   "paint",
		Schema: []byte(`{"type":"object","properties":{"color":{"type":"string","enum":["red","blue"]}},"required":["color"]}`),
	}}, dec, CompileOptions{EOS: -1})
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewMatcher(mask)
	if err != nil {
		t.Fatal(err)
	}

	// At the start only the envelope prefix token fits; the enum then branches.
	if got := m.FindJumpForwardTokens(len(dec), 0); !reflect.DeepEqual(got, []int{0}) {
		t.Fatalf("jump from start = %v, want [0]", got)
	}

	// After committing "r" the rest of "red" and the closing token are forced.
	if err := m.AdvanceByAccepted([]int{0, 1}); err != nil {
		t.Fatal(err)
	}
	before := m.History()
	got := m.FindJumpForwardTokens(len(dec), 0)
	if !reflect.DeepEqual(got, []int{2, 3, 7}) {
		t.Fatalf("jump after r = %v, want [2 3 7]", got)
	}
	if !reflect.DeepEqual(m.History(), before) {
		t.Fatalf("FindJumpForwardTokens mutated matcher history: %v -> %v", before, m.History())
	}

	// maxTokens caps the run; the cap never changes matcher state either.
	if got := m.FindJumpForwardTokens(len(dec), 2); !reflect.DeepEqual(got, []int{2, 3}) {
		t.Fatalf("capped jump = %v, want [2 3]", got)
	}

	// Forced tokens are legal to commit as a batch.
	if err := m.AdvanceByAccepted(got); err != nil {
		t.Fatalf("committing forced tokens: %v", err)
	}
}
