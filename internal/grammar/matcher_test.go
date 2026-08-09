package grammar

import (
	"reflect"
	"testing"
)

type byteTokenDecoder [][]byte

func (d byteTokenDecoder) TokenBytes(id int) []byte {
	if id < 0 || id >= len(d) {
		return nil
	}
	return d[id]
}

func TestMatcherCommitsOnlyAcceptedSpeculativeTokens(t *testing.T) {
	// The enum leaves a partially constrained region where both "red" and
	// "blue" are legal drafts. The target accepts the common prefix plus
	// "red" and rejects the draft suffix that starts "blue".
	pieces := []string{
		`{"name":"paint","arguments":{"color":"`,
		`r`, `e`, `d`, `b`, `l`, `u`, `e`, `"}}`,
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

	real, err := NewMatcher(mask)
	if err != nil {
		t.Fatal(err)
	}
	if err := real.AdvanceByAccepted([]int{0}); err != nil {
		t.Fatal(err)
	}

	// Speculative state explores a legal blue branch without mutating real.
	draft := real.Fork()
	if err := draft.AdvanceByAccepted([]int{4, 5, 6, 7}); err != nil {
		t.Fatalf("advance draft fork: %v", err)
	}
	if got, want := real.History(), []int{0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("draft advanced real matcher: got %v want %v", got, want)
	}

	accepted := []int{1, 2, 3} // verifier accepted red; blue fork was rejected
	if err := real.AdvanceByAccepted(accepted); err != nil {
		t.Fatalf("commit accepted tokens: %v", err)
	}
	baseline, _ := NewMatcher(mask)
	if err := baseline.AdvanceByAccepted(append([]int{0}, accepted...)); err != nil {
		t.Fatal(err)
	}
	if got, want := real.History(), baseline.History(); !reflect.DeepEqual(got, want) {
		t.Fatalf("real state includes rejected draft: got %v want %v", got, want)
	}
}

func TestMatcherAdvanceIsTransactional(t *testing.T) {
	dec := byteTokenDecoder{[]byte(`{"name":"x","arguments":{}}`), []byte(`!`)}
	mask, err := Compile([]ToolSpec{{Name: "x", Schema: []byte(`{"type":"object"}`)}}, dec, CompileOptions{EOS: -1})
	if err != nil {
		t.Fatal(err)
	}
	m, _ := NewMatcher(mask)
	before := m.History()
	if err := m.AdvanceByAccepted([]int{1}); err == nil {
		t.Fatal("expected off-grammar accepted token to be refused")
	}
	if got := m.History(); !reflect.DeepEqual(got, before) {
		t.Fatalf("failed advance mutated matcher: got %v want %v", got, before)
	}
}
