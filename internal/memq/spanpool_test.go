package memq

import (
	"context"
	"reflect"
	"testing"
)

// spanFixture builds the #4014 corpus: an isolated cell whose descriptor spikes on
// every intent term (step 0), a coherent 3-cell span (steps 3-5) that each hit three
// terms, and zero-relevance fillers so the spike sits among quiet neighbors.
//
// Raw overlap with "refund fee policy escalation gold": spike=5, span cells=3 each,
// fillers=0 — so plain relevance ranks the spike first, while a 3-wide pooled pass
// scores the span middle 3+3+3=9 > the spike's 0+5+0=5.
func spanFixture() *MemStore {
	m := NewMemStore()
	m.Add("search_notes", "tool_result", DurabilitySession,
		[]byte("refund fee policy escalation gold summary in one place"), false) // cell:0 — the spike, raw 5
	m.Add("clock", "system", DurabilitySession, []byte("weather is sunny"), false)                        // cell:1 — filler, raw 0
	m.Add("clock", "system", DurabilitySession, []byte("humid again this evening"), false)                // cell:2 — filler, raw 0
	m.Add("kb", "tool_result", DurabilitySession, []byte("refund fee schedule for gold tier"), false)     // cell:3 — span, raw 3
	m.Add("kb", "tool_result", DurabilitySession, []byte("escalation policy for refund disputes"), false) // cell:4 — span, raw 3
	m.Add("kb", "tool_result", DurabilitySession, []byte("gold tier fee escalation ladder"), false)       // cell:5 — span, raw 3
	m.Add("clock", "system", DurabilitySession, []byte("light breeze from the west"), false)              // cell:6 — filler, raw 0
	return m
}

// TestSpanPoolValidateFailsClosed pins the authored-window contract: the span rank key
// never guesses a kernel — a missing, zero, negative, or even window is refused before
// the query runs, and a well-formed odd window validates.
func TestSpanPoolValidateFailsClosed(t *testing.T) {
	bad := []Query{
		{Ops: []Op{{Kind: OpRank, By: RankRelevanceSpan}}},             // window unset
		{Ops: []Op{{Kind: OpRank, By: RankRelevanceSpan, Window: 0}}},  // zero
		{Ops: []Op{{Kind: OpRank, By: RankRelevanceSpan, Window: -3}}}, // negative
		{Ops: []Op{{Kind: OpRank, By: RankRelevanceSpan, Window: 2}}},  // even — no centered kernel
	}
	for i, q := range bad {
		if err := Validate(q); err == nil {
			t.Errorf("query %d validated but should be refused fail-closed: %+v", i, q)
		}
	}
	good := Query{Ops: []Op{{Kind: OpScan}, {Kind: OpRank, By: RankRelevanceSpan, Window: 3}}}
	if err := Validate(good); err != nil {
		t.Fatalf("well-formed span-rank query refused: %v", err)
	}
}

// TestSpanPooledRelevance is the #4014 acceptance: under a tight top-k budget a
// coherent 3-cell span beats an isolated spike once relevance is pooled over a 3-cell
// step window, while (a) the plain RankRelevance path still ranks the spike first —
// the default-off, today's-ranking-unchanged witness — and (b) window=1 (the identity
// kernel) reproduces plain relevance's ordering exactly.
func TestSpanPooledRelevance(t *testing.T) {
	ctx := context.Background()
	const intent = "refund fee policy escalation gold"

	rankQuery := func(by string, window, k int) Query {
		return Query{Intent: intent, Ops: []Op{
			{Kind: OpScan},
			{Kind: OpRank, By: by, Desc: true, Window: window},
			{Kind: OpLimit, K: k},
		}}
	}

	// Default path (no span key anywhere): the isolated spike outranks the span —
	// today's per-cell-independent ranking, unchanged.
	plain, err := Run(ctx, spanFixture(), rankQuery(RankRelevance, 0, 3), Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := ids(plain.Working), []string{"cell:0", "cell:3", "cell:4"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("plain relevance top-3 = %v, want %v (today's ranking must be unchanged)", got, want)
	}

	// window=1 is the identity kernel: the span rank key reproduces plain relevance
	// exactly (the degenerate case the issue pins as the equivalence witness).
	ident, err := Run(ctx, spanFixture(), rankQuery(RankRelevanceSpan, 1, 3), Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ids(ident.Working), ids(plain.Working)) {
		t.Fatalf("window=1 span ranking %v != plain relevance ranking %v", ids(ident.Working), ids(plain.Working))
	}

	// window=3: the coherent span (cells 3-5, middle first) takes the whole top-3;
	// the spike is out.
	pooled, err := Run(ctx, spanFixture(), rankQuery(RankRelevanceSpan, 3, 3), Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := ids(pooled.Working), []string{"cell:4", "cell:3", "cell:5"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("pooled top-3 = %v, want %v (the span must beat the spike)", got, want)
	}
	for _, c := range pooled.Working {
		if c.ID == "cell:0" {
			t.Fatal("the isolated spike survived the pooled top-3 cut")
		}
	}
}

// TestSpanPoolThreadBoundary pins the group seam: cells in different thread groups
// never lend each other score, a singleton pools to its own (zero-padded) score, and
// an unordered (Step < 0) cell keeps its raw score untouched.
func TestSpanPoolThreadBoundary(t *testing.T) {
	cells := []Cell{
		{ID: "a:0", Step: 0, Attrs: map[string]string{SpanGroupAttr: "a"}},
		{ID: "b:1", Step: 1, Attrs: map[string]string{SpanGroupAttr: "b"}},
		{ID: "a:2", Step: 2, Attrs: map[string]string{SpanGroupAttr: "a"}},
		{ID: "loose", Step: -1},
	}
	score := map[string]int{"a:0": 4, "b:1": 4, "a:2": 4, "loose": 7}
	pooled := poolSpanScores(cells, score, 3)
	// Thread a's two cells are adjacent WITHIN their group: each pools the other.
	if pooled["a:0"] != 8 || pooled["a:2"] != 8 {
		t.Errorf("thread-a pooled scores = %d/%d, want 8/8", pooled["a:0"], pooled["a:2"])
	}
	// Thread b's singleton has no in-group neighbors: zero padding leaves its own score.
	if pooled["b:1"] != 4 {
		t.Errorf("thread-b singleton pooled = %d, want 4 (no cross-thread smear)", pooled["b:1"])
	}
	// A Step<0 cell has no ordinal neighborhood: the raw score stands.
	if pooled["loose"] != 7 {
		t.Errorf("unordered cell pooled = %d, want raw 7", pooled["loose"])
	}
}
