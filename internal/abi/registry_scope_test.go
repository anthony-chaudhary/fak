package abi

import (
	"context"
	"fmt"
	"testing"
)

// uncAdj is an unconditional rung (no CallScope): consulted on every call.
type uncAdj struct{ id string }

func (uncAdj) Adjudicate(context.Context, *ToolCall) Verdict { return Verdict{Kind: VerdictDefer} }
func (uncAdj) Caps() []Capability                            { return nil }

// scpAdj scopes itself to specific tools via CallScope.
type scpAdj struct {
	id    string
	tools []string
}

func (scpAdj) Adjudicate(context.Context, *ToolCall) Verdict { return Verdict{Kind: VerdictDefer} }
func (scpAdj) Caps() []Capability                            { return nil }
func (s scpAdj) Tools() []string                             { return s.tools }

func adjIDs(as []Adjudicator) []string {
	out := make([]string, 0, len(as))
	for _, a := range as {
		switch v := a.(type) {
		case uncAdj:
			out = append(out, v.id)
		case scpAdj:
			out = append(out, v.id)
		}
	}
	return out
}

// TestAdjudicatorsForUnconditional asserts that with no tool-scoped rung
// registered, AdjudicatorsFor returns the full chain for every tool — identical to
// Adjudicators(). This is the backward-compat guarantee: the current all-general
// rung set is folded exactly as before.
func TestAdjudicatorsForUnconditional(t *testing.T) {
	ResetForTest()
	defer ResetForTest()
	RegisterAdjudicator(0, uncAdj{"a"})
	RegisterAdjudicator(1, uncAdj{"b"})
	for _, tool := range []string{"x", "y", ""} {
		if got := adjIDs(AdjudicatorsFor(&ToolCall{Tool: tool})); !eqStrs(got, []string{"a", "b"}) {
			t.Errorf("AdjudicatorsFor(tool=%q) = %v, want full chain [a b]", tool, got)
		}
	}
	if got := adjIDs(Adjudicators()); !eqStrs(got, []string{"a", "b"}) {
		t.Errorf("Adjudicators() = %v, want [a b]", got)
	}
}

// TestAdjudicatorsForScoped asserts a tool-scoped rung is folded ONLY into calls
// for its tool, merged with unconditional rungs in rank order.
func TestAdjudicatorsForScoped(t *testing.T) {
	ResetForTest()
	defer ResetForTest()
	RegisterAdjudicator(0, uncAdj{"u"})
	RegisterAdjudicator(5, scpAdj{"s_write", []string{"write"}})
	RegisterAdjudicator(9, uncAdj{"u2"})

	if got := adjIDs(AdjudicatorsFor(&ToolCall{Tool: "write"})); !eqStrs(got, []string{"u", "s_write", "u2"}) {
		t.Errorf("write chain = %v, want [u s_write u2] (rank order, scoped rung included)", got)
	}
	if got := adjIDs(AdjudicatorsFor(&ToolCall{Tool: "read"})); !eqStrs(got, []string{"u", "u2"}) {
		t.Errorf("read chain = %v, want [u u2] (scoped rung excluded)", got)
	}
}

// TestAdjudicatorsForScalesWithUnrelatedTools is the headline scaling proof: with
// 500 rungs each scoped to a distinct tool plus 1 unconditional rung, a call for an
// UNRELATED tool folds only the 1 unconditional rung — independent of how many
// tool-scoped rungs exist. Adding the 500th per-tool policy costs an unrelated call
// nothing.
func TestAdjudicatorsForScalesWithUnrelatedTools(t *testing.T) {
	ResetForTest()
	defer ResetForTest()
	RegisterAdjudicator(0, uncAdj{"u"})
	const n = 500
	for i := 0; i < n; i++ {
		RegisterAdjudicator(i+1, scpAdj{fmt.Sprintf("s%d", i), []string{fmt.Sprintf("tool-%d", i)}})
	}
	if got := AdjudicatorsFor(&ToolCall{Tool: "unrelated"}); len(got) != 1 {
		t.Fatalf("unrelated call folds %d rungs; want 1 (must be independent of %d scoped rungs)", len(got), n)
	}
	if got := AdjudicatorsFor(&ToolCall{Tool: "tool-7"}); len(got) != 2 {
		t.Fatalf("tool-7 call folds %d rungs; want 2 (unconditional + its own scoped rung)", len(got))
	}
}

// TestAdjudicatorsForZeroAlloc proves the per-tool chain selection is itself
// allocation-free on both the indexed and fallback paths.
func TestAdjudicatorsForZeroAlloc(t *testing.T) {
	ResetForTest()
	defer ResetForTest()
	RegisterAdjudicator(0, uncAdj{"u"})
	for i := 0; i < 64; i++ {
		RegisterAdjudicator(i+1, scpAdj{fmt.Sprintf("s%d", i), []string{fmt.Sprintf("tool-%d", i)}})
	}
	wc := &ToolCall{Tool: "tool-3"} // indexed
	rc := &ToolCall{Tool: "nope"}   // fallback (unconditional only)
	if a := testing.AllocsPerRun(200, func() { benchSink += len(AdjudicatorsFor(wc)) }); a != 0 {
		t.Errorf("AdjudicatorsFor(indexed tool) allocates %.2f/op; want 0", a)
	}
	if a := testing.AllocsPerRun(200, func() { benchSink += len(AdjudicatorsFor(rc)) }); a != 0 {
		t.Errorf("AdjudicatorsFor(fallback tool) allocates %.2f/op; want 0", a)
	}
}
