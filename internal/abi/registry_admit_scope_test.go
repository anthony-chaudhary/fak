package abi

import (
	"context"
	"fmt"
	"testing"
)

// uncRA is an unconditional result gate (no CallScope): folded into every result.
type uncRA struct{ id string }

func (uncRA) Admit(context.Context, *ToolCall, *Result) Verdict {
	return Verdict{Kind: VerdictAllow}
}
func (uncRA) Caps() []Capability { return nil }

// scpRA scopes itself to specific tools via CallScope (the same interface
// adjudicators use — one scoping concept for both folds).
type scpRA struct {
	id    string
	tools []string
}

func (scpRA) Admit(context.Context, *ToolCall, *Result) Verdict {
	return Verdict{Kind: VerdictAllow}
}
func (scpRA) Caps() []Capability { return nil }
func (s scpRA) Tools() []string  { return s.tools }

func raIDs(ras []ResultAdmitter) []string {
	out := make([]string, 0, len(ras))
	for _, ra := range ras {
		switch v := ra.(type) {
		case uncRA:
			out = append(out, v.id)
		case scpRA:
			out = append(out, v.id)
		}
	}
	return out
}

// TestResultAdmittersForUnconditional: with no tool-scoped gate, ResultAdmittersFor
// returns the full chain for every tool — identical to ResultAdmitters(). The
// current result-side gate set (normgate/ctxmmu/ifc, all cross-cutting) is folded
// exactly as before.
func TestResultAdmittersForUnconditional(t *testing.T) {
	ResetForTest()
	defer ResetForTest()
	RegisterResultAdmitter(0, uncRA{"a"})
	RegisterResultAdmitter(1, uncRA{"b"})
	for _, tool := range []string{"x", "y", ""} {
		if got := raIDs(ResultAdmittersFor(&ToolCall{Tool: tool})); !eqStrs(got, []string{"a", "b"}) {
			t.Errorf("ResultAdmittersFor(tool=%q) = %v, want full chain [a b]", tool, got)
		}
	}
	if got := raIDs(ResultAdmitters()); !eqStrs(got, []string{"a", "b"}) {
		t.Errorf("ResultAdmitters() = %v, want [a b]", got)
	}
}

// TestResultAdmittersForScoped: a tool-scoped gate is folded ONLY into results for
// its tool, merged with unconditional gates in rank order.
func TestResultAdmittersForScoped(t *testing.T) {
	ResetForTest()
	defer ResetForTest()
	RegisterResultAdmitter(0, uncRA{"u"})
	RegisterResultAdmitter(5, scpRA{"s_fetch", []string{"fetch"}})
	RegisterResultAdmitter(9, uncRA{"u2"})

	if got := raIDs(ResultAdmittersFor(&ToolCall{Tool: "fetch"})); !eqStrs(got, []string{"u", "s_fetch", "u2"}) {
		t.Errorf("fetch gate chain = %v, want [u s_fetch u2]", got)
	}
	if got := raIDs(ResultAdmittersFor(&ToolCall{Tool: "read"})); !eqStrs(got, []string{"u", "u2"}) {
		t.Errorf("read gate chain = %v, want [u u2] (scoped gate excluded)", got)
	}
}

// TestResultAdmittersForScalesWithUnrelatedTools: with 500 gates each scoped to a
// distinct tool + 1 unconditional, a result for an unrelated tool folds only the 1
// unconditional gate — independent of the 500. The exfil/quarantine floor stays
// O(1) for unrelated tools as result-side features accumulate.
func TestResultAdmittersForScalesWithUnrelatedTools(t *testing.T) {
	ResetForTest()
	defer ResetForTest()
	RegisterResultAdmitter(0, uncRA{"u"})
	const n = 500
	for i := 0; i < n; i++ {
		RegisterResultAdmitter(i+1, scpRA{fmt.Sprintf("s%d", i), []string{fmt.Sprintf("tool-%d", i)}})
	}
	if got := ResultAdmittersFor(&ToolCall{Tool: "unrelated"}); len(got) != 1 {
		t.Fatalf("unrelated result folds %d gates; want 1 (independent of %d scoped gates)", len(got), n)
	}
	if got := ResultAdmittersFor(&ToolCall{Tool: "tool-7"}); len(got) != 2 {
		t.Fatalf("tool-7 result folds %d gates; want 2 (unconditional + its own scoped gate)", len(got))
	}
}

// TestResultAdmittersForZeroAlloc: per-tool gate selection is allocation-free on
// both the indexed and fallback paths.
func TestResultAdmittersForZeroAlloc(t *testing.T) {
	ResetForTest()
	defer ResetForTest()
	RegisterResultAdmitter(0, uncRA{"u"})
	for i := 0; i < 64; i++ {
		RegisterResultAdmitter(i+1, scpRA{fmt.Sprintf("s%d", i), []string{fmt.Sprintf("tool-%d", i)}})
	}
	fc := &ToolCall{Tool: "tool-3"} // indexed
	rc := &ToolCall{Tool: "nope"}   // fallback (unconditional only)
	if a := testing.AllocsPerRun(200, func() { benchSink += len(ResultAdmittersFor(fc)) }); a != 0 {
		t.Errorf("ResultAdmittersFor(indexed tool) allocates %.2f/op; want 0", a)
	}
	if a := testing.AllocsPerRun(200, func() { benchSink += len(ResultAdmittersFor(rc)) }); a != 0 {
		t.Errorf("ResultAdmittersFor(fallback tool) allocates %.2f/op; want 0", a)
	}
}
