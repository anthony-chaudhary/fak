package abi

import (
	"context"
	"testing"
)

// namedAdj is a rung that self-reports a By (its name) and optionally a per-call
// suffix, so RungName's suffix-stripping is exercised.
type namedAdj struct {
	by     string
	suffix string // appended in parens on every verdict, e.g. "(off)"
}

func (n namedAdj) Adjudicate(context.Context, *ToolCall) Verdict {
	by := n.by
	if n.suffix != "" {
		by = n.by + "(" + n.suffix + ")"
	}
	return Verdict{Kind: VerdictDefer, By: by}
}
func (namedAdj) Caps() []Capability { return nil }

// unnamedAdj reports no By at all — it cannot be addressed by name.
type unnamedAdj struct{}

func (unnamedAdj) Adjudicate(context.Context, *ToolCall) Verdict { return Verdict{Kind: VerdictDefer} }
func (unnamedAdj) Caps() []Capability                            { return nil }

func TestRungName_ReadsByAndStripsSuffix(t *testing.T) {
	cases := []struct {
		a    Adjudicator
		want string
	}{
		{namedAdj{by: "grammar"}, "grammar"},
		{namedAdj{by: "ifc-sink", suffix: "off"}, "ifc-sink"},
		{namedAdj{by: "ifc-sink", suffix: "authorized"}, "ifc-sink"},
		{unnamedAdj{}, ""},
		{nil, ""},
	}
	for _, c := range cases {
		if got := RungName(c.a); got != c.want {
			t.Errorf("RungName(%v) = %q, want %q", c.a, got, c.want)
		}
	}
}

func names(chain []Adjudicator) []string { return RungNames(chain) }

func TestWithoutRung_RemovesExactlyTheNamedRung(t *testing.T) {
	chain := []Adjudicator{
		namedAdj{by: "grammar"},
		namedAdj{by: "preflight"},
		namedAdj{by: "ifc-sink", suffix: "off"}, // suffix must not defeat the match
		namedAdj{by: "monitor"},
	}

	masked, removed := WithoutRung(chain, "ifc-sink")
	if removed != 1 {
		t.Fatalf("removed = %d, want 1", removed)
	}
	got := names(masked)
	want := []string{"grammar", "preflight", "monitor"}
	if !eqStrs(got, want) {
		t.Errorf("masked chain = %v, want %v", got, want)
	}
	// The input chain is never mutated — masking returns a fresh copy.
	if !eqStrs(names(chain), []string{"grammar", "preflight", "ifc-sink", "monitor"}) {
		t.Errorf("input chain was mutated: %v", names(chain))
	}
}

func TestWithoutRung_UnknownNameIsANoOp(t *testing.T) {
	chain := []Adjudicator{namedAdj{by: "grammar"}, namedAdj{by: "monitor"}}
	masked, removed := WithoutRung(chain, "does-not-exist")
	if removed != 0 {
		t.Errorf("removed = %d, want 0 (no rung named that)", removed)
	}
	if !eqStrs(names(masked), []string{"grammar", "monitor"}) {
		t.Errorf("masked = %v, want full chain", names(masked))
	}
}

func TestWithoutRung_EmptyNameRemovesNothing(t *testing.T) {
	// An unnamed rung (By=="") must NOT be removed by an empty-name mask — a no-op,
	// never an accidental ablation of every unnamed rung.
	chain := []Adjudicator{unnamedAdj{}, namedAdj{by: "monitor"}}
	masked, removed := WithoutRung(chain, "")
	if removed != 0 {
		t.Errorf("removed = %d, want 0 (empty name matches nothing)", removed)
	}
	if len(masked) != 2 {
		t.Errorf("masked len = %d, want 2", len(masked))
	}
}
