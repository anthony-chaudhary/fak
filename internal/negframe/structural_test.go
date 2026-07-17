package negframe

import (
	"reflect"
	"testing"
)

func TestDeMorganCandidateAnd(t *testing.T) {
	t.Parallel()
	input := "Choose not (alpha and beta) today."
	got := DetectStructuralNegation(input)
	want := []StructuralFinding{{
		ScopeStart:  11,
		ScopeEnd:    27,
		Scope:       "(alpha and beta)",
		Operator:    "and",
		Distributed: []string{"not alpha", "not beta"},
	}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("DetectStructuralNegation(%q) = %#v, want %#v", input, got, want)
	}
}

func TestDeMorganCandidateOr(t *testing.T) {
	t.Parallel()
	input := "not (red or blue)"
	got := DetectStructuralNegation(input)
	want := []StructuralFinding{{
		ScopeStart:  4,
		ScopeEnd:    len(input),
		Scope:       "(red or blue)",
		Operator:    "or",
		Distributed: []string{"not red", "not blue"},
	}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("DetectStructuralNegation(%q) = %#v, want %#v", input, got, want)
	}
}

func TestDeMorganCandidateBareNotIsNegative(t *testing.T) {
	t.Parallel()
	for _, input := range []string{
		"This is not red.",
		"not (alpha)",
		"not alpha and beta",
		"`not (alpha and beta)`",
		"```text\nnot (alpha and beta)\n```",
	} {
		if got := DetectStructuralNegation(input); len(got) != 0 {
			t.Errorf("DetectStructuralNegation(%q) = %#v, want none", input, got)
		}
	}
}

func TestDeMorganCandidateNestedScope(t *testing.T) {
	t.Parallel()
	input := "not ((alpha or beta) and gamma)"
	got := DetectStructuralNegation(input)
	if len(got) != 1 || got[0].Operator != "and" || !reflect.DeepEqual(got[0].Distributed, []string{"not (alpha or beta)", "not gamma"}) {
		t.Fatalf("nested finding = %#v", got)
	}
}

func TestDeMorganCandidatePureAndClassifierUnchanged(t *testing.T) {
	t.Parallel()
	input := "not (alpha and beta)\nDo not forget to stamp."
	before := Classify("fixture.md", input)
	first := DetectStructuralNegation(input)
	second := DetectStructuralNegation(input)
	after := Classify("fixture.md", input)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("nondeterministic: first=%#v second=%#v", first, second)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("Classify changed: before=%#v after=%#v", before, after)
	}
}
