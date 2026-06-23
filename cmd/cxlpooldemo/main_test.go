package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestRunProducesPoolStory asserts the demo runs deterministically and emits the
// load-bearing claims: the pool topology with the fabric-shareable CXL tier, the
// three-way fleet economics with the both-axes savings headline, and the cross-tenant
// trust gate refusing a poisoned / private / wrong-model cell.
func TestRunProducesPoolStory(t *testing.T) {
	var buf bytes.Buffer
	if err := run(&buf); err != nil {
		t.Fatalf("run: %v", err)
	}
	out := buf.String()
	for _, want := range []string{
		"Pool topology",
		"cxl_hdm",               // CXL zero-copy share kind
		"yes (one shared copy)", // CXL is fabric-shareable
		"coherent CXL pool",     // the winning regime
		"28000 prefill tokens saved",
		"448MB of memory deduplicated",
		"Cross-tenant reuse gate",
		"REUSE",
		"REFUSE",
		"model_mismatch",
		"scope_denied",
		"taint_denied",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("demo output missing %q\n---\n%s", want, out)
		}
	}
}

// TestRunDeterministic guards the no-wall-clock property: two runs are byte-identical.
func TestRunDeterministic(t *testing.T) {
	var a, b bytes.Buffer
	_ = run(&a)
	_ = run(&b)
	if a.String() != b.String() {
		t.Fatalf("demo output is not deterministic across runs")
	}
}
