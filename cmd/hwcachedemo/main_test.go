package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestRunProducesHardwareAwareStory asserts the demo runs deterministically and emits
// the load-bearing claims: the tier ladder including CXL, zero-copy sharing, the
// demote-not-evict walk, and the LRU-vs-tiered savings headline.
func TestRunProducesHardwareAwareStory(t *testing.T) {
	var buf bytes.Buffer
	if err := run(&buf); err != nil {
		t.Fatalf("run: %v", err)
	}
	out := buf.String()
	for _, want := range []string{
		"Residency tier ladder",
		"cxl",
		"cxl_hdm", // CXL zero-copy share kind
		"A hot 4000-token prefix under escalating memory pressure",
		"demote  hbm -> dram",
		"demote  numa_far -> cxl",
		"spill   cxl -> disk",
		"recompute_cheaper_than_retain",
		"28000 prefill tokens saved",
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
