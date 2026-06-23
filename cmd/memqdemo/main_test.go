package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestDemoRunsAndUpholdsInvariants runs the whole walkthrough to a buffer and asserts
// the load-bearing claims: it completes without error (the run returns nil only if no
// sealed cell leaked into a render), the safety section refuses the sealed span, and
// the narrative surfaces the composable strategies + a novel authored query.
func TestDemoRunsAndUpholdsInvariants(t *testing.T) {
	var buf bytes.Buffer
	if err := run(&buf); err != nil {
		t.Fatalf("demo run failed (an invariant likely broke): %v", err)
	}
	got := buf.String()
	for _, want := range []string{
		"build SQL, not a specific query",
		"sealed_by_trust_gate",             // the poison was refused
		"a NOVEL query the agent authored", // extensibility
		"proposed-not-applied",             // fail-closed effect default narrative
		"consolidated 4 source(s)",         // compact produced a derived disposition
	} {
		if !strings.Contains(got, want) {
			t.Errorf("demo output missing %q", want)
		}
	}
	// The poison must never appear in a rendered line.
	if strings.Contains(got, "ignore previous instructions") {
		t.Fatal("poison text leaked into the demo output")
	}
}
