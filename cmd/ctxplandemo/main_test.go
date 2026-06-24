package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// TestRunSelfcheckPasses is the headline test: the demo's self-check (every invariant the
// command asserts) must pass at the default budget. run() returns nil only when the O(1)
// view stayed under budget, kept the sealed poison out, proved faithfulness, beat the
// optimality/dedup/coverage/learning/index/scaling assertions — so a green run() is a
// behavior witness for the whole demo.
func TestRunSelfcheckPasses(t *testing.T) {
	if err := run(64); err != nil {
		t.Fatalf("run(64) self-check failed: %v", err)
	}
}

// TestNewDemoStoreShape pins the synthetic history the demo plans over: seven spans, with
// span:6 the SEALED poison result the trust gate must quarantine.
func TestNewDemoStoreShape(t *testing.T) {
	store := newDemoStore()
	spans, err := store.Spans(context.Background())
	if err != nil {
		t.Fatalf("Spans: %v", err)
	}
	if len(spans) != 7 {
		t.Fatalf("expected 7 demo spans, got %d", len(spans))
	}
	var poison *ctxplan.Span
	for i := range spans {
		if spans[i].ID == "span:6" {
			poison = &spans[i]
		}
	}
	if poison == nil {
		t.Fatal("expected span:6 (the sealed poison) in the demo store")
	}
}

// TestPlanAndShowViewKeepsPoisonOut is a focused behavior test on the extracted helper:
// the materialized view stays under budget and never renders the sealed poison span.
func TestPlanAndShowViewKeepsPoisonOut(t *testing.T) {
	const budget = 64
	view, err := planAndShowView(context.Background(), newDemoStore(), demoForecast(), budget)
	if err != nil {
		t.Fatalf("planAndShowView: %v", err)
	}
	if view.RenderedTokens() > budget {
		t.Fatalf("resident %d exceeded budget %d", view.RenderedTokens(), budget)
	}
	for _, r := range view.Rendered {
		if r.ID == "span:6" {
			t.Fatal("the sealed poison span:6 was rendered into the view")
		}
	}
}

// TestPinSet checks the pin-membership helper round-trips its input.
func TestPinSet(t *testing.T) {
	got := pinSet([]string{"span:0", "span:1"})
	if !got["span:0"] || !got["span:1"] {
		t.Fatalf("pins missing from set: %v", got)
	}
	if got["span:2"] {
		t.Fatal("pinSet reported a pin it was never given")
	}
}

// TestWordCounter is the deterministic stand-in tokenizer: whitespace word count.
func TestWordCounter(t *testing.T) {
	if n := (wordCounter{}).Count("mint roll revoke"); n != 3 {
		t.Fatalf("Count(\"mint roll revoke\") = %d, want 3", n)
	}
	if n := (wordCounter{}).Count(""); n != 0 {
		t.Fatalf("Count(\"\") = %d, want 0", n)
	}
}
