package main

import "testing"

// The roster/report/fold/render logic — and its full test suite — lives in
// internal/fleet now. What remains CLI-local is the selector and the usage glue, so
// this file tests only that. selectBoxes is the one piece of filtering logic the
// commands share; the rest of main.go is flag parsing over the fleet package.

func TestSelectBoxes(t *testing.T) {
	ro := Roster{Boxes: []Box{
		{ID: "a", Class: "h100x8", Group: "lab-1"},
		{ID: "b", Class: "a100x8", Group: "lab-1"},
		{ID: "c", Class: "h100x8", Group: "lab-2"},
	}}
	if got := selectBoxes(ro, "", "").Boxes; len(got) != 3 {
		t.Fatalf("no filter should pass all, got %d", len(got))
	}
	if got := selectBoxes(ro, "lab-1", "").Boxes; len(got) != 2 || got[0].ID != "a" || got[1].ID != "b" {
		t.Fatalf("group filter wrong: %+v", got)
	}
	if got := selectBoxes(ro, "", "h100x8").Boxes; len(got) != 2 {
		t.Fatalf("class filter wrong: %+v", got)
	}
	if got := selectBoxes(ro, "lab-2", "h100x8").Boxes; len(got) != 1 || got[0].ID != "c" {
		t.Fatalf("group+class filter wrong: %+v", got)
	}
	if got := selectBoxes(ro, "nope", "").Boxes; len(got) != 0 {
		t.Fatalf("no-match filter should be empty, got %+v", got)
	}
}

// TestFirstNonEmpty pins the tiny CLI helper that prints a roster's effective schema.
func TestFirstNonEmpty(t *testing.T) {
	if got := firstNonEmpty("", "fallback"); got != "fallback" {
		t.Fatalf("firstNonEmpty(\"\", x) = %q, want fallback", got)
	}
	if got := firstNonEmpty("set", "fallback"); got != "set" {
		t.Fatalf("firstNonEmpty(set, x) = %q, want set", got)
	}
}
