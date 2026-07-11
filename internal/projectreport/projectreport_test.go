package projectreport

import (
	"strings"
	"testing"
)

func healthy() []Item {
	return []Item{
		{Issue: 4030, Status: "In progress", Generation: "now", Priority: "P1"},
		{Issue: 4017, Status: "Todo", Generation: "now", Priority: "P2"},
		{Issue: 3906, Status: "Backlog", Generation: "next", Priority: "P2"},
		{Issue: 3908, Status: "Backlog", Generation: "next", Priority: "P3"},
	}
}

func TestFoldHealthyIsOK(t *testing.T) {
	r := Fold(healthy(), FoldOpts{})
	if !r.OK || r.Verdict != "OK" {
		t.Fatalf("healthy board: got OK=%v verdict=%q, want OK verdict", r.OK, r.Verdict)
	}
	if r.Total != 4 {
		t.Fatalf("total = %d, want 4", r.Total)
	}
	if len(r.Unclassified) != 0 {
		t.Fatalf("unclassified = %v, want none", r.Unclassified)
	}
	if r.ByGeneration["now"] != 2 || r.ByGeneration["next"] != 2 {
		t.Fatalf("by_generation = %v, want now:2 next:2", r.ByGeneration)
	}
	if r.ByStatus["Backlog"] != 2 {
		t.Fatalf("by_status Backlog = %d, want 2", r.ByStatus["Backlog"])
	}
}

func TestFoldDriftFlagsUnclassifiedAsAction(t *testing.T) {
	items := append(healthy(),
		Item{Issue: 9999}, // just-filed, no fields
		Item{Issue: 8888, Status: "Todo", Priority: "P2"}, // no gen/* label yet
	)
	r := Fold(items, FoldOpts{})
	if r.OK || r.Verdict != "ACTION" {
		t.Fatalf("drifted board: got OK=%v verdict=%q, want ACTION", r.OK, r.Verdict)
	}
	if len(r.Unclassified) != 2 {
		t.Fatalf("unclassified = %v, want [8888 9999]", r.Unclassified)
	}
	// sorted ascending
	if r.Unclassified[0] != 8888 || r.Unclassified[1] != 9999 {
		t.Fatalf("unclassified not sorted ascending: %v", r.Unclassified)
	}
	if !strings.Contains(r.Finding, "2/6") {
		t.Fatalf("finding = %q, want it to mention 2/6", r.Finding)
	}
}

func TestFoldEmptyBoardIsAction(t *testing.T) {
	r := Fold(nil, FoldOpts{})
	if r.OK || r.Verdict != "ACTION" {
		t.Fatalf("empty board: got OK=%v verdict=%q, want ACTION", r.OK, r.Verdict)
	}
	if !strings.Contains(r.Finding, "empty") {
		t.Fatalf("finding = %q, want 'empty'", r.Finding)
	}
}

func TestUnmeasuredIsAdvisoryButVisible(t *testing.T) {
	r := Unmeasured("", FoldOpts{})
	if !r.OK {
		t.Fatalf("unmeasured should be advisory-OK (not a hard failure), got OK=false")
	}
	if r.Measured {
		t.Fatalf("unmeasured.Measured should be false")
	}
	if r.Verdict != "UNMEASURED" {
		t.Fatalf("verdict = %q, want UNMEASURED", r.Verdict)
	}
}

func TestRenderMeasuredAndUnmeasured(t *testing.T) {
	got := Render(Fold(healthy(), FoldOpts{Commit: "abcdef123456789", Date: "2026-07-11"}))
	for _, want := range []string{"OK", "total", "status", "horizon", "->"} {
		if !strings.Contains(got, want) {
			t.Fatalf("render missing %q in:\n%s", want, got)
		}
	}
	un := Render(Unmeasured("", FoldOpts{}))
	if !strings.Contains(un, "unmeasured") {
		t.Fatalf("unmeasured render missing 'unmeasured':\n%s", un)
	}
}
