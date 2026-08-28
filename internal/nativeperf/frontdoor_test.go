package nativeperf

import (
	"strings"
	"testing"
	"time"
)

func TestFrontDoorSnapshotSeparatesAcceptedApproximateAndDiagnostic(t *testing.T) {
	s, err := BuildFrontDoorSnapshot(ActiveGraph(), time.Date(2026, 8, 26, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if s.Accepted == nil || s.Approximate == nil || s.Diagnostic == nil {
		t.Fatalf("snapshot lost a class: %+v", s)
	}
	if got := s.Approximate.Ratio; got < .4736 || got > .4738 {
		t.Fatalf("ratio=%v, want 3.3/6.966061", got)
	}
	if s.Diagnostic.Quality != "0/5 exact" || s.Diagnostic.Class != "diagnostic" {
		t.Fatalf("diagnostic=%+v", s.Diagnostic)
	}
}

func TestFrontDoorSnapshotReapsExpiredPresentationOnly(t *testing.T) {
	s, err := BuildFrontDoorSnapshot(ActiveGraph(), time.Date(2026, 9, 2, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if s.Accepted != nil || s.Approximate != nil || s.Diagnostic != nil || len(s.Reaped) != 3 {
		t.Fatalf("expired snapshot=%+v", s)
	}
	block, err := FrontDoorBlock(s, FrontDoorSurfaceIndex)
	if err != nil {
		t.Fatal(err)
	}
	for _, stale := range []string{"2.3-2.9", "6.966061", "~0.2 tok/s"} {
		if strings.Contains(block, stale) {
			t.Fatalf("expired active number %q survived:\n%s", stale, block)
		}
	}
	if !strings.Contains(block, "immutable witnesses remain") {
		t.Fatalf("reap lost retention statement:\n%s", block)
	}
}

func TestFrontDoorBlockRoundTrip(t *testing.T) {
	s, err := BuildFrontDoorSnapshot(ActiveGraph(), time.Date(2026, 8, 26, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	block, err := FrontDoorBlock(s, FrontDoorSurfaceREADME)
	if err != nil {
		t.Fatal(err)
	}
	doc := "before\n" + FrontDoorBegin + "\nstale\n" + FrontDoorEnd + "\nafter\n"
	next, err := SpliceFrontDoorBlock(doc, block)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := ExtractFrontDoorBlock(next)
	if !ok || got != block || !strings.Contains(next, "~47%") {
		t.Fatalf("round trip failed:\n%s", next)
	}
}

func TestFrontDoorEdgeAdversarialInputsFailClosed(t *testing.T) {
	valid := FrontDoorBegin + "\ncurrent\n" + FrontDoorEnd
	tests := []struct {
		name      string
		doc       string
		wantOK    bool
		wantBlock string
	}{
		{name: "empty"},
		{name: "begin only", doc: FrontDoorBegin},
		{name: "end only", doc: FrontDoorEnd},
		{name: "reversed markers", doc: FrontDoorEnd + "\n" + FrontDoorBegin},
		{name: "duplicate begin", doc: FrontDoorBegin + "\n" + valid},
		{name: "duplicate end", doc: valid + "\n" + FrontDoorEnd},
		{name: "oversized surroundings", doc: strings.Repeat("x", 1<<20) + valid + strings.Repeat("y", 1<<20), wantOK: true, wantBlock: valid},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, ok := ExtractFrontDoorBlock(test.doc)
			if ok != test.wantOK || got != test.wantBlock {
				t.Fatalf("ExtractFrontDoorBlock() = (%q, %t), want (%q, %t)", got, ok, test.wantBlock, test.wantOK)
			}
		})
	}

	snapshot, err := BuildFrontDoorSnapshot(ActiveGraph(), time.Date(2026, 8, 26, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	for _, surface := range []string{FrontDoorSurfaceREADME, FrontDoorSurfaceIndex, FrontDoorSurfaceLatest} {
		block, err := FrontDoorBlock(snapshot, surface)
		if err != nil {
			t.Fatalf("FrontDoorBlock(%q): %v", surface, err)
		}
		if strings.Count(block, FrontDoorBegin) != 1 || strings.Count(block, FrontDoorEnd) != 1 {
			t.Fatalf("FrontDoorBlock(%q) emitted ambiguous markers:\n%s", surface, block)
		}
	}
	if _, err := FrontDoorBlock(snapshot, "../../hostile"); err == nil {
		t.Fatal("unknown front-door surface accepted")
	}
	if _, err := SpliceFrontDoorBlock("no markers", valid); err == nil {
		t.Fatal("marker-free document accepted for splicing")
	}

	graphTests := []struct {
		name   string
		mutate func(*Graph)
		want   string
	}{
		{name: "invalid graph", mutate: func(g *Graph) { g.Schema = "hostile" }, want: "invalid native performance graph"},
		{name: "missing baseline pair", mutate: func(g *Graph) { g.Rungs, g.Features = nil, nil }, want: "lacks the witnessed Metal baseline/comparison pair"},
		{name: "mismatched provenance", mutate: func(g *Graph) { g.Comparison.Provenance = "different" }, want: "provenance or observation date differs"},
		{name: "mismatched observation date", mutate: func(g *Graph) { g.Comparison.ObservedOn = "2026-08-24" }, want: "provenance or observation date differs"},
	}
	for _, test := range graphTests {
		t.Run(test.name, func(t *testing.T) {
			graph := ActiveGraph()
			test.mutate(&graph)
			if _, err := BuildFrontDoorSnapshot(graph, time.Date(2026, 8, 26, 0, 0, 0, 0, time.UTC)); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("BuildFrontDoorSnapshot() error = %v, want containing %q", err, test.want)
			}
		})
	}
}
