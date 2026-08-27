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
