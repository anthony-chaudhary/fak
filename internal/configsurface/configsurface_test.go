package configsurface

import (
	"testing"
)

func TestCurrentConfigSurfaceIsBoundedDefaultedAndDescribed(t *testing.T) {
	report := Audit()
	if err := report.Check(); err != nil {
		t.Fatalf("%v: %+v", err, report.Findings)
	}
	if report.Keys != 32 {
		t.Fatalf("keys=%d, want witnessed vocabulary of 32", report.Keys)
	}
	if report.Postures != 4 {
		t.Fatalf("postures=%d, want 4", report.Postures)
	}
	if report.DefaultCoverage != 1 || report.DescriptionCoverage != 1 {
		t.Fatalf("coverage defaults=%v descriptions=%v", report.DefaultCoverage, report.DescriptionCoverage)
	}
	if report.GuideCoverage <= 0 || report.GuideCoverage >= 1 {
		t.Fatalf("guide coverage=%v, want intentional subset: guide exposes intents, not every knob", report.GuideCoverage)
	}
}

func TestSurfaceBudgetsLeaveRoomWithoutAllowingExplosion(t *testing.T) {
	if MaxKeys < 13 || MaxKeys > 32 {
		t.Fatalf("MaxKeys=%d, want bounded headroom through 32", MaxKeys)
	}
	if MaxPostures < 4 || MaxPostures > 8 {
		t.Fatalf("MaxPostures=%d, want bounded headroom through 8", MaxPostures)
	}
}
