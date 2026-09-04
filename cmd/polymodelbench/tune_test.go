package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestTune(t *testing.T) {
	TestMTPSweep(t)
}

func TestMTPSweep(t *testing.T) {
	report := RunMTPSweep(true)
	if len(report.Points) == 0 {
		t.Fatal("expected non-empty Points in MTPSweepReport")
	}

	// 4 values of K * 2 thresholds * 3 categories = 24 points
	if len(report.Points) != 24 {
		t.Errorf("expected 24 Points in MTPSweepReport, got %d", len(report.Points))
	}

	kCovered := make(map[int]bool)
	for _, p := range report.Points {
		kCovered[p.K] = true
		if p.TotalAccepted > p.TotalProposed {
			t.Errorf("point K=%d Category=%s: TotalAccepted (%d) > TotalProposed (%d)", p.K, p.Category, p.TotalAccepted, p.TotalProposed)
		}
		if p.AcceptedRate <= 0 || p.AcceptedRate > 1.0 {
			t.Errorf("point K=%d Category=%s: AcceptedRate=%.4f, want (0, 1.0]", p.K, p.Category, p.AcceptedRate)
		}
		if p.TokensPerSec <= 0 {
			t.Errorf("point K=%d Category=%s: TokensPerSec=%.2f, want > 0", p.K, p.Category, p.TokensPerSec)
		}
		if p.Speedup <= 0 {
			t.Errorf("point K=%d Category=%s: Speedup=%.3f, want > 0", p.K, p.Category, p.Speedup)
		}
	}
	for _, k := range []int{1, 2, 3, 4} {
		if !kCovered[k] {
			t.Errorf("expected K=%d to be covered in Points", k)
		}
	}

	if report.OptimalK < 1 || report.OptimalK > 4 {
		t.Errorf("OptimalK=%d, want within [1, 4]", report.OptimalK)
	}
	if report.OptimalSpeedup < 1.0 {
		t.Errorf("OptimalSpeedup=%.2f, want >= 1.0", report.OptimalSpeedup)
	}

	if !strings.Contains(report.RecommendedArgs, "--draft-depth") {
		t.Errorf("RecommendedArgs=%q, want containing '--draft-depth'", report.RecommendedArgs)
	}

	// Verify JSON serialization round-trip
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("json.Marshal(MTPSweepReport) failed: %v", err)
	}
	var unmarshaled MTPSweepReport
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("json.Unmarshal(MTPSweepReport) failed: %v", err)
	}
	if len(unmarshaled.Points) != len(report.Points) || unmarshaled.OptimalK != report.OptimalK {
		t.Errorf("unmarshaled report mismatch: got K=%d, want %d", unmarshaled.OptimalK, report.OptimalK)
	}

	md := RenderMTPSweepMarkdown(report)
	expectedColumns := []string{"| K |", "| Threshold |", "| Category |", "| Acceptance Rate |", "| Tokens/sec |", "| Speedup |"}
	for _, col := range expectedColumns {
		if !strings.Contains(md, col) {
			t.Errorf("markdown missing expected column header %q", col)
		}
	}
	if !strings.Contains(md, "| K | Threshold | Category | Acceptance Rate | Tokens/sec | Speedup |") {
		t.Errorf("markdown missing full table header row")
	}
	if !strings.Contains(md, report.RecommendedArgs) {
		t.Errorf("markdown missing recommended args %q", report.RecommendedArgs)
	}

	// Verify RunMTPSweep with quiet=false
	reportNotQuiet := RunMTPSweep(false)
	if len(reportNotQuiet.Points) != len(report.Points) {
		t.Errorf("RunMTPSweep(false) returned %d points, want %d", len(reportNotQuiet.Points), len(report.Points))
	}
}
