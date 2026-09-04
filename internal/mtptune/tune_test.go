package mtptune

import (
	"encoding/json"
	"testing"
)

func TestSweepGridRanges(t *testing.T) {
	cfg := DefaultSweepConfig()
	cfg.KMin = 1
	cfg.KMax = 8
	cfg.PMin = 0.0
	cfg.PMax = 1.0
	cfg.PStep = 0.2 // 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 -> 6 steps

	report, err := RunSweep(cfg)
	if err != nil {
		t.Fatalf("RunSweep failed: %v", err)
	}

	// 8 K values * 6 P values * 3 tasks = 144 measurement points
	expectedPoints := 8 * 6 * 3
	if len(report.Points) != expectedPoints {
		t.Fatalf("expected %d measurement points, got %d", expectedPoints, len(report.Points))
	}

	if len(report.ParetoFront) == 0 {
		t.Fatal("Pareto front should not be empty")
	}

	if report.OptimalProfile.K < 1 || report.OptimalProfile.K > 8 {
		t.Fatalf("optimal profile K=%d out of bounds [1, 8]", report.OptimalProfile.K)
	}
}

func TestTaskCategoryAcceptanceVariation(t *testing.T) {
	cfg := DefaultSweepConfig()
	k := 4
	p := 0.4

	ptJSON := SimulateMTPStep(TaskJSON, k, p, cfg)
	ptCode := SimulateMTPStep(TaskCode, k, p, cfg)
	ptMath := SimulateMTPStep(TaskMath, k, p, cfg)

	// JSON should have highest predictability, then Code, then Math
	if ptJSON.AcceptanceRate <= ptCode.AcceptanceRate {
		t.Fatalf("expected JSON acceptance (%.3f) > Code acceptance (%.3f)",
			ptJSON.AcceptanceRate, ptCode.AcceptanceRate)
	}
	if ptCode.AcceptanceRate <= ptMath.AcceptanceRate {
		t.Fatalf("expected Code acceptance (%.3f) > Math acceptance (%.3f)",
			ptCode.AcceptanceRate, ptMath.AcceptanceRate)
	}
}

func TestBusSaturationIncreasesWithK(t *testing.T) {
	cfg := DefaultSweepConfig()
	p := 0.4

	prevBusSat := 0.0
	for k := 1; k <= 8; k++ {
		pt := SimulateMTPStep(TaskCode, k, p, cfg)
		if pt.BusSaturation < prevBusSat {
			t.Fatalf("bus saturation did not increase monotonically with K: K=%d (%.3f) < K=%d (%.3f)",
				k, pt.BusSaturation, k-1, prevBusSat)
		}
		prevBusSat = pt.BusSaturation
	}

	// At K=8, bus saturation should be high
	pt8 := SimulateMTPStep(TaskCode, 8, p, cfg)
	if pt8.BusSaturation < 0.80 {
		t.Fatalf("expected high bus saturation at K=8, got %.3f", pt8.BusSaturation)
	}
}

func TestParetoFrontIdentification(t *testing.T) {
	pts := []ParetoPoint{
		{K: 1, P: 0.2, AvgTPS: 20.0, AvgAcceptRate: 0.85, AvgBusSat: 0.40},
		{K: 2, P: 0.2, AvgTPS: 28.0, AvgAcceptRate: 0.80, AvgBusSat: 0.50}, // Dominates next point
		{K: 3, P: 0.2, AvgTPS: 25.0, AvgAcceptRate: 0.70, AvgBusSat: 0.60}, // Dominated by (K=2)
		{K: 4, P: 0.4, AvgTPS: 35.0, AvgAcceptRate: 0.82, AvgBusSat: 0.65}, // High TPS, good acceptance
	}

	front := identifyParetoPoints(pts)

	// K=3 should be dominated by K=2 (K=2 has higher TPS, higher accept rate, lower bus sat)
	for _, p := range front {
		if p.K == 3 {
			t.Fatal("point K=3 should have been filtered from Pareto front")
		}
	}
}

func TestOptimalProfileSelection(t *testing.T) {
	cfg := DefaultSweepConfig()
	report, err := RunSweep(cfg)
	if err != nil {
		t.Fatalf("RunSweep failed: %v", err)
	}

	opt := report.OptimalProfile
	if !opt.IsOptimal {
		t.Fatal("expected IsOptimal flag to be true on optimal profile")
	}

	// Single-stream sweet spot on 200 GB/s bus should be around K=3..5
	if opt.K < 2 || opt.K > 6 {
		t.Fatalf("optimal K=%d outside expected sweet spot [2, 6]", opt.K)
	}
	if opt.AvgTPS < 20.0 {
		t.Fatalf("optimal profile AvgTPS = %.2f, expected >= 20 tok/s", opt.AvgTPS)
	}

	// Per-task optimal checks
	for _, task := range cfg.Tasks {
		taskOpt, ok := report.OptimalPerTask[task]
		if !ok {
			t.Fatalf("missing optimal profile for task %s", task)
		}
		if taskOpt.SimulatedTPS <= 0 {
			t.Fatalf("task %s optimal TPS <= 0", task)
		}
	}
}

func TestFormatReport(t *testing.T) {
	cfg := DefaultSweepConfig()
	cfg.KMax = 3
	cfg.PStep = 0.5

	report, err := RunSweep(cfg)
	if err != nil {
		t.Fatalf("RunSweep failed: %v", err)
	}

	table := FormatReportTable(report)
	if len(table) == 0 {
		t.Fatal("empty formatted table")
	}

	jsonBytes, err := FormatReportJSON(report)
	if err != nil {
		t.Fatalf("FormatReportJSON failed: %v", err)
	}

	var parsed SweepReport
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		t.Fatalf("failed to unmarshal generated JSON: %v", err)
	}
	if parsed.OptimalProfile.K != report.OptimalProfile.K {
		t.Fatalf("JSON roundtrip mismatch: K=%d, want %d", parsed.OptimalProfile.K, report.OptimalProfile.K)
	}
}

func TestValidateSweepConfig(t *testing.T) {
	valid := DefaultSweepConfig()
	if err := valid.Validate(); err != nil {
		t.Fatalf("expected DefaultSweepConfig to be valid, got: %v", err)
	}

	invalidK := valid
	invalidK.KMin = 8
	invalidK.KMax = 2
	if err := invalidK.Validate(); err == nil {
		t.Fatal("expected error for KMin > KMax")
	}

	invalidP := valid
	invalidP.PMin = 0.8
	invalidP.PMax = 0.2
	if err := invalidP.Validate(); err == nil {
		t.Fatal("expected error for PMin > PMax")
	}

	invalidTasks := valid
	invalidTasks.Tasks = nil
	if err := invalidTasks.Validate(); err == nil {
		t.Fatal("expected error for empty tasks")
	}

	invalidBus := valid
	invalidBus.BusBandwidthGBs = -10
	if err := invalidBus.Validate(); err == nil {
		t.Fatal("expected error for negative bus bandwidth")
	}
}

func TestSweepHighBusSaturationFallback(t *testing.T) {
	cfg := DefaultSweepConfig()
	// Constrain bandwidth to force high bus saturation (> 0.85) on all points
	cfg.BusBandwidthGBs = 10.0
	cfg.KMax = 4

	report, err := RunSweep(cfg)
	if err != nil {
		t.Fatalf("RunSweep failed with constrained bandwidth: %v", err)
	}

	if report.OptimalProfile.K < 1 {
		t.Fatalf("expected fallback optimal profile K >= 1, got %d", report.OptimalProfile.K)
	}
	for _, task := range cfg.Tasks {
		taskOpt, ok := report.OptimalPerTask[task]
		if !ok || taskOpt.K < 1 {
			t.Fatalf("expected fallback optimal per task %s K >= 1", task)
		}
	}
}

func TestUnknownTaskCategoryFallback(t *testing.T) {
	cfg := DefaultSweepConfig()
	customTask := TaskCategory("CustomAgentic")
	pt := SimulateMTPStep(customTask, 2, 0.4, cfg)
	if pt.AcceptanceRate <= 0 {
		t.Fatalf("expected positive acceptance rate for unknown task, got %f", pt.AcceptanceRate)
	}
	if pt.SimulatedTPS <= 0 {
		t.Fatalf("expected positive TPS for unknown task, got %f", pt.SimulatedTPS)
	}
}
