package armtracking

import (
	"math"
	"path/filepath"
	"testing"
	"time"
)

func TestArmTrackingValidation(t *testing.T) {
	// Empty arm_id
	badRes := ArmResult{
		ArmID:         "",
		Workload:      "test-workload",
		ArmKind:       ArmKindBaseline,
		PrimaryMetric: "tok_s",
		PrimaryValue:  100.0,
		Direction:     HigherIsBetter,
	}
	if err := badRes.Validate(); err == nil {
		t.Errorf("expected error for empty ArmID, got nil")
	}

	// Bad arm kind
	badRes.ArmID = "arm1"
	badRes.ArmKind = "unrecognized"
	if err := badRes.Validate(); err == nil {
		t.Errorf("expected error for unrecognized ArmKind, got nil")
	}

	// Bad direction
	badRes.ArmKind = ArmKindSweep
	badRes.Direction = "sideways"
	if err := badRes.Validate(); err == nil {
		t.Errorf("expected error for unrecognized Direction, got nil")
	}

	// NaN or Inf value
	badRes.Direction = HigherIsBetter
	badRes.PrimaryValue = math.NaN()
	if err := badRes.Validate(); err == nil {
		t.Errorf("expected error for NaN PrimaryValue, got nil")
	}
}

func TestArmTrackingShiftingBaselinesAndAudit(t *testing.T) {
	reg := NewRegistry()

	t0 := time.Date(2026, 8, 1, 12, 0, 0, 0, time.UTC)
	t1 := t0.Add(24 * time.Hour)
	t2 := t1.Add(24 * time.Hour)

	// Step 1: Record initial baseline arm
	baseRun1 := ArmResult{
		ArmID:         "cpu_reference",
		Workload:      "qwen-inference",
		ArmKind:       ArmKindBaseline,
		PrimaryMetric: "throughput_tok_s",
		PrimaryValue:  45.0,
		PrimaryUnit:   "tok/s",
		Direction:     HigherIsBetter,
		Metadata: ArmMetadata{
			Dimension: "target",
			Hardware:  "x86_64_desktop",
			CommitSHA: "a1b2c3d",
			Witness:   "experiments/qwen/cpu_ref_v1.json",
		},
		Timestamp: t0,
	}

	evt1, err := reg.Record(baseRun1, "initial CPU reference baseline")
	if err != nil {
		t.Fatalf("unexpected error recording baseline: %v", err)
	}
	if evt1.Action != ActionBaselineShift {
		t.Errorf("expected ActionBaselineShift, got %s", evt1.Action)
	}

	// Step 2: Baseline shifts! A new optimized CPU baseline run achieves 52.0 tok/s
	baseRun2 := ArmResult{
		ArmID:         "cpu_reference",
		Workload:      "qwen-inference",
		ArmKind:       ArmKindBaseline,
		PrimaryMetric: "throughput_tok_s",
		PrimaryValue:  52.0,
		PrimaryUnit:   "tok/s",
		Direction:     HigherIsBetter,
		Metadata: ArmMetadata{
			Dimension: "target",
			Hardware:  "x86_64_desktop",
			CommitSHA: "e4f5g6h",
			Witness:   "experiments/qwen/cpu_ref_v2.json",
		},
		Timestamp: t1,
	}

	evt2, err := reg.Record(baseRun2, "CPU baseline shift after parallel matmul optimization")
	if err != nil {
		t.Fatalf("unexpected error recording baseline shift: %v", err)
	}
	if evt2.Action != ActionBaselineShift {
		t.Errorf("expected ActionBaselineShift on improved baseline, got %s", evt2.Action)
	}
	if evt2.PriorBestValue == nil || *evt2.PriorBestValue != 45.0 {
		t.Errorf("expected prior best 45.0, got %v", evt2.PriorBestValue)
	}
	if evt2.DeltaFromPrior == nil || *evt2.DeltaFromPrior != 7.0 {
		t.Errorf("expected delta +7.0, got %v", evt2.DeltaFromPrior)
	}

	// Step 3: A third run on baseline regresses to 48.0 tok/s
	baseRun3 := baseRun2
	baseRun3.PrimaryValue = 48.0
	baseRun3.Timestamp = t2
	evt3, err := reg.Record(baseRun3, "re-test under thermal throttling")
	if err != nil {
		t.Fatalf("unexpected error recording regressed run: %v", err)
	}
	if evt3.Action != ActionRecordRun {
		t.Errorf("expected ActionRecordRun for non-improving run, got %s", evt3.Action)
	}

	// Verify LatestBest for baseline remains 52.0 tok/s, while run count is 3
	arm, count, ok := reg.GetArm("qwen-inference", "cpu_reference")
	if !ok {
		t.Fatalf("arm cpu_reference not found")
	}
	if count != 3 {
		t.Errorf("expected run count 3, got %d", count)
	}
	if arm.PrimaryValue != 52.0 {
		t.Errorf("expected LatestBest 52.0, got %f", arm.PrimaryValue)
	}

	// Verify Audit History
	history, err := reg.AuditHistory("qwen-inference", "cpu_reference")
	if err != nil {
		t.Fatalf("unexpected error fetching audit history: %v", err)
	}
	if len(history) != 3 {
		t.Fatalf("expected 3 audit history entries, got %d", len(history))
	}
}

func TestNextBestComparisonHigherIsBetter(t *testing.T) {
	reg := NewRegistry()
	workload := "strix-q4k-gemv"

	// Arm 1: Baseline CPU reference (75 tok/s)
	_, _ = reg.Record(ArmResult{
		ArmID:         "cpu_reference",
		Workload:      workload,
		ArmKind:       ArmKindBaseline,
		PrimaryMetric: "throughput_tok_s",
		PrimaryValue:  75.0,
		PrimaryUnit:   "tok/s",
		Direction:     HigherIsBetter,
	}, "baseline")

	// Arm 2: Ablation arm with discrete Vulkan (250 tok/s)
	_, _ = reg.Record(ArmResult{
		ArmID:         "vulkan_discrete",
		Workload:      workload,
		ArmKind:       ArmKindAblation,
		PrimaryMetric: "throughput_tok_s",
		PrimaryValue:  250.0,
		PrimaryUnit:   "tok/s",
		Direction:     HigherIsBetter,
	}, "ablation arm")

	// Arm 3: Candidate with fused device-local Q4_K (450 tok/s)
	_, _ = reg.Record(ArmResult{
		ArmID:         "vulkan_fused_q4k",
		Workload:      workload,
		ArmKind:       ArmKindCandidate,
		PrimaryMetric: "throughput_tok_s",
		PrimaryValue:  450.0,
		PrimaryUnit:   "tok/s",
		Direction:     HigherIsBetter,
	}, "candidate champion")

	// Test 1: Compare Champion (vulkan_fused_q4k, Rank 1)
	// By default, champion compares against runner-up (#2 vulkan_discrete: 250 tok/s)
	cmpChamp, err := reg.CompareToNextBest(workload, "vulkan_fused_q4k")
	if err != nil {
		t.Fatalf("unexpected error in CompareToNextBest: %v", err)
	}
	if !cmpChamp.IsChampion {
		t.Errorf("expected vulkan_fused_q4k to be champion")
	}
	if cmpChamp.TargetRank != 1 {
		t.Errorf("expected TargetRank 1, got %d", cmpChamp.TargetRank)
	}
	if cmpChamp.NextBestArm.ArmID != "vulkan_discrete" {
		t.Errorf("expected NextBestArm vulkan_discrete, got %s", cmpChamp.NextBestArm.ArmID)
	}
	if cmpChamp.NextBestRank != 2 {
		t.Errorf("expected NextBestRank 2, got %d", cmpChamp.NextBestRank)
	}
	// Speedup vs next-best = 450 / 250 = 1.80x
	if cmpChamp.SpeedupOrLift < 1.79 || cmpChamp.SpeedupOrLift > 1.81 {
		t.Errorf("expected speedup ~1.80x, got %f", cmpChamp.SpeedupOrLift)
	}
	// Percentage delta = +80.0%
	if cmpChamp.PercentageDelta < 79.9 || cmpChamp.PercentageDelta > 80.1 {
		t.Errorf("expected percentage delta ~+80%%, got %f", cmpChamp.PercentageDelta)
	}

	// Test 2: Compare Runner-up (vulkan_discrete, Rank 2)
	// Runner-up compares against the result immediately ahead of it (Rank 1 vulkan_fused_q4k: 450 tok/s)
	cmpRunnerUp, err := reg.CompareToNextBest(workload, "vulkan_discrete")
	if err != nil {
		t.Fatalf("unexpected error comparing runner-up: %v", err)
	}
	if cmpRunnerUp.IsChampion {
		t.Errorf("expected vulkan_discrete not to be champion")
	}
	if cmpRunnerUp.TargetRank != 2 {
		t.Errorf("expected TargetRank 2, got %d", cmpRunnerUp.TargetRank)
	}
	if cmpRunnerUp.NextBestArm.ArmID != "vulkan_fused_q4k" {
		t.Errorf("expected superior arm vulkan_fused_q4k, got %s", cmpRunnerUp.NextBestArm.ArmID)
	}
	if cmpRunnerUp.NextBestRank != 1 {
		t.Errorf("expected NextBestRank 1, got %d", cmpRunnerUp.NextBestRank)
	}
	// Ratio vs next-best = 250 / 450 ≈ 0.555x
	if cmpRunnerUp.SpeedupOrLift < 0.55 || cmpRunnerUp.SpeedupOrLift > 0.56 {
		t.Errorf("expected speedup ~0.556x, got %f", cmpRunnerUp.SpeedupOrLift)
	}
}

func TestNextBestComparisonLowerIsBetter(t *testing.T) {
	reg := NewRegistry()
	workload := "auth-latency"

	// Arm 1: Uncached Baseline (12.0 ms)
	_, _ = reg.Record(ArmResult{
		ArmID:         "baseline-uncached",
		Workload:      workload,
		ArmKind:       ArmKindBaseline,
		PrimaryMetric: "latency_ms",
		PrimaryValue:  12.0,
		PrimaryUnit:   "ms",
		Direction:     LowerIsBetter,
	}, "baseline")

	// Arm 2: Redis Client Cache (3.0 ms)
	_, _ = reg.Record(ArmResult{
		ArmID:         "redis-cache",
		Workload:      workload,
		ArmKind:       ArmKindAblation,
		PrimaryMetric: "latency_ms",
		PrimaryValue:  3.0,
		PrimaryUnit:   "ms",
		Direction:     LowerIsBetter,
	}, "ablation")

	// Arm 3: In-Kernel vDSO (0.6 ms)
	_, _ = reg.Record(ArmResult{
		ArmID:         "vdso-inkernel",
		Workload:      workload,
		ArmKind:       ArmKindCandidate,
		PrimaryMetric: "latency_ms",
		PrimaryValue:  0.6,
		PrimaryUnit:   "ms",
		Direction:     LowerIsBetter,
	}, "champion candidate")

	// Champion is vdso-inkernel (0.6 ms vs next-best redis-cache 3.0 ms)
	cmpChamp, err := reg.CompareToNextBest(workload, "vdso-inkernel")
	if err != nil {
		t.Fatalf("unexpected error comparing champion: %v", err)
	}
	if !cmpChamp.IsChampion {
		t.Errorf("expected vdso-inkernel to be champion")
	}
	if cmpChamp.NextBestArm.ArmID != "redis-cache" {
		t.Errorf("expected next best redis-cache, got %s", cmpChamp.NextBestArm.ArmID)
	}
	// For LowerIsBetter, speedup = 3.0 / 0.6 = 5.0x faster!
	if cmpChamp.SpeedupOrLift < 4.99 || cmpChamp.SpeedupOrLift > 5.01 {
		t.Errorf("expected 5.0x speedup, got %f", cmpChamp.SpeedupOrLift)
	}
	// Percentage delta = (0.6 - 3.0) / 3.0 = -80.0% latency reduction
	if cmpChamp.PercentageDelta < -80.1 || cmpChamp.PercentageDelta > -79.9 {
		t.Errorf("expected -80%% latency reduction, got %f", cmpChamp.PercentageDelta)
	}
}

func TestLeaderboard(t *testing.T) {
	reg := NewRegistry()
	workload := "batch-sweep"

	arms := []struct {
		id  string
		val float64
	}{
		{"b1", 100.0},
		{"b2", 180.0},
		{"b4", 320.0},
		{"b8", 450.0},
	}

	for _, a := range arms {
		_, err := reg.Record(ArmResult{
			ArmID:         a.id,
			Workload:      workload,
			ArmKind:       ArmKindSweep,
			PrimaryMetric: "tok_s",
			PrimaryValue:  a.val,
			PrimaryUnit:   "tok/s",
			Direction:     HigherIsBetter,
		}, "sweep arm")
		if err != nil {
			t.Fatalf("error recording %s: %v", a.id, err)
		}
	}

	lb, err := reg.Leaderboard(workload)
	if err != nil {
		t.Fatalf("unexpected error fetching leaderboard: %v", err)
	}
	if len(lb) != 4 {
		t.Fatalf("expected 4 rows, got %d", len(lb))
	}

	// b8 must be rank 1
	if lb[0].ArmID != "b8" || lb[0].Rank != 1 {
		t.Errorf("expected b8 at rank 1, got %s at rank %d", lb[0].ArmID, lb[0].Rank)
	}
	// b8 next-best should be b4
	if lb[0].NextBestArmID != "b4" {
		t.Errorf("expected b8 next best to be b4, got %s", lb[0].NextBestArmID)
	}

	// b1 must be rank 4
	if lb[3].ArmID != "b1" || lb[3].Rank != 4 {
		t.Errorf("expected b1 at rank 4, got %s at rank %d", lb[3].ArmID, lb[3].Rank)
	}
}

func TestRegistryStorePersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "shifting-baselines.json")

	reg := NewRegistry()
	_, _ = reg.Record(ArmResult{
		ArmID:         "baseline-control",
		Workload:      "cache-hit-test",
		ArmKind:       ArmKindBaseline,
		PrimaryMetric: "hit_rate",
		PrimaryValue:  0.45,
		Direction:     HigherIsBetter,
		Metadata: ArmMetadata{
			Feature: "no_cache",
		},
	}, "initial control")

	if err := SaveRegistry(reg, path); err != nil {
		t.Fatalf("failed to save registry: %v", err)
	}

	loaded, err := LoadRegistry(path)
	if err != nil {
		t.Fatalf("failed to load registry: %v", err)
	}

	arm, count, ok := loaded.GetArm("cache-hit-test", "baseline-control")
	if !ok {
		t.Fatalf("loaded registry missing arm")
	}
	if count != 1 {
		t.Errorf("expected 1 run, got %d", count)
	}
	if arm.PrimaryValue != 0.45 {
		t.Errorf("expected value 0.45, got %f", arm.PrimaryValue)
	}
}

func TestMetricMismatchAndNotFoundErrors(t *testing.T) {
	reg := NewRegistry()
	_, err := reg.Record(ArmResult{
		ArmID:         "arm1",
		Workload:      "workload1",
		ArmKind:       ArmKindBaseline,
		PrimaryMetric: "tok_s",
		PrimaryValue:  10.0,
		Direction:     HigherIsBetter,
	}, "")
	if err != nil {
		t.Fatalf("unexpected record error: %v", err)
	}

	// Metric mismatch error
	_, err = reg.Record(ArmResult{
		ArmID:         "arm2",
		Workload:      "workload1",
		ArmKind:       ArmKindAblation,
		PrimaryMetric: "latency_ms",
		PrimaryValue:  5.0,
		Direction:     HigherIsBetter,
	}, "")
	if err == nil {
		t.Errorf("expected error on metric mismatch, got nil")
	}

	// Direction mismatch error
	_, err = reg.Record(ArmResult{
		ArmID:         "arm2",
		Workload:      "workload1",
		ArmKind:       ArmKindAblation,
		PrimaryMetric: "tok_s",
		PrimaryValue:  5.0,
		Direction:     LowerIsBetter,
	}, "")
	if err == nil {
		t.Errorf("expected error on direction mismatch, got nil")
	}

	// Workload list
	list := reg.WorkloadsList()
	if len(list) != 1 || list[0] != "workload1" {
		t.Errorf("expected [workload1], got %v", list)
	}

	// Single-arm comparison
	singleCmp, err := reg.CompareToNextBest("workload1", "arm1")
	if err != nil {
		t.Fatalf("unexpected error comparing single arm: %v", err)
	}
	if singleCmp.TotalArms != 1 || !singleCmp.IsChampion {
		t.Errorf("expected single arm champion, got %+v", singleCmp)
	}

	// Unknown arm comparison
	if _, err := reg.CompareToNextBest("workload1", "unknown"); err == nil {
		t.Errorf("expected error comparing unknown arm, got nil")
	}

	// Unknown workload comparison
	if _, err := reg.CompareToNextBest("unknown", "arm1"); err == nil {
		t.Errorf("expected error comparing unknown workload, got nil")
	}
}
