package workflowoutcomestudy

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestWIP_PR_CohortEvaluation(t *testing.T) {
	data := DefaultCohortDataset()

	// 1. Matched issue count per arm >= 10
	if len(data.DetachedWorker) < 10 {
		t.Fatalf("expected >= 10 detached worker issues, got %d", len(data.DetachedWorker))
	}
	if len(data.PRLane) < 10 {
		t.Fatalf("expected >= 10 PR lane issues, got %d", len(data.PRLane))
	}
	if len(data.DetachedWorker) != len(data.PRLane) {
		t.Fatalf("mismatched arm counts: %d vs %d", len(data.DetachedWorker), len(data.PRLane))
	}

	report1 := EvaluatePRIsolationCohort(data)
	if report1 == nil {
		t.Fatal("expected non-nil cohort report")
	}

	if report1.Schema != PRCohortSchema {
		t.Errorf("expected schema %q, got %q", PRCohortSchema, report1.Schema)
	}

	if report1.CohortSize < 10 {
		t.Errorf("report CohortSize < 10: got %d", report1.CohortSize)
	}
	if report1.Arms.DetachedWorker.IssueCount < 10 {
		t.Errorf("report DetachedWorker.IssueCount < 10: got %d", report1.Arms.DetachedWorker.IssueCount)
	}
	if report1.Arms.PRLane.IssueCount < 10 {
		t.Errorf("report PRLane.IssueCount < 10: got %d", report1.Arms.PRLane.IssueCount)
	}

	// 2. Issue classes >= 2
	if len(report1.IssueClasses) < 2 {
		t.Errorf("expected >= 2 issue classes, got %d: %v", len(report1.IssueClasses), report1.IssueClasses)
	}

	// 3. WIP safety metrics: time-to-protection, collision_rework_rate, abandonment_recovery
	wipSafetyMetrics := []string{
		"time-to-protection",
		"collision_rework_rate",
		"abandonment_recovery",
	}

	// 4. Workflow outcome metrics: review_latency, green_to_land_latency, operator_touches
	workflowOutcomeMetrics := []string{
		"review_latency",
		"green_to_land_latency",
		"operator_touches",
	}

	metricsMap := make(map[string]CohortMetric)
	for _, m := range report1.Metrics {
		metricsMap[m.Name] = m
	}

	for _, name := range wipSafetyMetrics {
		m, ok := metricsMap[name]
		if !ok {
			t.Errorf("missing WIP safety metric: %q", name)
			continue
		}
		if m.Unit == "" {
			t.Errorf("metric %q has empty unit", name)
		}
		if m.DetachedWorkerValue <= 0 && name != "collision_rework_rate" {
			t.Errorf("metric %q has non-positive DetachedWorkerValue: %f", name, m.DetachedWorkerValue)
		}
		if m.PRLaneValue <= 0 && name != "collision_rework_rate" {
			t.Errorf("metric %q has non-positive PRLaneValue: %f", name, m.PRLaneValue)
		}
	}

	for _, name := range workflowOutcomeMetrics {
		m, ok := metricsMap[name]
		if !ok {
			t.Errorf("missing workflow outcome metric: %q", name)
			continue
		}
		if m.Unit == "" {
			t.Errorf("metric %q has empty unit", name)
		}
		if m.DetachedWorkerValue <= 0 {
			t.Errorf("metric %q has non-positive DetachedWorkerValue: %f", name, m.DetachedWorkerValue)
		}
		if m.PRLaneValue <= 0 {
			t.Errorf("metric %q has non-positive PRLaneValue: %f", name, m.PRLaneValue)
		}
	}

	// 5. Deterministic verdict
	if report1.Decision != CohortDecisionRejectPRLane {
		t.Errorf("expected decision %q, got %q", CohortDecisionRejectPRLane, report1.Decision)
	}
	if report1.Rationale == "" {
		t.Error("expected non-empty rationale")
	}

	report2 := EvaluatePRIsolationCohort(data)
	if !reflect.DeepEqual(report1, report2) {
		t.Errorf("non-deterministic evaluation: report1 != report2")
	}

	// 6. Subtests for decision boundaries
	t.Run("InsufficientCohortSizeYieldsHold", func(t *testing.T) {
		small := ArmData{
			DetachedWorker: data.DetachedWorker[:5],
			PRLane:         data.PRLane[:5],
		}
		r := EvaluatePRIsolationCohort(small)
		if r.Decision != CohortDecisionHoldDefault {
			t.Errorf("expected HOLD_DEFAULT for small cohort, got %s", r.Decision)
		}
	})

	t.Run("InsufficientClassesYieldsHold", func(t *testing.T) {
		singleClassDW := make([]IssueRecord, 10)
		singleClassPR := make([]IssueRecord, 10)
		for i := 0; i < 10; i++ {
			singleClassDW[i] = data.DetachedWorker[i]
			singleClassDW[i].Class = "bugfix"
			singleClassPR[i] = data.PRLane[i]
			singleClassPR[i].Class = "bugfix"
		}
		r := EvaluatePRIsolationCohort(ArmData{
			DetachedWorker: singleClassDW,
			PRLane:         singleClassPR,
		})
		if r.Decision != CohortDecisionHoldDefault {
			t.Errorf("expected HOLD_DEFAULT for single class, got %s", r.Decision)
		}
	})

	t.Run("PromoteNarrowClassWhenSuperiorDefectReduction", func(t *testing.T) {
		// If PR lane has high defect reduction that offsets latency
		promDW := make([]IssueRecord, 10)
		promPR := make([]IssueRecord, 10)
		for i := 0; i < 10; i++ {
			promDW[i] = IssueRecord{
				IssueID:             data.DetachedWorker[i].IssueID,
				Class:               data.DetachedWorker[i].Class,
				TimeToProtection:    15.0,
				CollisionReworkRate: 0.02,
				ReviewLatency:       60.0,
				GreenToLandLatency:  45.0,
				OperatorTouches:     1.0,
				AbandonmentRecovery: 0.95,
				DefectRate:          0.15, // high defect rate on trunk
			}
			promPR[i] = IssueRecord{
				IssueID:             data.PRLane[i].IssueID,
				Class:               data.PRLane[i].Class,
				TimeToProtection:    25.0,
				CollisionReworkRate: 0.03,
				ReviewLatency:       90.0,
				GreenToLandLatency:  80.0,
				OperatorTouches:     1.2,
				AbandonmentRecovery: 0.90,
				DefectRate:          0.02, // 13% defect reduction offsets minor latency
			}
		}
		r := EvaluatePRIsolationCohort(ArmData{
			DetachedWorker: promDW,
			PRLane:         promPR,
		})
		if r.Decision != CohortDecisionPromoteNarrowClass {
			t.Errorf("expected PROMOTE_NARROW_CLASS with substantial defect reduction, got %s", r.Decision)
		}
	})

	t.Run("ArtifactIntegrity", func(t *testing.T) {
		artifactPath := filepath.Join("..", "..", "docs", "benchmarks", "workflow-outcome-study", "pr_isolation_lane_decision.json")
		raw, err := os.ReadFile(artifactPath)
		if err != nil {
			t.Skipf("skipping artifact check: %v", err)
			return
		}
		var artifact PRCohortReport
		if err := json.Unmarshal(raw, &artifact); err != nil {
			t.Fatalf("failed to unmarshal artifact: %v", err)
		}
		if artifact.Schema != PRCohortSchema {
			t.Errorf("artifact schema %q != %q", artifact.Schema, PRCohortSchema)
		}
		if artifact.Decision != CohortDecisionRejectPRLane {
			t.Errorf("artifact decision %q != %q", artifact.Decision, CohortDecisionRejectPRLane)
		}
		if artifact.CohortSize < 10 {
			t.Errorf("artifact cohort size < 10: %d", artifact.CohortSize)
		}
		if len(artifact.IssueClasses) < 2 {
			t.Errorf("artifact issue classes < 2: %v", artifact.IssueClasses)
		}
		if len(artifact.Metrics) < 6 {
			t.Errorf("artifact metrics < 6: %d", len(artifact.Metrics))
		}
		expected := EvaluatePRIsolationCohort(DefaultCohortDataset())
		if !reflect.DeepEqual(&artifact, expected) {
			t.Errorf("artifact does not match evaluated DefaultCohortDataset")
		}
	})
}
