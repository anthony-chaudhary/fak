package taskmgr

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestQualitySLOTaskSnapshotShowsPassingEvidence(t *testing.T) {
	maxStalls := 0
	m := NewManager()
	task, err := m.StartTask(TaskSpec{
		TaskID: "task_slo_pass",
		QualitySLO: &QualitySLO{
			OutputShape:          &OutputShapeSLO{MaxRepeat: 0.9},
			MaxStallCount:        &maxStalls,
			RequiredWitnessState: VerifiedDone,
		},
	})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	rec, err := task.BeatWithEvidence([]byte(coherentOutput))
	if err != nil {
		t.Fatalf("beat with evidence: %v", err)
	}
	if rec.VerifiedState != VerifiedDone {
		t.Fatalf("witness = %q, want verified_done", rec.VerifiedState)
	}

	got := m.Snapshot().Tasks[0]
	if got.QualitySLO == nil || got.QualitySLO.OutputShape == nil {
		t.Fatalf("snapshot missing quality SLO: %+v", got.QualitySLO)
	}
	if got.QualitySLOStatus == nil || !got.QualitySLOStatus.Passed {
		t.Fatalf("quality status = %+v, want passed", got.QualitySLOStatus)
	}
	if got.QualitySLOStatus.WitnessState != VerifiedDone || got.QualitySLOStatus.StallCount != 0 {
		t.Fatalf("quality status evidence = %+v, want verified_done/no stalls", got.QualitySLOStatus)
	}
	if err := ValidateSnapshot(m.Snapshot()); err != nil {
		t.Fatalf("quality SLO snapshot failed validation: %v", err)
	}
	if b, _ := json.Marshal(got); !strings.Contains(string(b), "quality_slo_status") {
		t.Fatalf("snapshot JSON missing quality_slo_status: %s", b)
	}
}

func TestQualitySLOTaskSnapshotFailsShapeThreshold(t *testing.T) {
	m := NewManager()
	task, err := m.StartTask(TaskSpec{
		TaskID: "task_slo_shape_fail",
		QualitySLO: &QualitySLO{
			OutputShape:          &OutputShapeSLO{MaxChars: 16},
			RequiredWitnessState: VerifiedDone,
		},
	})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	rec, err := task.BeatWithEvidence([]byte(coherentOutput))
	if err != nil {
		t.Fatalf("beat with evidence: %v", err)
	}
	if rec.VerifiedState != VerifiedRefused {
		t.Fatalf("witness = %q, want verified_refused from tight max chars", rec.VerifiedState)
	}

	got := m.Snapshot().Tasks[0]
	if got.QualitySLOStatus == nil || got.QualitySLOStatus.Passed {
		t.Fatalf("quality status = %+v, want failed", got.QualitySLOStatus)
	}
	reasons := strings.Join(got.QualitySLOStatus.Reasons, "; ")
	if !strings.Contains(reasons, "output shape witness") || !strings.Contains(reasons, "required verified_done") {
		t.Fatalf("quality reasons = %q, want shape and required-witness failures", reasons)
	}
	if got.VerifiedProgressing() {
		t.Fatalf("shape SLO refusal must flip VerifiedProgressing() false")
	}
}

func TestQualitySLOStepSnapshotFailsAndParentSeesRefusal(t *testing.T) {
	m := NewManager()
	task, err := m.StartTask(TaskSpec{TaskID: "task_slo_step"})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	step, err := task.StartStep(StepSpec{
		StepID:     "step_slo_shape",
		QualitySLO: &QualitySLO{OutputShape: &OutputShapeSLO{MaxChars: 16}},
	})
	if err != nil {
		t.Fatalf("start step: %v", err)
	}
	if _, err := step.BeatWithEvidence([]byte(coherentOutput)); err != nil {
		t.Fatalf("step beat with evidence: %v", err)
	}

	got := m.Snapshot().Tasks[0]
	stepSnap := got.Steps[0]
	if stepSnap.QualitySLO == nil || stepSnap.QualitySLOStatus == nil {
		t.Fatalf("step quality SLO/status missing: %+v / %+v", stepSnap.QualitySLO, stepSnap.QualitySLOStatus)
	}
	if stepSnap.QualitySLOStatus.Passed {
		t.Fatalf("step quality status = %+v, want failed", stepSnap.QualitySLOStatus)
	}
	if stepSnap.VerifiedProgressing() {
		t.Fatalf("step shape refusal must flip step VerifiedProgressing() false")
	}
	if got.VerifiedProgressing() {
		t.Fatalf("step shape refusal must flip parent task VerifiedProgressing() false")
	}
}

func TestQualitySLOSnapshotFailsStallCount(t *testing.T) {
	base := time.Unix(1700000000, 0)
	now := base
	maxStalls := 0
	m := NewManager(
		WithClock(func() time.Time { return now }),
		WithLivenessTimeout(1*time.Second),
	)
	task, err := m.StartTask(TaskSpec{
		TaskID:     "task_slo_stall",
		QualitySLO: &QualitySLO{MaxStallCount: &maxStalls},
	})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := task.Beat(); err != nil {
		t.Fatalf("beat: %v", err)
	}
	now = now.Add(2 * time.Second)

	got := m.Snapshot().Tasks[0]
	if got.LivenessClass != LivenessStalled {
		t.Fatalf("liveness = %s, want stalled", got.LivenessClass)
	}
	if got.QualitySLOStatus == nil || got.QualitySLOStatus.Passed {
		t.Fatalf("quality status = %+v, want failed", got.QualitySLOStatus)
	}
	if got.QualitySLOStatus.StallCount != 1 {
		t.Fatalf("stall count = %d, want 1", got.QualitySLOStatus.StallCount)
	}
	if reasons := strings.Join(got.QualitySLOStatus.Reasons, "; "); !strings.Contains(reasons, "stall count 1 > max 0") {
		t.Fatalf("quality reasons = %q, want stall-count failure", reasons)
	}
}
