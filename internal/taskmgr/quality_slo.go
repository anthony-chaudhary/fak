package taskmgr

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/answershape"
)

// QualitySLO is the origin-declared quality contract for a task or step. It is
// copied into snapshots so readers can see the expectation next to the evidence
// that currently passes or fails it.
type QualitySLO struct {
	OutputShape          *OutputShapeSLO `json:"output_shape,omitempty"`
	MaxStallCount        *int            `json:"max_stall_count,omitempty"`
	RequiredWitnessState VerifiedState   `json:"required_witness_state,omitempty"`
}

// OutputShapeSLO carries the ShapeWitness limits a task or step expects for
// beat-time output. A nil OutputShape means the default ShapeWitness limits apply
// only when a caller explicitly runs a shape witness; a non-nil value makes the
// expectation visible in the spec and snapshot.
type OutputShapeSLO struct {
	MaxRepeat float64 `json:"max_repeat,omitempty"`
	MaxChars  int     `json:"max_chars,omitempty"`
	NGram     int     `json:"ngram,omitempty"`
}

// QualitySLOStatus is the current evidence verdict against a QualitySLO.
type QualitySLOStatus struct {
	Passed       bool          `json:"passed"`
	Reasons      []string      `json:"reasons,omitempty"`
	WitnessState VerifiedState `json:"witness_state,omitempty"`
	StallCount   int           `json:"stall_count"`
}

func (s OutputShapeSLO) limits() answershape.Limits {
	return answershape.Limits{
		MaxRepeat: s.MaxRepeat,
		MaxChars:  s.MaxChars,
		NGram:     s.NGram,
	}
}

func shapeWitnessForSLO(slo *QualitySLO) ShapeWitness {
	if slo == nil || slo.OutputShape == nil {
		return ShapeWitness{}
	}
	return ShapeWitness{Limits: slo.OutputShape.limits()}
}

func (m *Manager) taskShapeWitness(taskID string) (ShapeWitness, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	task, ok := m.tasks[taskID]
	if !ok {
		return ShapeWitness{}, fmt.Errorf("taskmgr: task %q not found", taskID)
	}
	return shapeWitnessForSLO(task.spec.QualitySLO), nil
}

func (m *Manager) stepShapeWitness(taskID, stepID string) (ShapeWitness, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	task, ok := m.tasks[taskID]
	if !ok {
		return ShapeWitness{}, fmt.Errorf("taskmgr: task %q not found", taskID)
	}
	step, ok := task.steps[stepID]
	if !ok {
		return ShapeWitness{}, fmt.Errorf("taskmgr: step %q not found on task %q", stepID, taskID)
	}
	return shapeWitnessForSLO(step.spec.QualitySLO), nil
}

func evaluateTaskQualitySLO(slo *QualitySLO, witness *WitnessRecord, liveness LivenessClass, steps []StepSnapshot) *QualitySLOStatus {
	if slo == nil {
		return nil
	}
	status := newQualitySLOStatus(slo, witness, currentStallCount(liveness))
	for _, step := range steps {
		if step.LivenessClass == LivenessStalled {
			status.StallCount++
		}
	}
	finishQualitySLOStatus(slo, status)
	return status
}

func evaluateStepQualitySLO(slo *QualitySLO, witness *WitnessRecord, liveness LivenessClass) *QualitySLOStatus {
	if slo == nil {
		return nil
	}
	status := newQualitySLOStatus(slo, witness, currentStallCount(liveness))
	finishQualitySLOStatus(slo, status)
	return status
}

func newQualitySLOStatus(slo *QualitySLO, witness *WitnessRecord, stallCount int) *QualitySLOStatus {
	status := &QualitySLOStatus{
		Passed:       true,
		WitnessState: currentWitnessState(witness),
		StallCount:   stallCount,
	}
	if slo.OutputShape != nil && status.WitnessState != VerifiedDone {
		status.Reasons = append(status.Reasons, "output shape witness is "+qualityStateName(status.WitnessState))
	}
	if slo.RequiredWitnessState != VerifiedUnknown && status.WitnessState != slo.RequiredWitnessState {
		status.Reasons = append(status.Reasons, "witness state "+qualityStateName(status.WitnessState)+" != required "+qualityStateName(slo.RequiredWitnessState))
	}
	return status
}

func finishQualitySLOStatus(slo *QualitySLO, status *QualitySLOStatus) {
	if slo.MaxStallCount != nil && status.StallCount > *slo.MaxStallCount {
		status.Reasons = append(status.Reasons, fmt.Sprintf("stall count %d > max %d", status.StallCount, *slo.MaxStallCount))
	}
	status.Passed = len(status.Reasons) == 0
}

func currentWitnessState(w *WitnessRecord) VerifiedState {
	if w == nil {
		return VerifiedUnknown
	}
	return w.VerifiedState
}

func currentStallCount(liveness LivenessClass) int {
	if liveness == LivenessStalled {
		return 1
	}
	return 0
}

func qualityStateName(s VerifiedState) string {
	if s == VerifiedUnknown {
		return "verified_unknown"
	}
	return string(s)
}

func validateQualitySLO(ctx string, slo *QualitySLO) error {
	if slo == nil {
		return nil
	}
	if slo.OutputShape != nil {
		if slo.OutputShape.MaxRepeat < 0 || slo.OutputShape.MaxRepeat > 1 {
			return fmt.Errorf("taskmgr: %s output_shape.max_repeat = %v, want 0..1", ctx, slo.OutputShape.MaxRepeat)
		}
		if slo.OutputShape.MaxChars < 0 {
			return fmt.Errorf("taskmgr: %s output_shape.max_chars is negative: %d", ctx, slo.OutputShape.MaxChars)
		}
		if slo.OutputShape.NGram < 0 {
			return fmt.Errorf("taskmgr: %s output_shape.ngram is negative: %d", ctx, slo.OutputShape.NGram)
		}
	}
	if slo.MaxStallCount != nil && *slo.MaxStallCount < 0 {
		return fmt.Errorf("taskmgr: %s max_stall_count is negative: %d", ctx, *slo.MaxStallCount)
	}
	if err := validateVerifiedState(slo.RequiredWitnessState); err != nil {
		return fmt.Errorf("taskmgr: %s required_witness_state: %w", ctx, err)
	}
	return nil
}

func validateQualitySLOStatus(ctx string, slo *QualitySLO, status *QualitySLOStatus) error {
	if slo == nil {
		if status != nil {
			return fmt.Errorf("taskmgr: %s has quality_slo_status without quality_slo", ctx)
		}
		return nil
	}
	if status == nil {
		return fmt.Errorf("taskmgr: %s has quality_slo without quality_slo_status", ctx)
	}
	if err := validateVerifiedState(status.WitnessState); err != nil {
		return fmt.Errorf("taskmgr: %s quality_slo_status witness_state: %w", ctx, err)
	}
	if status.StallCount < 0 {
		return fmt.Errorf("taskmgr: %s quality_slo_status stall_count is negative: %d", ctx, status.StallCount)
	}
	for _, reason := range status.Reasons {
		if strings.TrimSpace(reason) == "" {
			return fmt.Errorf("taskmgr: %s quality_slo_status has a blank reason", ctx)
		}
	}
	if status.Passed && len(status.Reasons) > 0 {
		return fmt.Errorf("taskmgr: %s quality_slo_status passed with failure reasons", ctx)
	}
	if !status.Passed && len(status.Reasons) == 0 {
		return fmt.Errorf("taskmgr: %s quality_slo_status failed without reasons", ctx)
	}
	return nil
}

func cloneQualitySLO(in *QualitySLO) *QualitySLO {
	if in == nil {
		return nil
	}
	out := *in
	if in.OutputShape != nil {
		shape := *in.OutputShape
		out.OutputShape = &shape
	}
	if in.MaxStallCount != nil {
		maxStalls := *in.MaxStallCount
		out.MaxStallCount = &maxStalls
	}
	return &out
}
