package scorecardpane

import (
	"fmt"
	"strings"
)

// AffordanceRatioSchema tags the machine-readable payload for affordance ratio reporting.
const AffordanceRatioSchema = "fak-scorecard-affordance-ratio/1"

// DefaultAffordanceTargetRatio is the target ratio of interventions supplying
// an actionable next step versus bare denials (1.0 = 100%).
const DefaultAffordanceTargetRatio = 1.0

// InterventionRecord records a single intervention event, capturing whether it
// supplied an actionable next action or resulted in a bare denial.
type InterventionRecord struct {
	Tool               string `json:"tool,omitempty"`
	Reason             string `json:"reason,omitempty"`
	NextAction         string `json:"next_action,omitempty"`
	ActionableNextStep string `json:"actionable_next_step,omitempty"`
	HasNextAction      bool   `json:"has_next_action,omitempty"`
	Actionable         bool   `json:"actionable,omitempty"`
}

// HasActionableNextStep reports whether the intervention provides an actionable next step.
func (r InterventionRecord) HasActionableNextStep() bool {
	if strings.TrimSpace(r.NextAction) != "" || strings.TrimSpace(r.ActionableNextStep) != "" {
		return true
	}
	return r.HasNextAction || r.Actionable
}

// IsBareDenial reports whether the intervention was a bare denial without an actionable next step.
func (r InterventionRecord) IsBareDenial() bool {
	return !r.HasActionableNextStep()
}

// AffordanceRatioKPI tracks the ratio of actionable next-steps versus bare denials.
// The KPI passes when AffordanceRatio meets or exceeds TargetRatio.
type AffordanceRatioKPI struct {
	TotalInterventions          int     `json:"total_interventions"`
	InterventionsWithNextAction int     `json:"interventions_with_next_action"`
	AffordanceRatio             float64 `json:"affordance_ratio"`
	BareDenialCount             int     `json:"bare_denial_count"`
	TargetRatio                 float64 `json:"target_ratio"`
	Pass                        bool    `json:"pass"`
}

// NewAffordanceRatioKPI initializes an AffordanceRatioKPI with default target 1.0 and passing status.
func NewAffordanceRatioKPI() AffordanceRatioKPI {
	return AffordanceRatioKPI{
		TargetRatio:     DefaultAffordanceTargetRatio,
		AffordanceRatio: 1.0,
		Pass:            true,
	}
}

// Record records an intervention from an InterventionRecord, updating counts, ratio, and compliance.
func (k *AffordanceRatioKPI) Record(record InterventionRecord) {
	k.TotalInterventions++
	if record.HasActionableNextStep() {
		k.InterventionsWithNextAction++
	} else {
		k.BareDenialCount++
	}
	k.CalculateRatio()
	k.EvaluateCompliance()
}

// RecordIntervention records an intervention by whether it carries an actionable next step.
func (k *AffordanceRatioKPI) RecordIntervention(hasActionableNextStep bool) {
	k.Record(InterventionRecord{HasNextAction: hasActionableNextStep})
}

// RecordNextAction records an intervention that provided an actionable next step.
func (k *AffordanceRatioKPI) RecordNextAction() {
	k.RecordIntervention(true)
}

// RecordBareDenial records an intervention that was a bare denial without an actionable next step.
func (k *AffordanceRatioKPI) RecordBareDenial() {
	k.RecordIntervention(false)
}

// CalculateRatio computes and updates AffordanceRatio based on
// InterventionsWithNextAction and TotalInterventions.
// When TotalInterventions == 0, the ratio defaults to 1.0 (100%).
func (k *AffordanceRatioKPI) CalculateRatio() float64 {
	if k.TotalInterventions <= 0 {
		k.AffordanceRatio = 1.0
	} else {
		k.AffordanceRatio = float64(k.InterventionsWithNextAction) / float64(k.TotalInterventions)
	}
	return k.AffordanceRatio
}

// EvaluateCompliance checks whether AffordanceRatio meets or exceeds TargetRatio.
// If TargetRatio is zero or negative, it defaults to DefaultAffordanceTargetRatio (1.0).
func (k *AffordanceRatioKPI) EvaluateCompliance() bool {
	if k.TargetRatio <= 0 {
		k.TargetRatio = DefaultAffordanceTargetRatio
	}
	k.Pass = k.AffordanceRatio >= k.TargetRatio
	return k.Pass
}

// Calculate re-computes the ratio and evaluates compliance.
func (k *AffordanceRatioKPI) Calculate() {
	k.CalculateRatio()
	k.EvaluateCompliance()
}

// Evaluate is a convenience alias for EvaluateCompliance.
func (k *AffordanceRatioKPI) Evaluate() bool {
	return k.EvaluateCompliance()
}

// Deficit reports how far below TargetRatio the current AffordanceRatio is.
// Returns 0.0 when meeting or exceeding the target.
func (k AffordanceRatioKPI) Deficit() float64 {
	if k.AffordanceRatio >= k.TargetRatio {
		return 0.0
	}
	return k.TargetRatio - k.AffordanceRatio
}

// Summary returns a concise one-line summary of the KPI state.
func (k AffordanceRatioKPI) Summary() string {
	return fmt.Sprintf("affordance-ratio=%.1f%% (%d/%d interventions, %d bare denials, target=%.1f%%, pass=%t)",
		k.AffordanceRatio*100, k.InterventionsWithNextAction, k.TotalInterventions, k.BareDenialCount, k.TargetRatio*100, k.Pass)
}

// ComputeAffordanceRatio calculates AffordanceRatioKPI from a slice of InterventionRecord items.
func ComputeAffordanceRatio(records []InterventionRecord) AffordanceRatioKPI {
	kpi := NewAffordanceRatioKPI()
	for _, r := range records {
		kpi.Record(r)
	}
	kpi.CalculateRatio()
	kpi.EvaluateCompliance()
	return kpi
}

// ComputeAffordanceRatioWithTarget calculates AffordanceRatioKPI with an explicit target ratio.
func ComputeAffordanceRatioWithTarget(records []InterventionRecord, targetRatio float64) AffordanceRatioKPI {
	kpi := NewAffordanceRatioKPI()
	if targetRatio > 0 {
		kpi.TargetRatio = targetRatio
	}
	for _, r := range records {
		kpi.Record(r)
	}
	kpi.CalculateRatio()
	kpi.EvaluateCompliance()
	return kpi
}
