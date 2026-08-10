package metrics

import (
	"fmt"
	"strings"
)

// GenerationContentionWindow records the observable planning-conflict signals
// for one equally defined measurement window.
type GenerationContentionWindow struct {
	Issues                int `json:"issues"`
	Dispatches            int `json:"dispatches"`
	AmbiguityComments     int `json:"ambiguity_comments"`
	Relabels              int `json:"relabels"`
	BlockedDispatches     int `json:"blocked_dispatches"`
	OperatorInterventions int `json:"operator_interventions"`
}

// GenerationContentionExperiment compares a baseline without generation
// labels with a dogfood window using one generation label.
type GenerationContentionExperiment struct {
	Generation string                     `json:"generation"`
	Before     GenerationContentionWindow `json:"before"`
	After      GenerationContentionWindow `json:"after"`
}

// GenerationContentionResult is the agent-runnable before/after readout.
type GenerationContentionResult struct {
	Generation       string  `json:"generation"`
	BeforeEvents     int     `json:"before_events"`
	AfterEvents      int     `json:"after_events"`
	BeforeRatePer100 float64 `json:"before_rate_per_100_opportunities"`
	AfterRatePer100  float64 `json:"after_rate_per_100_opportunities"`
	ReductionPercent float64 `json:"reduction_percent"`
	Reduced          bool    `json:"reduced"`
	OperatorReadout  string  `json:"operator_readout"`
}

// GenerationContentionDecisionContract keeps the dogfood decision criteria
// beside the measurement so a future agent need not reconstruct the epic.
type GenerationContentionDecisionContract struct {
	Orthogonality                string `json:"orthogonality"`
	PromotionEvidence            string `json:"promotion_evidence"`
	DemotionOrRetirementEvidence string `json:"demotion_or_retirement_evidence"`
	InvalidatingAssumption       string `json:"invalidating_assumption"`
	Continuation                 string `json:"continuation"`
}

// GenerationContentionContract returns the fixed interpretation contract.
func GenerationContentionContract() GenerationContentionDecisionContract {
	return GenerationContentionDecisionContract{
		Orthogonality:                "Generation is a planning horizon only; it does not set priority, create a branch or shared trunk exception, or replace runtime feature gates.",
		PromotionEvidence:            "Promote after two comparable dogfood windows show a lower contention-event rate per 100 issue-plus-dispatch opportunities without an increase in blocked dispatches.",
		DemotionOrRetirementEvidence: "Demote or retire the label workflow when two comparable windows show no reduction, relabel churn increases, or operator interventions move rather than disappear.",
		InvalidatingAssumption:       "The comparison is invalid if label adoption or event collection differs materially between windows, because missing observations can look like reduced contention.",
		Continuation:                 "Collect before and after counts for ambiguity comments, generation relabels, blocked dispatches, and operator interventions; then call MeasureGenerationContention and retain its readout with the window dates.",
	}
}

// MeasureGenerationContention normalizes observed conflict signals by issue and
// dispatch opportunities so differently sized windows remain comparable.
func MeasureGenerationContention(experiment GenerationContentionExperiment) (GenerationContentionResult, error) {
	if strings.TrimSpace(experiment.Generation) == "" {
		return GenerationContentionResult{}, fmt.Errorf("generation is required")
	}
	if err := validateGenerationContentionWindow("before", experiment.Before); err != nil {
		return GenerationContentionResult{}, err
	}
	if err := validateGenerationContentionWindow("after", experiment.After); err != nil {
		return GenerationContentionResult{}, err
	}

	beforeEvents := experiment.Before.events()
	afterEvents := experiment.After.events()
	beforeRate := float64(beforeEvents) * 100 / float64(experiment.Before.Issues+experiment.Before.Dispatches)
	afterRate := float64(afterEvents) * 100 / float64(experiment.After.Issues+experiment.After.Dispatches)
	reduction := 0.0
	if beforeRate > 0 {
		reduction = (beforeRate - afterRate) * 100 / beforeRate
	}

	return GenerationContentionResult{
		Generation:       experiment.Generation,
		BeforeEvents:     beforeEvents,
		AfterEvents:      afterEvents,
		BeforeRatePer100: beforeRate,
		AfterRatePer100:  afterRate,
		ReductionPercent: reduction,
		Reduced:          afterRate < beforeRate,
		OperatorReadout: fmt.Sprintf("generation %s contention events/100 opportunities: %.2f -> %.2f (reduction %.2f%%)",
			experiment.Generation, beforeRate, afterRate, reduction),
	}, nil
}

func validateGenerationContentionWindow(name string, window GenerationContentionWindow) error {
	if window.Issues <= 0 || window.Dispatches <= 0 {
		return fmt.Errorf("%s window requires positive issue and dispatch opportunities", name)
	}
	counts := []int{window.AmbiguityComments, window.Relabels, window.BlockedDispatches, window.OperatorInterventions}
	for _, count := range counts {
		if count < 0 {
			return fmt.Errorf("%s window contains a negative event count", name)
		}
	}
	return nil
}

func (window GenerationContentionWindow) events() int {
	return window.AmbiguityComments + window.Relabels + window.BlockedDispatches + window.OperatorInterventions
}
