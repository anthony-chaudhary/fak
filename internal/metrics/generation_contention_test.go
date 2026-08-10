package metrics

import (
	"strings"
	"testing"
)

func TestMeasureGenerationContentionReduction(t *testing.T) {
	experiment := GenerationContentionExperiment{
		Generation: "gen/next",
		Before: GenerationContentionWindow{
			Issues: 10, Dispatches: 20, AmbiguityComments: 6,
			Relabels: 4, BlockedDispatches: 3, OperatorInterventions: 2,
		},
		After: GenerationContentionWindow{
			Issues: 20, Dispatches: 40, AmbiguityComments: 4,
			Relabels: 2, BlockedDispatches: 2, OperatorInterventions: 1,
		},
	}

	got, err := MeasureGenerationContention(experiment)
	if err != nil {
		t.Fatalf("MeasureGenerationContention: %v", err)
	}
	if got.BeforeEvents != 15 || got.AfterEvents != 9 {
		t.Fatalf("events = before %d after %d, want 15 and 9", got.BeforeEvents, got.AfterEvents)
	}
	if got.BeforeRatePer100 != 50 || got.AfterRatePer100 != 15 {
		t.Fatalf("rates = before %.2f after %.2f, want 50 and 15", got.BeforeRatePer100, got.AfterRatePer100)
	}
	if got.ReductionPercent != 70 || !got.Reduced {
		t.Fatalf("reduction = %.2f reduced=%v, want 70 true", got.ReductionPercent, got.Reduced)
	}
	if !strings.Contains(got.OperatorReadout, "gen/next") || !strings.Contains(got.OperatorReadout, "50.00 -> 15.00") {
		t.Fatalf("operator readout does not preserve the comparable before/after result: %q", got.OperatorReadout)
	}
}

func TestGenerationContentionContractNamesDecisionEvidence(t *testing.T) {
	contract := GenerationContentionContract()
	for name, value := range map[string]string{
		"orthogonality": contract.Orthogonality,
		"promotion":     contract.PromotionEvidence,
		"retirement":    contract.DemotionOrRetirementEvidence,
		"assumption":    contract.InvalidatingAssumption,
		"continuation":  contract.Continuation,
	} {
		if strings.TrimSpace(value) == "" {
			t.Errorf("%s evidence is empty", name)
		}
	}
	if !strings.Contains(contract.Orthogonality, "priority") ||
		!strings.Contains(contract.Orthogonality, "shared trunk") ||
		!strings.Contains(contract.Orthogonality, "runtime feature gates") {
		t.Fatalf("orthogonality statement is incomplete: %q", contract.Orthogonality)
	}
}

func TestMeasureGenerationContentionRejectsIncomparableWindows(t *testing.T) {
	_, err := MeasureGenerationContention(GenerationContentionExperiment{
		Generation: "gen/next",
		Before:     GenerationContentionWindow{Issues: 0, Dispatches: 1},
		After:      GenerationContentionWindow{Issues: 1, Dispatches: 1},
	})
	if err == nil {
		t.Fatal("expected zero issue exposure to be rejected")
	}
}
