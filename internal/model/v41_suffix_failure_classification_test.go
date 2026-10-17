package model

import (
	"errors"
	"fmt"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41SuffixFailureSelectedStageWrappers(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name     string
		cause    error
		selected bool
	}{
		{"retired backend", &BackendForwardOperationError{Path: "v41-routed-expert", Stage: "down", Cause: ErrV41ForwardStage}, true},
		{"selected expert", v41ExpertOperationErr(0, "down", ErrV41ForwardStage), true},
		{"structural refusal", v41StageErr(v41StageMoE, 0, ErrV41ForwardStage), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := fmt.Errorf("suffix operation: %w", tc.cause)
			if !errors.Is(err, ErrV41ForwardStage) {
				t.Fatal("fixture must preserve the structural stage sentinel")
			}
			if got := v41SuffixFailureIsSelected(&Model{}, err); got != tc.selected {
				t.Fatalf("selected=%t, want %t: %v", got, tc.selected, err)
			}
		})
	}
}
