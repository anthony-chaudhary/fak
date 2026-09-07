package main

import (
	"context"
	"io"
	"testing"
	"time"
)

func TestIsGPURelatedValidation(t *testing.T) {
	tests := []struct {
		name     string
		mine     []string
		expected bool
	}{
		{
			name:     "non-gpu changes",
			mine:     []string{"cmd/fak/new_verb.go", "internal/policy/policy.go", "docs/README.md"},
			expected: false,
		},
		{
			name:     "amdgpu package changes",
			mine:     []string{"internal/amdgpu/strixhalo.go"},
			expected: true,
		},
		{
			name:     "compute package changes",
			mine:     []string{"internal/compute/vulkan.go"},
			expected: true,
		},
		{
			name:     "roofline package changes",
			mine:     []string{"internal/roofline/empirical.go"},
			expected: true,
		},
		{
			name:     "acceptance validation changes",
			mine:     []string{"cmd/fak/validate_acceptance.go"},
			expected: true,
		},
		{
			name:     "strix named file changes",
			mine:     []string{"internal/devcmd/amd_strix_validate.go"},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isGPURelatedValidation(tt.mine)
			if got != tt.expected {
				t.Errorf("isGPURelatedValidation(%v) = %v, want %v", tt.mine, got, tt.expected)
			}
		})
	}
}

func TestValidateStrix(t *testing.T) {
	// Invariants on shouldRunStrixValidation
	if !shouldRunStrixValidation(true, nil) {
		t.Errorf("expected explicit strix to run validation")
	}
	if shouldRunStrixValidation(false, []string{"docs/README.md"}) {
		t.Errorf("expected non-gpu paths without explicit flag to skip")
	}
	if !shouldRunStrixValidation(false, []string{"internal/amdgpu/strix_validation.go"}) {
		t.Errorf("expected gpu paths to trigger strix validation check")
	}

	// Execution phase fast-skip on non-GPU changes
	var res validateResult
	recorder := &validateRecorder{
		ctx:     context.Background(),
		stderr:  io.Discard,
		started: time.Now(),
		res:     &res,
	}
	err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, &res, recorder, false, "", "", "", []string{"docs/README.md"})
	if err != nil {
		t.Fatalf("unexpected error on non-gpu skip: %v", err)
	}
	foundSkipped := false
	for _, p := range res.SkippedPhases {
		if p == "strix_validation" {
			foundSkipped = true
			break
		}
	}
	if !foundSkipped {
		t.Errorf("expected strix_validation in skipped phases, got %v", res.SkippedPhases)
	}
}
