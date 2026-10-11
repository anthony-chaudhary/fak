package model

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=6s lane=default
// fak-test:usefulness protects=v41-per-output-full-depth-forward-coverage
func TestNativeForwardCoverageV41TraversalAndValidation(t *testing.T) {
	m := v41IncrementalExpertFixture(t, false, false)
	s := m.NewSession()
	t.Cleanup(s.Close)

	if logits := s.Prefill([]int{1, 2, 3}); len(logits) == 0 {
		t.Fatal("cold prefill returned no logits")
	}
	cold := s.NativeForwardCoverage()
	if cold == nil {
		t.Fatal("cold prefill published no forward coverage")
	}
	if cold.Schema != NativeForwardCoverageSchemaV1 || cold.ModelType != m.Cfg.ModelType ||
		cold.ConfiguredLayers != m.Cfg.NumLayers || cold.HeadDim != m.Cfg.HeadDim ||
		cold.ExecutedLayers != m.Cfg.NumLayers || !cold.FullDepth || !cold.Complete ||
		cold.LogitsSource != NativeForwardLogitsSourceV41Full {
		t.Fatalf("cold coverage = %+v, config layers/head/model = %d/%d/%q", *cold, m.Cfg.NumLayers, m.Cfg.HeadDim, m.Cfg.ModelType)
	}
	if err := cold.Validate(); err != nil {
		t.Fatalf("cold coverage validation: %v", err)
	}

	if logits := s.Step(4); len(logits) == 0 {
		t.Fatal("incremental step returned no logits")
	}
	step := s.NativeForwardCoverage()
	if step == nil || step.ExecutedLayers != m.Cfg.NumLayers || step.LogitsSource != NativeForwardLogitsSourceV41Incremental {
		t.Fatalf("incremental coverage = %+v, want %d executed layers from incremental source", step, m.Cfg.NumLayers)
	}
	if err := ValidateNativeForwardCoverage([]NativeForwardCoverage{*cold, *step}, 2); err != nil {
		t.Fatalf("valid per-token coverage: %v", err)
	}

	for _, tc := range []struct {
		name     string
		coverage []NativeForwardCoverage
		tokens   int
	}{
		{name: "cardinality", coverage: []NativeForwardCoverage{*cold}, tokens: 2},
		{name: "incomplete", coverage: []NativeForwardCoverage{func() NativeForwardCoverage { c := *cold; c.Complete = false; return c }()}, tokens: 1},
		{name: "short traversal", coverage: []NativeForwardCoverage{func() NativeForwardCoverage { c := *cold; c.ExecutedLayers--; c.FullDepth = false; return c }()}, tokens: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateNativeForwardCoverage(tc.coverage, tc.tokens); !errors.Is(err, ErrNativeForwardCoverageInvalid) {
				t.Fatalf("ValidateNativeForwardCoverage error = %v, want %v", err, ErrNativeForwardCoverageInvalid)
			}
		})
	}
}
