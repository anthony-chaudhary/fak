package nativeperf

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestComparisonCriterionMutationChangesIdentity(t *testing.T) {
	r := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	criterion, err := ResolveComparisonCriterion(r)
	if err != nil {
		t.Fatal(err)
	}
	before, err := comparisonCriterionDigest(criterion)
	if err != nil {
		t.Fatal(err)
	}
	criterion.MinimumQualityScore = .99
	after, err := comparisonCriterionDigest(criterion)
	if err != nil {
		t.Fatal(err)
	}
	if before == after {
		t.Fatal("criterion mutation did not change its digest")
	}
}

func TestGateRejectsStaleOrMissingFreeFormCriterionBounds(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*GatePolicy)
	}{
		{"stale", func(p *GatePolicy) { p.MinimumQualityScore = .99 }},
		{"missing", func(p *GatePolicy) { p.MinimumThroughput = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := gateRequest(t)
			tc.edit(&r.Policy)
			if _, err := Gate(r); err == nil || !strings.Contains(err.Error(), "criterion") {
				t.Fatalf("err=%v", err)
			}
		})
	}
}

func TestComparisonCriterionReplayAndExistingPathDigest(t *testing.T) {
	r := gateRequest(t)
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var replay GateRequest
	if err := json.Unmarshal(data, &replay); err != nil {
		t.Fatal(err)
	}
	first, err := Gate(r)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Gate(replay)
	if err != nil {
		t.Fatal(err)
	}
	if first.CriterionDigest != second.CriterionDigest || first.Classification != second.Classification || first.CriterionDigest == "" || first.CriterionDigest != r.LastAccepted.Comparison.CriterionDigest {
		t.Fatalf("replay identity mismatch: first=%+v second=%+v", first, second)
	}
	comparison, err := CompareReceipts(ActiveGraph(), r.LastAccepted, r.Candidate)
	if err != nil {
		t.Fatal(err)
	}
	if comparison.CriterionDigest != first.CriterionDigest {
		t.Fatalf("comparison digest=%q verdict digest=%q", comparison.CriterionDigest, first.CriterionDigest)
	}
}

func TestComparisonCriterionRejectsNonFiniteBoundsAndQuality(t *testing.T) {
	r := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	criterion, err := ResolveComparisonCriterion(r)
	if err != nil {
		t.Fatal(err)
	}
	criterion.MaximumNoisePercent = math.NaN()
	if _, err := comparisonCriterionDigest(criterion); err == nil {
		t.Fatal("non-finite criterion bound accepted")
	}
	r.Quality.Score = math.Inf(1)
	if err := ValidateReceipt(ActiveGraph(), r); err == nil || !strings.Contains(err.Error(), "finite") {
		t.Fatalf("err=%v", err)
	}
}
