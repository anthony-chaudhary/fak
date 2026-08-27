package systembaseline

import (
	"encoding/json"
	"testing"
	"time"
)

func TestHealthScorecardEmitsGradeAndNamedEvidence(t *testing.T) {
	clean := Build(quietFixture(100e6), fixture(100e6, 0), 10, time.Second, DefaultPolicy(), 0, false)
	report := HealthScorecard([]Report{clean})
	if report.Schema != ScorecardSchema || !report.OK {
		t.Fatalf("scorecard = %+v", report)
	}
	if report.Corpus["grade"] != "A" {
		t.Fatalf("grade = %v, want A", report.Corpus["grade"])
	}
	evidence, ok := report.Corpus["evidence"].(map[string]int)
	if !ok || evidence["total"] != 1 || evidence["success"] != 1 {
		t.Fatalf("evidence = %#v", report.Corpus["evidence"])
	}
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("system-baseline scorecard: %s", raw)
}

func TestHealthScorecardGradesMissingAndUnhealthyEvidence(t *testing.T) {
	missing := HealthScorecard(nil)
	if missing.OK || missing.Corpus[ScorecardDebtKey] != 1 {
		t.Fatalf("missing evidence scorecard = %+v", missing)
	}

	refusalPolicy := DefaultPolicy()
	refusalPolicy.MaximumNonSUTCPUPercent = 1
	refusal := Build(quietFixture(100e6), fixture(500e6, 0), 10, time.Second, refusalPolicy, 0, false)
	corrupt := refusal
	corrupt.Digest = "sha256:corrupt"
	unhealthy := HealthScorecard([]Report{refusal, corrupt})
	if unhealthy.OK || unhealthy.Corpus[ScorecardDebtKey] != 2 {
		t.Fatalf("unhealthy scorecard = %+v", unhealthy)
	}
	evidence := unhealthy.Corpus["evidence"].(map[string]int)
	if evidence["refusal"] != 1 || evidence["error"] != 1 {
		t.Fatalf("evidence = %#v", evidence)
	}
}
