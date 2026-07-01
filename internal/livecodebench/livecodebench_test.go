package livecodebench

import (
	"strings"
	"testing"
)

func TestLoadFixtureAndSmokeReport(t *testing.T) {
	fixture, err := LoadFile("testdata/fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(ScenarioNames(fixture), ","); got != "codeexecution,codegeneration,selfrepair,testoutputprediction" {
		t.Fatalf("scenarios = %q", got)
	}
	report := SmokeReport(fixture)
	if err := ValidateSmokeReport(report); err != nil {
		t.Fatal(err)
	}
	if report.ResultClaimAllowed {
		t.Fatal("fixture smoke must not allow a result claim")
	}
	if report.EvidenceClass != "fixture-smoke" {
		t.Fatalf("evidence class = %q", report.EvidenceClass)
	}
}

func TestValidateSmokeReportRejectsClaimAllowed(t *testing.T) {
	report := SmokeReport(Fixture{
		Schema:         FixtureSchema,
		ReleaseVersion: "fixture_release",
		StartDate:      "2026-01-01",
		EndDate:        "2026-01-02",
		Items: []FixtureItem{
			{QuestionID: "cg", Scenario: "codegeneration"},
			{QuestionID: "sr", Scenario: "selfrepair"},
			{QuestionID: "to", Scenario: "testoutputprediction"},
			{QuestionID: "ce", Scenario: "codeexecution"},
		},
	})
	report.ResultClaimAllowed = true
	err := ValidateSmokeReport(report)
	if err == nil || !strings.Contains(err.Error(), "result_claim_allowed must be false") {
		t.Fatalf("err = %v, want result_claim_allowed refusal", err)
	}
}

func TestLoadRejectsUnknownScenario(t *testing.T) {
	body := `{
		"schema":"fak.livecodebench.fixture.v1",
		"release_version":"fixture_release",
		"start_date":"2026-01-01",
		"end_date":"2026-01-02",
		"items":[{"question_id":"bad","scenario":"unknown","prompt":"x"}]
	}`
	_, err := Load(strings.NewReader(body))
	if err == nil || !strings.Contains(err.Error(), "not supported") {
		t.Fatalf("err = %v, want unsupported scenario", err)
	}
}
