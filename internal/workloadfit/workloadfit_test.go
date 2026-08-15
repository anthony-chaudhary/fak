package workloadfit

import (
	"strings"
	"testing"
	"time"
)

func TestCodingAndLegalSelectDifferentStacks(t *testing.T) {
	coding, legal, err := Selfcheck()
	if err != nil {
		t.Fatal(err)
	}
	if coding.Chosen == legal.Chosen {
		t.Fatalf("same choice %q", coding.Chosen)
	}
	ponytailLegal := assessment(legal, "ponytail@r8")
	if ponytailLegal.Status != "refuse" || !finding(ponytailLegal, "citations", Missing) || !finding(ponytailLegal, "human-review", Denied) {
		t.Fatalf("legal ponytail assessment = %+v", ponytailLegal)
	}
}

func TestUnknownExpiredUnsupportedRemainDistinct(t *testing.T) {
	raw := []byte(strings.Replace(string(codingLegalFixture), "2026-09-15T00:00:00Z", "2026-08-01T00:00:00Z", 1))
	fixture, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	assessment := Assess(fixture.Contracts[0], fixture.Catalog.Candidates[0], fixture.AsOf)
	if !finding(assessment, "loop", Expired) {
		t.Fatalf("findings = %+v", assessment.Findings)
	}
}

func TestStackAdapterExcludesPreferencesAndCost(t *testing.T) {
	fixture, err := Parse(codingLegalFixture)
	if err != nil {
		t.Fatal(err)
	}
	relations := StackRequirements(fixture.Contracts[0])
	if len(relations) != 2 {
		t.Fatalf("relations = %+v", relations)
	}
	for _, relation := range relations {
		if relation.Target == "coding.fast-loop" || relation.Target == "usd_per_task" {
			t.Fatalf("soft requirement leaked: %+v", relation)
		}
	}
}

func TestEvidenceFloorRejectsDeclaredClaim(t *testing.T) {
	req := Requirement{ID: "citation", Class: Evidence, Capability: "citation.traceable", MinimumTier: Evaluated, Source: Source{Authority: "domain", Reference: "c"}}
	contract := Contract{Schema: Schema, ID: "legal@1", Domain: "legal", Revision: "1", Owner: "domain", Requirements: []Requirement{req}}
	candidate := Candidate{ID: "h@1", Claims: []Claim{{Capability: "citation.traceable", Status: Supported, Source: Source{Authority: "vendor", Reference: "claim", Tier: Declared}}}}
	got := Assess(contract, candidate, time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC))
	if got.Status != "refuse" || !finding(got, "citation", Missing) {
		t.Fatalf("assessment = %+v", got)
	}
}

func assessment(selection Selection, id string) Assessment {
	for _, a := range selection.Assessments {
		if a.CandidateID == id {
			return a
		}
	}
	return Assessment{}
}
func finding(a Assessment, id string, state FindingState) bool {
	for _, f := range a.Findings {
		if f.Requirement == id && f.State == state {
			return true
		}
	}
	return false
}
