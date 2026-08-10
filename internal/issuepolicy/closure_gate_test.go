package issuepolicy

import (
	"encoding/json"
	"strings"
	"testing"
)

func closureCandidate() Candidate {
	c := envelopeCandidate()
	c.WorkEstimate = "Estimate: 3 points (small)"
	c.ScopeContribution = "Contribution: 3/34 points"
	c.CompletionStandard = "production"
	c.TargetEnvelope = "- concurrency: >= 1 users"
	c.WitnessedEnvelope = "- concurrency: 1 users"
	c.Witness = "dos commit-audit returns OK and go test ./internal/issuecontract passes"
	return c
}

func TestClosureGateDefaultsBareCompletionToProduction(t *testing.T) {
	c := closureCandidate()
	c.ClosureClaim = "complete"
	c.ClosureWitnessStandard = "production"
	r := ReviewCandidate(c, Options{})
	if !r.OK || r.Closure.Status != ClosureEligible || r.Closure.ClaimedStandard != "production" || !r.Closure.ProductionCredit {
		t.Fatalf("review=%+v closure=%+v", r, r.Closure)
	}
}

func TestClosureGateDemoCanCloseWithoutProductionCredit(t *testing.T) {
	c := closureCandidate()
	c.CompletionStandard = "demo"
	c.TargetEnvelope = ""
	c.WitnessedEnvelope = ""
	c.ClosureClaim = "demo complete"
	c.ClosureWitnessStandard = "demo"
	r := ReviewCandidate(c, Options{})
	if !r.OK || r.Closure.Status != ClosureEligible || r.Closure.ProductionCredit || r.ProjectWork.ProductionCredit {
		t.Fatalf("demo closure=%+v project=%+v reasons=%v", r.Closure, r.ProjectWork, r.Reasons)
	}
}

func TestClosureGateRefusesToyDemoAsProduction(t *testing.T) {
	c := closureCandidate()
	c.CompletionStandard = "demo"
	c.ClosureClaim = "complete"
	c.ClosureWitnessStandard = "demo"
	r := ReviewCandidate(c, Options{})
	for _, want := range []string{ReasonClosureWitnessMismatch, ReasonClosureProductionGap} {
		if !containsString(r.Reasons, want) {
			t.Fatalf("reasons=%v want=%s", r.Reasons, want)
		}
	}
	if r.Closure.Status != ClosureRefused || r.Closure.ProductionCredit {
		t.Fatalf("closure=%+v", r.Closure)
	}
}

func TestClosureGateMissingWitnessFailsClosedWithRepair(t *testing.T) {
	c := closureCandidate()
	c.ClosureClaim = "production complete"
	c.ClosureWitnessStandard = ""
	c.Witness = "agent says it completed"
	r := ReviewCandidate(c, Options{})
	if !containsString(r.Reasons, ReasonClosureWitnessMissing) || len(r.Closure.Repair) == 0 {
		t.Fatalf("closure=%+v reasons=%v", r.Closure, r.Reasons)
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"closure"`, ReasonClosureWitnessMissing, `"production_credit":false`} {
		if !strings.Contains(string(b), want) {
			t.Fatalf("JSON missing %q: %s", want, b)
		}
	}
}

func TestCandidateFromIssueDraftParsesClosureSections(t *testing.T) {
	c := CandidateFromIssueDraft(IssueDraft{Title: "close", Body: "## Closure claim\ndemo complete\n\n## Closure witness standard\ndemo"})
	if c.ClosureClaim != "demo complete" || c.ClosureWitnessStandard != "demo" {
		t.Fatalf("candidate=%+v", c)
	}
}
