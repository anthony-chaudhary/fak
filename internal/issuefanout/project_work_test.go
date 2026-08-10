package issuefanout

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

func TestBuildProjectWorkMetadataPassesStrictReview(t *testing.T) {
	p, err := Build(Input{Title: "model bring-up", Leaf: "model", SpineRef: "abc123", ParentIssue: 36, ParentBaseline: 100, CompletionStandard: "production", TargetEnvelope: "- concurrent users: 10 users\n- sustained duration: 60 minutes", WitnessedEnvelope: "- concurrent users: 10 users\n- sustained duration: 60 minutes", Max: MinFanout})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range p.Candidates {
		r := issuepolicy.ReviewCandidate(c, issuepolicy.Options{StrictProjectWork: true})
		if r.ProjectWork.Status != issuepolicy.ProjectWorkValid {
			t.Fatalf("%s project work = %+v", c.Key, r.ProjectWork)
		}
		if !r.ProjectWork.ProductionCredit {
			t.Fatalf("%s lacks production credit metadata", c.Key)
		}
	}
}

func TestBuildPreservesToyBringupAsDemo(t *testing.T) {
	p, err := Build(Input{Title: "toy model bring-up", Leaf: "model", SpineRef: "abc123", ParentIssue: 36, ParentBaseline: 100, CompletionStandard: "demo", Max: MinFanout})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range p.Candidates {
		if c.CompletionStandard != "demo" {
			t.Fatalf("maturity=%q", c.CompletionStandard)
		}
		if !strings.Contains(LiveBody(c), "## Completion standard\n\ndemo") {
			t.Fatalf("body hid demo maturity:\n%s", LiveBody(c))
		}
	}
}
