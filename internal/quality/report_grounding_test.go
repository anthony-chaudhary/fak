package quality

import (
	"strings"
	"testing"
)

// execEvidence is the hermetic allowed-source corpus for the grounding tests:
// two source snippets an executive rollup may draw claims from.
var execEvidence = []string{
	"Weekly ops rollup: throughput increased 12% week over week.",
	"Support queue: median latency held flat at 250ms for the week.",
}

const (
	// faithfulReport asserts two claims, each backed by one evidence snippet.
	faithfulReport = "Throughput increased 12% week over week. Median latency held flat at 250ms."
	// fabricatedClaim is supported by NO evidence snippet — the hallucination.
	fabricatedClaim = "Churn dropped 40% quarter over quarter"
	// fabricatedReport is the faithful report plus the fabricated claim.
	fabricatedReport = faithfulReport + " " + fabricatedClaim + "."
)

// groundingCase builds a valid case whose Rubric.Required carries the evidence
// snippets the claim-grounding oracle checks report claims against.
func groundingCase(evidence []string, minScore float64) QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "claim-grounding-exec-report",
		Version:   1,
		Prompt:    "Summarize the weekly ops evidence for the executive rollup.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 32},
		Reference: Trace{Text: faithfulReport},
		Oracles:   []string{"claim-grounding"},
		Rubric:    RubricSpec{Required: evidence, MinScore: minScore},
	}
}

// TestClaimGroundingRegistered proves the oracle registered under its stable
// name and kind, so cases can reference it by name.
func TestClaimGroundingRegistered(t *testing.T) {
	os, err := Lookup([]string{"claim-grounding"})
	if err != nil {
		t.Fatalf("Lookup(claim-grounding): %v", err)
	}
	if got := os[0].Kind(); got != "rubric" {
		t.Errorf("Kind() = %q, want rubric", got)
	}
}

// TestClaimGroundingFaithfulReportPasses is the happy path: every claim in the
// report is backed by an evidence snippet, so the oracle passes at score 1.0.
func TestClaimGroundingFaithfulReportPasses(t *testing.T) {
	c := groundingCase(execEvidence, 1)
	v := ClaimGrounding{}.Judge(Trace{}, Trace{Text: faithfulReport}, c)
	if !v.Pass {
		t.Fatalf("faithful report must pass; got %+v", v)
	}
	if v.Score != 1 {
		t.Errorf("score = %v, want 1.0", v.Score)
	}
}

// TestClaimGroundingFabricatedClaimFails is the defect witness: one claim no
// evidence snippet supports fails the oracle, and Detail names that exact claim.
func TestClaimGroundingFabricatedClaimFails(t *testing.T) {
	c := groundingCase(execEvidence, 1)
	v := ClaimGrounding{}.Judge(Trace{}, Trace{Text: fabricatedReport}, c)
	if v.Pass {
		t.Fatalf("fabricated claim must not pass; got %+v", v)
	}
	if want := 2.0 / 3.0; v.Score != want {
		t.Errorf("score = %v, want %v (2 of 3 claims grounded)", v.Score, want)
	}
	if !strings.Contains(v.Detail, fabricatedClaim) {
		t.Errorf("Detail must name the fabricated claim %q; got %q", fabricatedClaim, v.Detail)
	}
}

// TestClaimGroundingSpineIntegration runs the fabricated report through the full
// spine: the failure bundle names claim-grounding as the failing oracle and
// carries the offending claim in its detail.
func TestClaimGroundingSpineIntegration(t *testing.T) {
	c := groundingCase(execEvidence, 1)
	eng := ScriptedRunner{Label: "engine-fabricated-claim", Trace: Trace{Text: fabricatedReport}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("fabricated report must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "claim-grounding" {
		t.Errorf("failing oracle = %q, want claim-grounding", fb.FailingOracle)
	}
	if !strings.Contains(fb.Detail, fabricatedClaim) {
		t.Errorf("bundle detail must name the fabricated claim; got %q", fb.Detail)
	}
}

// TestClaimGroundingMinScoreTolerance proves the threshold gate: the same
// fabricated report (2/3 grounded) passes when the case tolerates MinScore 0.5.
func TestClaimGroundingMinScoreTolerance(t *testing.T) {
	c := groundingCase(execEvidence, 0.5)
	v := ClaimGrounding{}.Judge(Trace{}, Trace{Text: fabricatedReport}, c)
	if !v.Pass {
		t.Fatalf("2/3 grounded must pass at MinScore 0.5; got %+v", v)
	}
	if !strings.Contains(v.Detail, fabricatedClaim) {
		t.Errorf("tolerated-ungrounded detail should still name the claim; got %q", v.Detail)
	}
}

// TestClaimGroundingEmptyReport defines the empty-report edge: no claims are
// asserted, so nothing can be ungrounded — pass at score 1, no panic.
func TestClaimGroundingEmptyReport(t *testing.T) {
	c := groundingCase(execEvidence, 1)
	for _, text := range []string{"", "   \n  ", "."} {
		v := ClaimGrounding{}.Judge(Trace{}, Trace{Text: text}, c)
		if !v.Pass || v.Score != 1 {
			t.Errorf("empty report %q: got %+v, want pass at score 1", text, v)
		}
	}
}

// TestClaimGroundingEmptyEvidence defines the empty-evidence edge: a report that
// asserts claims against no evidence fails closed at score 0, naming the first
// claim — an unsupported assertion is a hallucination, not a skipped check.
func TestClaimGroundingEmptyEvidence(t *testing.T) {
	c := groundingCase(nil, 1)
	v := ClaimGrounding{}.Judge(Trace{}, Trace{Text: faithfulReport}, c)
	if v.Pass {
		t.Fatalf("claims with no evidence must fail closed; got %+v", v)
	}
	if v.Score != 0 {
		t.Errorf("score = %v, want 0", v.Score)
	}
	if !strings.Contains(v.Detail, "Throughput increased 12% week over week") {
		t.Errorf("Detail must name the first ungrounded claim; got %q", v.Detail)
	}
}
