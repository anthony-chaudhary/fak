package issuepolicy

import (
	"strings"
	"testing"
)

func envelopeCandidate() Candidate {
	return Candidate{
		Key: "issue/4648", Title: "model serving envelope", ParentRef: "#4636",
		CurrentState: "Toy path passes.", WhyNow: "Production target is larger.", WorkingSpine: "declare -> compare -> review",
		WorkUnit: "leaf", ExpectedSteps: 3, InScope: "Envelope contract.", OutOfScope: "Load generator.",
		DoneCondition: "Target is enforced.", Witness: "go test ./internal/issuecontract", AcceptanceGate: "go test ./internal/issuecontract",
		Lane: "issuecontract", Paths: []string{"internal/issuecontract/**"}, ClosureBinding: "Commit cites #4648.",
		ProblemFrame: completeProblemFrame(),
	}
}

func TestOperatingEnvelopeProductionToyLoadStaysBelowTarget(t *testing.T) {
	c := envelopeCandidate()
	c.CompletionStandard = "Production complete"
	c.TargetEnvelope = "- concurrency: >= 1000 requests\n- duration: >= 60 minutes"
	c.WitnessedEnvelope = "- concurrency: 1 requests\n- duration: 1 minutes"
	r := ReviewCandidate(c, Options{})
	if r.OK || r.OperatingEnvelope.Status != EnvelopeStatusGap || !hasReason(r.Reasons, ReasonEnvelopeUnderTarget) {
		t.Fatalf("review = %+v, want fail-closed envelope gap", r)
	}
	if len(r.OperatingEnvelope.Gaps) != 2 {
		t.Fatalf("gaps = %+v, want concurrency and duration", r.OperatingEnvelope.Gaps)
	}
}

func TestOperatingEnvelopeProductionTargetMet(t *testing.T) {
	c := envelopeCandidate()
	c.CompletionStandard = "production"
	c.TargetEnvelope = "- concurrency: >= 1 user\n- error rate: <= 1 percent\n- regions: not-applicable (local-only command)"
	c.WitnessedEnvelope = "- concurrency: 1 user\n- error rate: 0 percent"
	r := ReviewCandidate(c, Options{})
	if !r.OK || r.OperatingEnvelope.Status != EnvelopeMet || len(r.OperatingEnvelope.Gaps) != 0 {
		t.Fatalf("review = %+v, want met single-user envelope", r)
	}
}

func TestOperatingEnvelopeProductionRequiresTarget(t *testing.T) {
	c := envelopeCandidate()
	c.CompletionStandard = "production"
	r := ReviewCandidate(c, Options{})
	if r.OK || r.OperatingEnvelope.Status != EnvelopeTargetMissing || !hasReason(r.Reasons, ReasonTargetEnvelopeMissing) {
		t.Fatalf("review = %+v, want missing target refusal", r)
	}
}

func TestOperatingEnvelopeNonProductionMayDeclareNarrowWitness(t *testing.T) {
	c := envelopeCandidate()
	c.CompletionStandard = "demo"
	c.WitnessedEnvelope = "- concurrency: 1 request"
	r := ReviewCandidate(c, Options{})
	if !r.OK || r.OperatingEnvelope.Status != EnvelopeNotRequired || r.OperatingEnvelope.Required {
		t.Fatalf("review = %+v, want explicit demo envelope without production implication", r)
	}
}

func TestOperatingEnvelopeRejectsMalformedAndMixedUnits(t *testing.T) {
	c := envelopeCandidate()
	c.CompletionStandard = "production"
	c.TargetEnvelope = "- concurrency: >= 1000 requests\n- duration: n/a"
	c.WitnessedEnvelope = "- concurrency: 1000 users"
	r := ReviewCandidate(c, Options{})
	if r.OK || !hasReason(r.Reasons, ReasonEnvelopeInvalid) || r.OperatingEnvelope.Status != EnvelopeInvalid {
		t.Fatalf("review = %+v, want invalid not-applicable declaration", r)
	}

	c.TargetEnvelope = "- concurrency: >= 1000 requests"
	c.WitnessedEnvelope = "- concurrency: 1000 users"
	r = ReviewCandidate(c, Options{})
	if r.OK || !hasReason(r.Reasons, ReasonEnvelopeUnderTarget) || !strings.Contains(r.OperatingEnvelope.Gaps[0].Reason, "unit mismatch") {
		t.Fatalf("review = %+v, want explicit unit mismatch gap", r)
	}
}

func TestIssueDraftParsesOperatingEnvelopeSections(t *testing.T) {
	body := strings.Join([]string{
		"## Parent context", "#4636", "## Current state", "Toy path passes.",
		"## Why this is next", "Scale is unknown.", "## Working spine", "declare -> compare -> review",
		"## Work unit", "leaf", "## Expected steps", "3", "## In scope", "Envelope contract.",
		"## Out of scope", "Load generator.", "## Done condition", "Target enforced.",
		"## Witness", "go test ./internal/issuecontract", "## Acceptance gate", "go test ./internal/issuecontract",
		"## Lane", "issuecontract", "## Likely files", "- `internal/issuecontract/**`",
		"## Closure binding", "Commit cites #4648.", "## Completion standard", "production",
		"## Target operating envelope", "- concurrency: >= 1000 requests",
		"## Witnessed operating envelope", "- concurrency: 1 requests",
	}, "\n")
	r := ReviewIssueDraft(IssueDraft{Number: 4648, Title: "model serving envelope", Body: body}, Options{})
	if r.OperatingEnvelope.Status != EnvelopeStatusGap || !hasReason(r.Reasons, ReasonEnvelopeUnderTarget) {
		t.Fatalf("review = %+v, want parsed CLI-path envelope gap", r)
	}
}
