package issuepolicy

import "testing"

func TestProjectWorkProductionContractValid(t *testing.T) {
	c := envelopeCandidate()
	c.ParentRef = "#4636"
	c.WorkEstimate = "Estimate: 5 points (medium). Uncertainty: consumers."
	c.ScopeContribution = "Parent scope baseline: #4636 rollout, 47 points. Contribution: 5/47 points (10.6%)."
	c.CompletionStandard = "production"
	c.TargetEnvelope = "- concurrency: >= 1 user"
	c.WitnessedEnvelope = "- concurrency: 1 user"
	r := ReviewCandidate(c, Options{StrictProjectWork: true})
	if !r.OK || r.ProjectWork.Status != ProjectWorkValid || r.ProjectWork.EstimatePoints != 5 || r.ProjectWork.ContributionShare < 0.106 || !r.ProjectWork.ProductionCredit {
		t.Fatalf("review=%+v", r)
	}
}

func TestProjectWorkExplicitDemoHasZeroProductionCredit(t *testing.T) {
	c := envelopeCandidate()
	c.ParentRef = "#4636"
	c.WorkEstimate = "Estimate: 1 point (small)."
	c.ScopeContribution = "Contribution: 1/47 points."
	c.CompletionStandard = "demo"
	r := ReviewCandidate(c, Options{StrictProjectWork: true})
	if !r.OK || r.ProjectWork.Status != ProjectWorkValid || r.ProjectWork.ProductionCredit {
		t.Fatalf("review=%+v", r)
	}
}

func TestProjectWorkStrictMissingFailsWithStableReason(t *testing.T) {
	c := envelopeCandidate()
	r := ReviewCandidate(c, Options{StrictProjectWork: true})
	if r.OK || !hasReason(r.Reasons, ReasonProjectWorkMissing) || r.ProjectWork.Status != ProjectWorkUndeclared {
		t.Fatalf("review=%+v", r)
	}
}

func TestProjectWorkDenominatorAndEstimateMismatchFail(t *testing.T) {
	c := envelopeCandidate()
	c.ParentRef = "#4636"
	c.WorkEstimate = "Estimate: 5 points."
	c.ScopeContribution = "Contribution: 8/5 points."
	c.CompletionStandard = "production"
	c.TargetEnvelope = "- concurrency: >= 1 user"
	c.WitnessedEnvelope = "- concurrency: 1 user"
	r := ReviewCandidate(c, Options{StrictProjectWork: true})
	if r.OK || !hasReason(r.Reasons, ReasonProjectWorkInvalid) || len(r.ProjectWork.Invalid) < 2 {
		t.Fatalf("review=%+v", r)
	}
}

func TestProjectWorkInvalidCompletionStandardFails(t *testing.T) {
	c := envelopeCandidate()
	c.ParentRef = "#4636"
	c.WorkEstimate = "Estimate: 1 point."
	c.ScopeContribution = "Contribution: 1/47 points."
	c.CompletionStandard = "toy-ish maybe done"
	r := ReviewCandidate(c, Options{StrictProjectWork: true})
	if r.OK || !hasReason(r.Reasons, ReasonProjectWorkInvalid) {
		t.Fatalf("review=%+v", r)
	}
}
