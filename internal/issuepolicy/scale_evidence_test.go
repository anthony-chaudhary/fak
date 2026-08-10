package issuepolicy

import "testing"

func TestScaleEvidenceToyAndModeledDoNotSatisfyProductionTarget(t *testing.T) {
	c := envelopeCandidate()
	c.CompletionStandard = "production"
	c.TargetEnvelope = "- concurrency: >= 1000 requests\n- duration: >= 60 minutes"
	c.RequiredScaleStages = "toy, target-load, recovery"
	c.ScaleEvidence = "- toy; witnessed; concurrency: 1 requests, duration: 1 minutes; environment=dev\n- target-load; modeled; concurrency: 1000 requests, duration: 60 minutes; workload=synthetic\n- recovery; witnessed; concurrency: 1 requests, duration: 1 minutes"
	r := ReviewCandidate(c, Options{})
	if r.OK || !hasReason(r.Reasons, ReasonEnvelopeUnderTarget) {
		t.Fatalf("review=%+v, modeled target load must not become production witness", r)
	}
	if len(r.ScaleEvidence.MissingStages) != 0 || len(r.ScaleEvidence.Records) != 3 {
		t.Fatalf("scale evidence=%+v, want all stages present but envelope under target", r.ScaleEvidence)
	}
}

func TestScaleEvidenceDirectTargetLoadAndRecoverySatisfyProduction(t *testing.T) {
	c := envelopeCandidate()
	c.CompletionStandard = "production"
	c.TargetEnvelope = "- concurrency: >= 1000 requests\n- duration: >= 60 minutes"
	c.RequiredScaleStages = "target-load, recovery"
	c.ScaleEvidence = "- target-load; witnessed; concurrency: 1000 requests, duration: 60 minutes; environment=staging; workload=representative\n- recovery; observed; concurrency: 1000 requests, duration: 60 minutes"
	r := ReviewCandidate(c, Options{})
	if !r.OK || r.OperatingEnvelope.Status != EnvelopeMet {
		t.Fatalf("review=%+v, direct target/recovery evidence should meet envelope", r)
	}
}

func TestScaleEvidenceMissingCriticalStageFailsClosed(t *testing.T) {
	c := envelopeCandidate()
	c.CompletionStandard = "production"
	c.TargetEnvelope = "- concurrency: >= 1 user"
	c.RequiredScaleStages = "target-load, soak, recovery"
	c.ScaleEvidence = "- target-load; witnessed; concurrency: 1 user\n- recovery; witnessed; concurrency: 1 user"
	r := ReviewCandidate(c, Options{})
	if r.OK || !hasReason(r.Reasons, ReasonScaleStageMissing) || len(r.ScaleEvidence.MissingStages) != 1 || r.ScaleEvidence.MissingStages[0] != "soak" {
		t.Fatalf("review=%+v, want missing soak stage", r)
	}
}

func TestScaleEvidenceSmallScopeNeedsOnlyDeclaredRelevantStage(t *testing.T) {
	c := envelopeCandidate()
	c.CompletionStandard = "production"
	c.TargetEnvelope = "- concurrency: >= 1 user"
	c.RequiredScaleStages = "target-load"
	c.ScaleEvidence = "- target-load; witnessed; concurrency: 1 user; environment=local"
	r := ReviewCandidate(c, Options{})
	if !r.OK || r.OperatingEnvelope.Status != EnvelopeMet {
		t.Fatalf("review=%+v, small target should not require artificial thousand-unit stages", r)
	}
}

func TestScaleEvidenceMalformedRecordIsTyped(t *testing.T) {
	c := envelopeCandidate()
	c.CompletionStandard = "production"
	c.TargetEnvelope = "- concurrency: >= 1 user"
	c.ScaleEvidence = "- magic; trusted-ish; concurrency: 1 user"
	r := ReviewCandidate(c, Options{})
	if r.OK || !hasReason(r.Reasons, ReasonScaleEvidenceInvalid) || len(r.ScaleEvidence.Invalid) == 0 {
		t.Fatalf("review=%+v, want invalid evidence reason", r)
	}
}
