package quantpolicy

import (
	"encoding/json"
	"os"
	"testing"
)

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestPolicyFixturesAreStructural(t *testing.T) {
	policy := fixture(t, "policy.json")
	tests := []struct {
		name    string
		outcome Outcome
		reason  ReasonCode
		gate    PredicateID
	}{
		{"allow.json", OutcomeAllow, ReasonSatisfied, PredicateAll},
		{"below-minimum.json", OutcomeRefuse, ReasonBelowMinimumPrecision, PredicateMinimumPrecision},
		{"above-maximum.json", OutcomeRefuse, ReasonAboveMaximumPrecision, PredicateMaximumPrecision},
		{"unapproved-format.json", OutcomeRefuse, ReasonFormatNotApproved, PredicateApprovedFormat},
		{"conversion-refused.json", OutcomeRefuse, ReasonConversionRefused, PredicateConversion},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseAndEvaluate(policy, fixture(t, tc.name))
			if err != nil {
				t.Fatalf("ParseAndEvaluate: %v", err)
			}
			if got.Outcome != tc.outcome || got.Reason != tc.reason || got.Predicate != tc.gate {
				t.Fatalf("got %s/%s at %s (%s), want %s/%s at %s", got.Outcome, got.Reason, got.Predicate, got.Detail, tc.outcome, tc.reason, tc.gate)
			}
			if got.Action == "" {
				t.Fatalf("decision has no next action: %#v", got)
			}
		})
	}
}

func TestPrecisionBoundsAreInclusive(t *testing.T) {
	policy, request := typedFixtures(t)
	for _, bits := range []float64{policy.MinPrecisionBits, policy.MaxPrecisionBits} {
		request.Metadata.Precision.Bits = bits
		if got := Evaluate(policy, request); got.Outcome != OutcomeAllow {
			t.Fatalf("bits=%g got %#v", bits, got)
		}
	}
}

func TestUnknownMetadataFailsClosed(t *testing.T) {
	policyRaw := fixture(t, "policy.json")
	requestRaw := fixture(t, "allow.json")
	var request map[string]any
	if err := json.Unmarshal(requestRaw, &request); err != nil {
		t.Fatal(err)
	}
	metadata := request["metadata"].(map[string]any)
	metadata["future_quantizer_hint"] = "silently-ignore-me"
	requestRaw, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ParseAndEvaluate(policyRaw, requestRaw)
	if err == nil {
		t.Fatal("unknown metadata field was silently accepted")
	}
	if got.Outcome != OutcomeAbstain || got.Reason != ReasonUnknownMetadata || got.Predicate != PredicateMetadata {
		t.Fatalf("got %#v, err=%v", got, err)
	}

	policy, typed := typedFixtures(t)
	typed.Metadata.Precision = Precision{}
	got = Evaluate(policy, typed)
	if got.Outcome != OutcomeAbstain || got.Reason != ReasonUnknownPrecision {
		t.Fatalf("missing precision did not fail closed: %#v", got)
	}

	policy, typed = typedFixtures(t)
	typed.Metadata.Provenance.Recipe.Kind = EvidenceKind("future-attestation")
	got = Evaluate(policy, typed)
	if got.Outcome != OutcomeAbstain || got.Reason != ReasonUnknownProvenance {
		t.Fatalf("unknown provenance did not fail closed: %#v", got)
	}
}

func TestUnknownAndUnsupportedInputsAreTyped(t *testing.T) {
	policy, request := typedFixtures(t)
	policy.Schema = "quantpolicy.policy/v99"
	if got := Evaluate(policy, request); got.Outcome != OutcomeAbstain || got.Reason != ReasonUnknownSchema {
		t.Fatalf("unknown policy schema: %#v", got)
	}

	policy, request = typedFixtures(t)
	request.Metadata.Schema = "quantpolicy.metadata/v99"
	if got := Evaluate(policy, request); got.Outcome != OutcomeAbstain || got.Reason != ReasonUnknownMetadataSchema {
		t.Fatalf("unknown metadata schema: %#v", got)
	}

	policy, request = typedFixtures(t)
	request.Metadata.Format.Version = "99"
	if got := Evaluate(policy, request); got.Outcome != OutcomeRefuse || got.Reason != ReasonFormatNotApproved || got.Predicate != PredicateApprovedFormat {
		t.Fatalf("unapproved format version: %#v", got)
	}

	policy, request = typedFixtures(t)
	request.Operation = Operation("transcode-in-place")
	if got := Evaluate(policy, request); got.Outcome != OutcomeDelegate || got.Reason != ReasonOperationNotHandled {
		t.Fatalf("unsupported operation: %#v", got)
	}
}

func TestTypedOutcomesPreserveSeparatedClaims(t *testing.T) {
	tests := []struct {
		name    string
		outcome Outcome
		mutate  func(*Policy, *Request)
	}{
		{
			name:    "refuse",
			outcome: OutcomeRefuse,
			mutate: func(policy *Policy, request *Request) {
				request.Metadata.Precision.Bits = policy.MinPrecisionBits - 1
			},
		},
		{
			name:    "abstain",
			outcome: OutcomeAbstain,
			mutate: func(_ *Policy, request *Request) {
				request.Metadata.Schema = "quantpolicy.metadata/v99"
			},
		},
		{
			name:    "delegate",
			outcome: OutcomeDelegate,
			mutate: func(_ *Policy, request *Request) {
				request.Operation = Operation("runtime-owned-operation")
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			policy, request := typedFixtures(t)
			tc.mutate(&policy, &request)
			wantClaims := request.Metadata.Provenance
			got := Evaluate(policy, request)
			if got.Outcome != tc.outcome {
				t.Fatalf("got outcome %q, want %q: %#v", got.Outcome, tc.outcome, got)
			}
			if got.Claims != wantClaims {
				t.Fatalf("typed %s outcome changed or collapsed claims:\n got: %#v\nwant: %#v", tc.outcome, got.Claims, wantClaims)
			}
		})
	}
}

func TestProvenanceRequirementsAndClaimEnvelope(t *testing.T) {
	policy, request := typedFixtures(t)
	request.Metadata.Provenance.Artifact = Evidence{}
	if got := Evaluate(policy, request); got.Outcome != OutcomeRefuse || got.Reason != ReasonProvenanceRequired || got.Predicate != PredicateArtifactProvenance {
		t.Fatalf("missing artifact provenance: %#v", got)
	}

	policy, request = typedFixtures(t)
	request.Metadata.Provenance.HardwareEnvelope.Kind = EvidenceObserved
	if got := Evaluate(policy, request); got.Outcome != OutcomeRefuse || got.Reason != ReasonProvenanceNotApproved || got.Predicate != PredicateHardwareProvenance {
		t.Fatalf("unapproved hardware provenance: %#v", got)
	}

	policy, request = typedFixtures(t)
	got := Evaluate(policy, request)
	if got.Outcome != OutcomeAllow {
		t.Fatalf("allow fixture: %#v", got)
	}
	if got.Claims.Artifact.Reference == "" || got.Claims.Recipe.Reference == "" || got.Claims.RuntimeDelegation.Reference == "" || got.Claims.HardwareEnvelope.Reference == "" {
		t.Fatalf("result does not keep artifact, recipe, runtime delegation, and hardware claims separate: %#v", got.Claims)
	}
	if got.Claims.HardwareEnvelope.Kind != EvidenceMeasured {
		t.Fatalf("hardware envelope is not explicitly measured: %#v", got.Claims.HardwareEnvelope)
	}
}

func TestMalformedAndInvalidPolicyReturnTypedResults(t *testing.T) {
	got, err := ParseAndEvaluate([]byte(`{"schema":`), fixture(t, "allow.json"))
	if err == nil || got.Outcome != OutcomeRefuse || got.Reason != ReasonInvalidContract {
		t.Fatalf("malformed policy got %#v err=%v", got, err)
	}

	policy, request := typedFixtures(t)
	policy.MinPrecisionBits = policy.MaxPrecisionBits + 1
	got = Evaluate(policy, request)
	if got.Outcome != OutcomeRefuse || got.Reason != ReasonInvalidContract {
		t.Fatalf("invalid policy got %#v", got)
	}
}

func typedFixtures(t *testing.T) (Policy, Request) {
	t.Helper()
	var policy Policy
	if err := json.Unmarshal(fixture(t, "policy.json"), &policy); err != nil {
		t.Fatal(err)
	}
	var request Request
	if err := json.Unmarshal(fixture(t, "allow.json"), &request); err != nil {
		t.Fatal(err)
	}
	return policy, request
}
