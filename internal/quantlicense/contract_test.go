package quantlicense

import (
	"encoding/json"
	"os"
	"testing"
)

func readFixture(t *testing.T, name string) Manifest {
	t.Helper()
	raw, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	var m Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestLicenseFixtures(t *testing.T) {
	tests := []struct {
		name    string
		outcome Outcome
		reason  ReasonCode
	}{
		{"compatible.json", OutcomeAllow, ReasonCompatible},
		{"missing-artifact.json", OutcomeRefuse, ReasonMissingArtifactLicense},
		{"incompatible-redistribution.json", OutcomeRefuse, ReasonArtifactRedistributionDenied},
		{"incompatible-chain.json", OutcomeRefuse, ReasonIncompatibleChain},
		{"unknown-source.json", OutcomeAbstain, ReasonUnknownLicense},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Evaluate(readFixture(t, tc.name))
			if got.Outcome != tc.outcome || got.Reason != tc.reason {
				t.Fatalf("got %s/%s (%s), want %s/%s", got.Outcome, got.Reason, got.Action, tc.outcome, tc.reason)
			}
			if got.NonLegalAdvice == "" || got.Action == "" {
				t.Fatalf("result must carry non-legal-advice and actionable guidance: %#v", got)
			}
		})
	}
}

func TestTracksIndependentChainAndClaimEnvelope(t *testing.T) {
	m := readFixture(t, "compatible.json")
	got := Evaluate(m)
	if got.Outcome != OutcomeAllow {
		t.Fatalf("got %#v", got)
	}
	if m.SourceWeights.License.ID == m.Artifact.License.ID {
		t.Fatal("fixture must prove source and artifact licenses are tracked independently")
	}
	if m.Recipe.ID == "" || m.Recipe.Quantizer.Name == "" || m.Recipe.Quantizer.License.ID == "" || m.Runtime.Name == "" || m.Runtime.License.ID == "" {
		t.Fatalf("recipe/runtime contract incomplete: %#v", m)
	}
	if m.Claims.Artifact == "" || m.Claims.Recipe == "" || m.Claims.RuntimeDelegation == "" || m.Claims.HardwareEnvelope == "" {
		t.Fatalf("claim envelope must distinguish artifact, recipe, runtime delegation, and hardware: %#v", m.Claims)
	}
}

func TestUnknownSchemaAndUnsupportedRequestNeverFallback(t *testing.T) {
	m := readFixture(t, "compatible.json")
	m.Schema = "quantlicense/v99"
	got := Evaluate(m)
	if got.Outcome != OutcomeAbstain || got.Reason != ReasonUnknownSchema {
		t.Fatalf("got %#v", got)
	}

	m = readFixture(t, "compatible.json")
	m.Request.Use = UseKind("patent-sublicense")
	got = Evaluate(m)
	if got.Outcome != OutcomeDelegate || got.Reason != ReasonUseOutsideContract {
		t.Fatalf("got %#v", got)
	}
}

func TestMalformedJSONReturnsTypedAbstain(t *testing.T) {
	got, err := ParseAndEvaluate([]byte(`{"schema":`))
	if err == nil {
		t.Fatal("expected parse error")
	}
	if got.Outcome != OutcomeAbstain || got.Reason != ReasonInvalidJSON || got.Action == "" {
		t.Fatalf("got %#v", got)
	}
}
