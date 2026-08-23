package model

import (
	"encoding/json"
	"os"
	"testing"
)

func TestDiscoverTorchAOCapabilityFixtureMatrix(t *testing.T) {
	data, err := os.ReadFile("testdata/torchao_capabilities.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Source string `json:"source"`
		Cases  []struct {
			Name     string            `json:"name"`
			Request  json.RawMessage   `json:"request"`
			Decision TorchAODecision   `json:"decision"`
			Reason   TorchAOReasonCode `json:"reason"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Source == "" || len(fixture.Cases) < 3 {
		t.Fatalf("fixture must cite a public source and cover at least three cases")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			got := DiscoverTorchAOCapability(tc.Request)
			if got.Decision != tc.Decision || got.Reason != tc.Reason {
				t.Fatalf("got %s/%s, want %s/%s", got.Decision, got.Reason, tc.Decision, tc.Reason)
			}
			if got.Decision == TorchAODelegate {
				if got.Runtime == "" || got.MeasuredEnvelope != "" {
					t.Fatalf("delegated result must name runtime without inventing measured envelope: %+v", got)
				}
			}
		})
	}
}

func TestDiscoverTorchAOCapabilityRejectsMalformedInput(t *testing.T) {
	got := DiscoverTorchAOCapability([]byte(`{"version":`))
	if got.Decision != TorchAORefuse || got.Reason != TorchAOInvalidCapability {
		t.Fatalf("got %+v", got)
	}
}
