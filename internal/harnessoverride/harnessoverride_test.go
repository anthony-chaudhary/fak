package harnessoverride

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func TestProposeRefusesUnknownAndMandatoryCapabilities(t *testing.T) {
	lock := harnessresolve.Lock{ID: "sha256:test", Assets: []harnesscompose.EffectiveAsset{{Kind: "workflow", ID: "audit", Source: "company", Mandatory: true}}}
	for _, tc := range []struct {
		capability string
		want       string
	}{
		{"opaque-label", "must be kind:id from harness inspect"},
		{"tool:missing", "is not active in the verified lock"},
		{"workflow:audit", "is mandatory and cannot be overridden"},
	} {
		_, err := Propose(lock, Request{Capability: tc.capability, Value: "off"})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%s error=%v, want %q", tc.capability, err, tc.want)
		}
	}
}

func TestProposePolicyOnlyNarrows(t *testing.T) {
	lock := harnessresolve.Lock{ID: "sha256:test", Assets: []harnesscompose.EffectiveAsset{{Kind: "policy", ID: "tools", Source: "team", Grants: []string{"search", "shell"}}}}
	proposal, err := Propose(lock, Request{Capability: "policy:tools", Denies: []string{"shell", "shell"}})
	if err != nil {
		t.Fatal(err)
	}
	asset := proposal.Layer.Assets[0]
	if asset.Operation != "add" || len(asset.Grants) != 0 || len(asset.Denies) != 1 || asset.Denies[0] != "shell" {
		t.Fatalf("asset=%+v", asset)
	}
	if _, err := Propose(lock, Request{Capability: "policy:tools"}); err == nil || !strings.Contains(err.Error(), "requires at least one --deny") {
		t.Fatalf("error=%v", err)
	}
}
