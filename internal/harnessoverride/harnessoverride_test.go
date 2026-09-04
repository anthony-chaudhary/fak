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

func TestProposeRefusesLockedCapabilities(t *testing.T) {
	lock := harnessresolve.Lock{
		ID: "sha256:test",
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "tool", ID: "network", Source: "layer:security", Locked: true},
		},
	}
	_, err := Propose(lock, Request{Capability: "tool:network", Value: "unrestricted"})
	if err == nil || !strings.Contains(err.Error(), "is locked by layer:security and cannot be overridden") {
		t.Fatalf("expected locked error, got: %v", err)
	}
}

func TestProposeValueOverrideAndRender(t *testing.T) {
	lock := harnessresolve.Lock{
		ID: "sha256:test",
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "tool", ID: "filesystem", Source: "manifest", Value: "chroot:/base"},
		},
	}
	propDefault, err := Propose(lock, Request{Capability: "tool:filesystem", Value: "chroot:/custom"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if propDefault.Layer.ID != "operator-override" {
		t.Fatalf("expected default layer ID operator-override, got %q", propDefault.Layer.ID)
	}
	if propDefault.Schema != Schema {
		t.Fatalf("expected schema %q, got %q", Schema, propDefault.Schema)
	}
	if len(propDefault.Selection) != 1 || propDefault.Selection[0] != "operator-override" {
		t.Fatalf("unexpected selection: %v", propDefault.Selection)
	}

	propCustom, err := Propose(lock, Request{Capability: "tool:filesystem", Value: "chroot:/custom", LayerID: "custom-layer"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if propCustom.Layer.ID != "custom-layer" {
		t.Fatalf("expected layer ID custom-layer, got %q", propCustom.Layer.ID)
	}

	rendered := Render(propCustom)
	for _, want := range []string{
		"HARNESS OVERRIDE | PROPOSAL",
		"current: sha256:test",
		"capability: tool:filesystem | from manifest",
		"layer: custom-layer | scope person",
		"change: replace | value chroot:/custom",
		"next:",
	} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("render output missing %q: %s", want, rendered)
		}
	}
}

func TestProposeRefusesMissingValue(t *testing.T) {
	lock := harnessresolve.Lock{
		ID: "sha256:test",
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "model", ID: "coder", Source: "manifest", Value: "qwen"},
		},
	}
	for _, val := range []string{"", "   "} {
		_, err := Propose(lock, Request{Capability: "model:coder", Value: val})
		if err == nil || !strings.Contains(err.Error(), "model override requires --value") {
			t.Fatalf("value %q: expected requires --value error, got: %v", val, err)
		}
	}
}
