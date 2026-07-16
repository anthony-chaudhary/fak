package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestGuardEffectivePolicyDigestEmptyOverlayPreservesBaseDigest(t *testing.T) {
	base := []byte(`{"version":"fak/policy/v1"}`)
	got := guardEffectivePolicyDigest(base, guardAllowOverlay{Version: guardAllowOverlayVersion}, guardDenyOverlay{})
	if want := guardPolicyDigest(base); got != want {
		t.Fatalf("empty overlay digest = %q, want historical base digest %q", got, want)
	}
}

func TestGuardEffectivePolicyDigestAttestsNormalizedOverlay(t *testing.T) {
	base := []byte(`{"version":"fak/policy/v1"}`)
	one := guardEffectivePolicyDigest(base, guardAllowOverlay{
		Allow:       []string{" beta ", "alpha", "alpha"},
		AllowPrefix: []string{"mcp__"},
	}, guardDenyOverlay{})
	equivalent := guardEffectivePolicyDigest(base, guardAllowOverlay{
		Version:     guardAllowOverlayVersion,
		Allow:       []string{"alpha", "beta"},
		AllowPrefix: []string{"mcp__"},
	}, guardDenyOverlay{})
	if one != equivalent {
		t.Fatalf("equivalent normalized overlays differ: %q != %q", one, equivalent)
	}
	different := guardEffectivePolicyDigest(base, guardAllowOverlay{Allow: []string{"gamma"}}, guardDenyOverlay{})
	if one == different || one == guardPolicyDigest(base) {
		t.Fatalf("effective digests do not distinguish overlays: one=%q different=%q base=%q", one, different, guardPolicyDigest(base))
	}
}

func TestLoadGuardCapabilityFloorAttestsEffectiveOverlay(t *testing.T) {
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	t.Setenv(guardAllowOverlayEnv, overlayPath)
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Allow: []string{"operator_tool"}}); err != nil {
		t.Fatal(err)
	}

	_, _, got, _ := loadGuardCapabilityFloor("")
	ov, err := loadGuardAllowOverlay(overlayPath)
	if err != nil {
		t.Fatal(err)
	}
	want := guardEffectivePolicyDigest(guardDefaultPolicyJSON, ov, guardDenyOverlay{})
	if got != want {
		t.Fatalf("spawn metadata source digest = %q, want effective digest %q", got, want)
	}
	if got == guardPolicyDigest(guardDefaultPolicyJSON) {
		t.Fatalf("load attested base-only digest %q despite non-empty overlay", got)
	}
}

func TestGuardPolicyReloaderReportsEffectiveDigest(t *testing.T) {
	t.Setenv(policyReloadWidenConfirmEnv, "1")
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	t.Setenv(guardAllowOverlayEnv, overlayPath)
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Allow: []string{"operator_tool"}}); err != nil {
		t.Fatal(err)
	}
	policyPath := filepath.Join(dir, "floor.json")
	if err := os.WriteFile(policyPath, guardDefaultPolicyJSON, 0o600); err != nil {
		t.Fatal(err)
	}

	resp, err := guardPolicyReloader(policyPath)(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ov, err := loadGuardAllowOverlay(overlayPath)
	if err != nil {
		t.Fatal(err)
	}
	want := guardEffectivePolicyDigest(guardDefaultPolicyJSON, ov, guardDenyOverlay{})
	if resp.EffectiveDigest != want {
		t.Fatalf("reload effective_digest = %q, want %q", resp.EffectiveDigest, want)
	}
	if resp.EffectiveDigest == guardPolicyDigest(guardDefaultPolicyJSON) {
		t.Fatalf("reload reported base-only digest %q despite non-empty overlay", resp.EffectiveDigest)
	}
}

func TestGuardEffectivePolicyDigestAttestsNormalizedDenyOverlay(t *testing.T) {
	base := []byte(`{"version":"fak/policy/v1"}`)
	one := guardEffectivePolicyDigest(base, guardAllowOverlay{}, guardDenyOverlay{Deny: []string{" beta ", "alpha", "alpha"}})
	equivalent := guardEffectivePolicyDigest(base, guardAllowOverlay{}, guardDenyOverlay{Version: guardDenyOverlayVersion, Deny: []string{"alpha", "beta"}})
	if one != equivalent {
		t.Fatalf("equivalent normalized deny overlays differ: %q != %q", one, equivalent)
	}
	different := guardEffectivePolicyDigest(base, guardAllowOverlay{}, guardDenyOverlay{Deny: []string{"gamma"}})
	if one == different || one == guardPolicyDigest(base) {
		t.Fatalf("deny digests do not distinguish overlays: one=%q different=%q base=%q", one, different, guardPolicyDigest(base))
	}
}

func TestGuardLaunchAndReloadAttestDenyOverlay(t *testing.T) {
	dir := t.TempDir()
	allowPath := filepath.Join(dir, "allow.json")
	denyPath := filepath.Join(dir, "deny.json")
	t.Setenv(guardAllowOverlayEnv, allowPath)
	t.Setenv(guardDenyOverlayEnv, denyPath)
	if err := saveGuardDenyOverlay(denyPath, guardDenyOverlay{Deny: []string{"Read"}}); err != nil {
		t.Fatal(err)
	}
	deny, err := loadGuardDenyOverlay(denyPath)
	if err != nil {
		t.Fatal(err)
	}
	want := guardEffectivePolicyDigest(guardDefaultPolicyJSON, guardAllowOverlay{}, deny)

	_, _, launchDigest, _ := loadGuardCapabilityFloor("")
	if launchDigest != want {
		t.Fatalf("launch digest = %q, want deny-aware %q", launchDigest, want)
	}
	resp, err := guardPolicyReloader("")(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if resp.EffectiveDigest != want {
		t.Fatalf("reload digest = %q, want deny-aware %q", resp.EffectiveDigest, want)
	}
}
