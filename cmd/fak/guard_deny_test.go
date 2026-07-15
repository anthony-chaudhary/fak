package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func TestGuardDenyOverlayRoundTripAndMalformed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "guard", "deny.json")
	if got, err := loadGuardDenyOverlay(path); err != nil || len(got.Deny) != 0 {
		t.Fatalf("missing overlay = %+v, %v; want empty no-op", got, err)
	}
	if err := saveGuardDenyOverlay(path, guardDenyOverlay{Deny: []string{"Write", " Read ", "Read"}}); err != nil {
		t.Fatal(err)
	}
	got, err := loadGuardDenyOverlay(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got.Deny, ",") != "Read,Write" || got.Version != guardDenyOverlayVersion {
		t.Fatalf("round trip = %+v", got)
	}
	if err := os.WriteFile(path, []byte(`{"version":"wrong","deny":["Read"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadGuardDenyOverlay(path); err == nil {
		t.Fatal("malformed/unsupported overlay did not fail loud")
	}
}

func TestGuardDenyOverlayWinsOverAllow(t *testing.T) {
	rt := policy.Runtime{Adjudicator: adjudicator.Policy{Allow: map[string]bool{"Read": true}}}
	guardApplyAllowOverlay(&rt, guardAllowOverlay{Allow: []string{"custom"}})
	if added := guardApplyDenyOverlay(&rt, guardDenyOverlay{Deny: []string{"Read", "custom"}}); added != 2 {
		t.Fatalf("added = %d, want 2", added)
	}
	for _, tool := range []string{"Read", "custom"} {
		v := adjudicator.New(rt.Adjudicator).Adjudicate(context.Background(), guardToolCall(t, tool, map[string]any{}))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("%s = %v/%v, want DENY/POLICY_BLOCK", tool, v.Kind, v.Reason)
		}
	}
}

func TestGuardDenyOverlayAppliedAtLaunchAndReload(t *testing.T) {
	dir := t.TempDir()
	denyPath := filepath.Join(dir, "deny.json")
	allowPath := filepath.Join(dir, "allow.json")
	t.Setenv(guardDenyOverlayEnv, denyPath)
	t.Setenv(guardAllowOverlayEnv, allowPath)
	if err := saveGuardAllowOverlay(allowPath, guardAllowOverlay{Allow: []string{"Read"}}); err != nil {
		t.Fatal(err)
	}
	if err := saveGuardDenyOverlay(denyPath, guardDenyOverlay{Deny: []string{"Read"}}); err != nil {
		t.Fatal(err)
	}

	rt, source, _, _ := loadGuardCapabilityFloor("")
	if !strings.Contains(source, "repo-local deny overlay") {
		t.Fatalf("launch source = %q, want deny overlay provenance", source)
	}
	if got := rt.Adjudicator.Deny["Read"]; got != abi.ReasonPolicyBlock {
		t.Fatalf("launch deny = %v, want POLICY_BLOCK", got)
	}

	resp, err := guardPolicyReloader("")(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Reloaded {
		t.Fatalf("reload response = %+v", resp)
	}
	v := adjudicator.Default.Adjudicate(context.Background(), guardToolCall(t, "Read", map[string]any{"file_path": "README.md"}))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("reloaded Read = %v/%v, want DENY/POLICY_BLOCK", v.Kind, v.Reason)
	}
}

func TestPrintGuardDenyOverlayShowsPathAndEntries(t *testing.T) {
	var out strings.Builder
	printGuardDenyOverlay(&out, ".fak/guard/deny.json", guardDenyOverlay{Deny: []string{"terraform"}})
	got := out.String()
	if !strings.Contains(got, ".fak/guard/deny.json") || !strings.Contains(got, "terraform") {
		t.Fatalf("list output = %q", got)
	}
}
