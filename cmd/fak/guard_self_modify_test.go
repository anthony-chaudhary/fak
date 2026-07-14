package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func TestGuardDefaultFloorProtectsAllowOverlay(t *testing.T) {
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, g := range rt.Adjudicator.SelfModifyGlobs {
		if g == ".fak/guard/" {
			found = true
		}
	}
	if !found {
		t.Fatalf("self_modify_globs=%v", rt.Adjudicator.SelfModifyGlobs)
	}
}

func TestProtectGuardPolicyConfigAddsResolvedPaths(t *testing.T) {
	overlay := filepath.Join(t.TempDir(), "custom", "allow.json")
	manifest := filepath.Join(t.TempDir(), "policy.json")
	rt := protectGuardPolicyConfig(policy.Runtime{}, overlay, manifest)
	got := map[string]bool{}
	for _, g := range rt.Adjudicator.SelfModifyGlobs {
		got[filepath.ToSlash(g)] = true
	}
	if !got[filepath.ToSlash(overlay)] || !got[filepath.ToSlash(manifest)] {
		t.Fatalf("globs=%v", rt.Adjudicator.SelfModifyGlobs)
	}
}

func TestReloadPolicyProtectsResolvedConfigurationPaths(t *testing.T) {
	overlay := filepath.Join(t.TempDir(), "custom", "allow.json")
	manifest := filepath.Join(t.TempDir(), "policy.json")
	t.Setenv("FAK_GUARD_ALLOW_OVERLAY", overlay)
	if err := os.WriteFile(manifest, guardDefaultPolicyJSON, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		adjudicator.Default.SetPolicy(adjudicator.DefaultPolicy())
	})

	rt, _, err := reloadPolicy(manifest)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, g := range rt.Adjudicator.SelfModifyGlobs {
		got[filepath.ToSlash(g)] = true
	}
	for _, want := range []string{overlay, manifest} {
		want = filepath.ToSlash(filepath.Clean(want))
		if !got[want] {
			t.Fatalf("self_modify_globs=%v, missing %q", rt.Adjudicator.SelfModifyGlobs, want)
		}
	}
}

func TestProtectedGuardPolicyConfigDeniesWritesButAllowsReads(t *testing.T) {
	overlay := filepath.ToSlash(filepath.Join(t.TempDir(), "custom", "allow.json"))
	manifest := filepath.ToSlash(filepath.Join(t.TempDir(), "policy.json"))
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	rt = protectGuardPolicyConfig(rt, overlay, manifest)
	a := adjudicator.New(rt.Adjudicator)

	for _, path := range []string{overlay, manifest} {
		t.Run(filepath.Base(path), func(t *testing.T) {
			write := guardToolCall(t, "Bash", map[string]any{"command": "echo x > " + path})
			if v := a.Adjudicate(context.Background(), write); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
				t.Fatalf("write verdict=(%v, %v), want (deny, SELF_MODIFY)", v.Kind, v.Reason)
			}

			read := guardToolCall(t, "Bash", map[string]any{"command": "cat " + path})
			if v := a.Adjudicate(context.Background(), read); v.Kind != abi.VerdictAllow {
				t.Fatalf("read verdict=(%v, %v), want allow", v.Kind, v.Reason)
			}
		})
	}
}

func guardToolCall(t *testing.T, tool string, args map[string]any) *abi.ToolCall {
	t.Helper()
	b, err := json.Marshal(args)
	if err != nil {
		t.Fatal(err)
	}
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: b}}
}
