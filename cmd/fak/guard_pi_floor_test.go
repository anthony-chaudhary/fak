package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// piBuiltinToolCalls are the first-turn calls pi sends under its own tool names
// and argument schema, as recorded in guarded pi Ops session journals.
var piBuiltinToolCalls = []struct {
	tool string
	args map[string]any
}{
	{"read", map[string]any{"path": "README.md"}},
	{"bash", map[string]any{"command": "go test ./..."}},
	{"powershell", map[string]any{"command": "Get-ChildItem"}},
	{"edit", map[string]any{"path": "notes.txt", "edits": []map[string]any{{"oldText": "a", "newText": "b"}}}},
	{"write", map[string]any{"path": "notes.txt", "content": "hello"}},
	{"grep", map[string]any{"pattern": "func"}},
	{"find", map[string]any{"pattern": "*.go"}},
	{"ls", map[string]any{"path": "."}},
}

func isolateGuardFloorOverlays(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv(guardAllowOverlayEnv, "")
	t.Setenv(guardDenyOverlayEnv, filepath.Join(t.TempDir(), "deny.json"))
	t.Setenv("FAK_GUARD_POSTURE", "")
	t.Setenv("FAK_PROFILE", "")
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(t.TempDir()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })
	prior := adjudicator.Default.PolicySnapshot()
	t.Cleanup(func() { adjudicator.Default.SetPolicy(prior) })
}

// fak-test:runtime fast est=1s
func TestGuardFloorAdmitsPiBuiltinTools(t *testing.T) {
	isolateGuardFloorOverlays(t)
	ctx := context.Background()
	for _, posture := range []string{"", "fail_closed"} {
		rt, _, _, _ := loadGuardCapabilityFloor("", posture)
		adj := adjudicator.New(rt.Adjudicator)
		for _, c := range piBuiltinToolCalls {
			v := adj.Adjudicate(ctx, guardToolCall(t, c.tool, c.args))
			if v.Kind != abi.VerdictAllow && v.Kind != abi.VerdictTransform {
				t.Errorf("posture %q: pi tool %s = %v/%s, want ALLOW or TRANSFORM", posture, c.tool, v.Kind, abi.ReasonName(v.Reason))
			}
		}
		danger := adj.Adjudicate(ctx, guardToolCall(t, "powershell", map[string]any{"command": `Remove-Item -Recurse -Force C:\work`}))
		if danger.Kind != abi.VerdictDeny || danger.Reason != abi.ReasonPolicyBlock {
			t.Errorf("posture %q: powershell recursive delete = %v/%s, want DENY/POLICY_BLOCK", posture, danger.Kind, abi.ReasonName(danger.Reason))
		}
	}
}

// fak-test:runtime fast est=1s
func TestDefaultPolicyAdmitsPiBuiltinTools(t *testing.T) {
	adj := adjudicator.New(adjudicator.DefaultPolicy())
	for _, c := range piBuiltinToolCalls {
		v := adj.Adjudicate(context.Background(), guardToolCall(t, c.tool, c.args))
		if v.Kind == abi.VerdictDeny && v.Reason == abi.ReasonDefaultDeny {
			t.Errorf("DefaultPolicy: pi tool %s = DENY/DEFAULT_DENY, want admitted", c.tool)
		}
	}
}
