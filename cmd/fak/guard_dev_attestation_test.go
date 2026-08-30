package main

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

func TestInstallGuardDevAttestationBindsLiveLaneAndFailsAfterRelease(t *testing.T) {
	oldDefault := adjudicator.Default
	adjudicator.Default = adjudicator.New(adjudicator.DevAgentPolicy())
	t.Cleanup(func() { adjudicator.Default = oldDefault })
	root := t.TempDir()
	wt := filepath.Join(root, "fak-worker-wt-issue-9850-abc")
	if err := os.MkdirAll(wt, 0o755); err != nil {
		t.Fatal(err)
	}
	ownerDir := filepath.Join(root, ".fak-worker-owners")
	if err := os.MkdirAll(ownerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ownerDir, filepath.Base(wt)+".json"), []byte(`{"schema":"fak-worker-worktree-owner/1","pid":`+strconv.Itoa(os.Getppid())+`,"lease_id":"issue-9850"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(wt)
	t.Setenv("DISPATCH_LANE", "issue-9850")
	t.Setenv("DISPATCH_ISSUE", "9850")
	t.Setenv("FLEET_WORKER_WORKTREE_DIR", wt)
	t.Setenv("FAK_REPO_ROOT", filepath.Join(root, "legacy-wrong-root"))
	oldWorkspace := guardDevWorkspaceForWorktree
	guardDevWorkspaceForWorktree = func(string) string { return root }
	t.Cleanup(func() { guardDevWorkspaceForWorktree = oldWorkspace })
	live := true
	oldLive := guardDevLeaseLive
	guardDevLeaseLive = func(context.Context, string) ([]byte, error) {
		if !live {
			return []byte(`[]`), nil
		}
		return []byte(`[{"lane":"issue-9850","holder":"worker-9850","pid":123,"mode":"exclusive","tree":["internal/adjudicator/dev_attestation.go"]}]`), nil
	}
	t.Cleanup(func() { guardDevLeaseLive = oldLive })
	if err := installGuardDevAttestation("trace-9850", ""); err != nil {
		t.Fatal(err)
	}
	call := guardToolCall(t, "write_file", map[string]any{"path": "internal/adjudicator/dev_attestation.go"})
	call.TraceID = "trace-9850"
	if v := adjudicator.Default.Adjudicate(context.Background(), call); v.Kind != abi.VerdictAllow {
		t.Fatalf("live matching lease: got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	live = false
	if v := adjudicator.Default.Adjudicate(context.Background(), call); v.Reason != abi.ReasonSelfModify {
		t.Fatalf("released lease: got %v/%s, want SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestInstallGuardDevAttestationRejectsMismatchedWorktreeAndOwner(t *testing.T) {
	oldDefault := adjudicator.Default
	adjudicator.Default = adjudicator.New(adjudicator.DevAgentPolicy())
	t.Cleanup(func() { adjudicator.Default = oldDefault })
	root := t.TempDir()
	wt := filepath.Join(root, "fak-worker-wt-issue-9850-abc")
	if err := os.MkdirAll(wt, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("DISPATCH_LANE", "issue-9850")
	t.Setenv("DISPATCH_ISSUE", "9850")
	t.Setenv("FLEET_WORKER_WORKTREE_DIR", wt)
	t.Setenv("FAK_REPO_ROOT", filepath.Join(root, "legacy-wrong-root"))
	oldWorkspace := guardDevWorkspaceForWorktree
	guardDevWorkspaceForWorktree = func(string) string { return root }
	t.Cleanup(func() { guardDevWorkspaceForWorktree = oldWorkspace })
	// CWD mismatch is rejected before an attacker-controlled environment can reach DOS.
	if err := installGuardDevAttestation("trace-9850", ""); err == nil {
		t.Fatal("expected worktree mismatch refusal")
	}
	t.Chdir(wt)
	ownerDir := filepath.Join(root, ".fak-worker-owners")
	if err := os.MkdirAll(ownerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ownerDir, filepath.Base(wt)+".json"), []byte(`{"schema":"fak-worker-worktree-owner/1","pid":`+strconv.Itoa(os.Getppid())+`,"lease_id":"different"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := installGuardDevAttestation("trace-9850", ""); err == nil {
		t.Fatal("expected owner/lane mismatch refusal")
	}
}
