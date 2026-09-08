package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// TestWorkerWorktreeEnabledGrammar pins the portable default: unset is isolated,
// every explicit off-ish value selects shared-root compatibility, and any other
// explicit value keeps isolation enabled.
func TestWorkerWorktreeEnabledGrammar(t *testing.T) {
	off := []string{"", "0", "off", "OFF", "false", "False", "no", "disable", "disabled", "  off  "}
	on := []string{"1", "on", "true", "yes", "enable", "enabled", "worktree"}

	// Unset -> on.
	t.Run("unset", func(t *testing.T) {
		if !workerWorktreeEnabled() {
			t.Fatal("unset FLEET_WORKER_WORKTREE must be ON")
		}
	})
	for _, v := range off {
		v := v
		t.Run("off/"+v, func(t *testing.T) {
			t.Setenv("FLEET_WORKER_WORKTREE", v)
			if workerWorktreeEnabled() {
				t.Fatalf("value %q must be OFF", v)
			}
		})
	}
	for _, v := range on {
		v := v
		t.Run("on/"+v, func(t *testing.T) {
			t.Setenv("FLEET_WORKER_WORKTREE", v)
			if !workerWorktreeEnabled() {
				t.Fatalf("value %q must be ON", v)
			}
		})
	}
}

func TestDispatchTickWorktreePrepareFailureIsTerminal(t *testing.T) {
	original := prepareManagedWorkerWorktreeFunc
	t.Cleanup(func() { prepareManagedWorkerWorktreeFunc = original })

	var gotRoot string
	prepareManagedWorkerWorktreeFunc = func(root, lane, key, baseSHA, wtRoot string, git workerworktree.GitRunner) workerworktree.Result {
		gotRoot = root
		return workerworktree.Result{OK: false, Code: "PREPARE_REFUSED", Reason: "synthetic refusal"}
	}

	payload := map[string]any{}
	spawnCWD, _, ok := prepareDispatchWorkerWorktree("D:/portable-repo", "D:/portable-repo", "workerworktree", "12379", "base-sha", map[string]string{"KEEP": "1"}, payload)
	if ok {
		t.Fatal("prepare failure must refuse dispatch admission")
	}
	if spawnCWD != "D:/portable-repo" {
		t.Fatalf("spawn cwd = %q, want selected root retained only as non-launched evidence", spawnCWD)
	}
	if gotRoot != "D:/portable-repo" {
		t.Fatalf("prepare root = %q, want selected repository root", gotRoot)
	}
	if payload["worker_worktree_mode"] != worktreeModeManagedDefault {
		t.Fatalf("mode = %v, want %q", payload["worker_worktree_mode"], worktreeModeManagedDefault)
	}
	if payload["verdict"] != "WORKTREE_PREPARE_FAILED" || !strings.Contains(payload["reason"].(string), "synthetic refusal") {
		t.Fatalf("prepare refusal payload = %#v", payload)
	}
}

func TestDispatchTickWorktreeExplicitOptOutIsVisible(t *testing.T) {
	t.Setenv("FLEET_WORKER_WORKTREE", "off")
	payload := map[string]any{}
	root := "D:/portable-repo"
	spawnCWD, env, ok := prepareDispatchWorkerWorktree(root, root, "workerworktree", "12379", "base-sha", map[string]string{"KEEP": "1"}, payload)
	if !ok || spawnCWD != root || env["KEEP"] != "1" {
		t.Fatalf("explicit opt-out must preserve shared-root compatibility: cwd=%q env=%v ok=%v", spawnCWD, env, ok)
	}
	if payload["worker_worktree_mode"] != worktreeModeSharedExplicitOptOut || payload["worker_worktree_unsafe_shared"] != true {
		t.Fatalf("explicit opt-out is not visible in receipt: %#v", payload)
	}
}
