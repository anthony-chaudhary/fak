package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

func TestScoreboardPostFlowFoldsLiveWorktreeStatus(t *testing.T) {
	root := t.TempDir()
	ledger := filepath.Join(root, "poison.jsonl")
	if err := os.WriteFile(ledger, []byte("{\"verdict\":\"TREE_POISONED\"}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(scoreboardPoisonLedgerEnv, ledger)

	oldRoot, oldList := scoreboardRepoRoot, scoreboardGitWorktreeList
	scoreboardRepoRoot = func() string { return root }
	scoreboardGitWorktreeList = func(gotRoot string) (string, error) {
		if gotRoot != root {
			t.Fatalf("git root = %q, want %q", gotRoot, root)
		}
		return "worktree /repo\n\nworktree /tmp/fak-worker-wt-a\n\nworktree C:\\tmp\\fak-worker-wt-b\n", nil
	}
	t.Cleanup(func() {
		scoreboardRepoRoot, scoreboardGitWorktreeList = oldRoot, oldList
	})

	var stdout, stderr bytes.Buffer
	code := scoreboardPostFlow(&stdout, &stderr, scoreboard.Update{Title: "fleet"}, scoreboardPostOpts{dryRun: true})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if got := stdout.String(); !strings.Contains(got, "2 isolated worktrees, 1 poison incidents") {
		t.Fatalf("render missing auditor status:\n%s", got)
	}
}
