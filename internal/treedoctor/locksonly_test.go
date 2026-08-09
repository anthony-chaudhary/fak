package treedoctor

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestLocksOnlySkipsWorktreeAndWIPWalks is the blast-radius witness for the UNATTENDED
// caller (`fak git-daily`). In LocksOnly mode the lock diagnoses still run — that is the
// whole point of the tick — but the worktree classification and the untracked-WIP
// inventory are not consulted at all, so Sweep's prune loop has nothing to iterate and
// CANNOT issue a `git worktree remove`. A false prune destroys a peer's in-flight work,
// so this property has to be a property of the CALL, not of a reviewer remembering which
// report fields Sweep acts on.
func TestLocksOnlySkipsWorktreeAndWIPWalks(t *testing.T) {
	root, gitDir := residueGitDir(t)
	now := time.Now()

	// Residue the lock half MUST still find: a renamed-aside index.lock (something after
	// `.lock`, so structurally never a lock git can be holding).
	residue := filepath.Join(gitDir, "index.lock"+StaleAsideSuffix+"20260716-044445")
	if err := os.WriteFile(residue, []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := now.Add(-13 * 24 * time.Hour)
	if err := os.Chtimes(residue, old, old); err != nil {
		t.Fatal(err)
	}

	var calls []string
	run := func(_ context.Context, _ string, args ...string) (string, int, error) {
		calls = append(calls, strings.Join(args, " "))
		switch {
		case len(args) >= 1 && args[0] == "worktree":
			// A merged, prunable worktree — offered so a regression that re-enables the
			// worktree half here has something to actually try to remove.
			return "worktree " + filepath.Join(root, "wt-merged") + "\nHEAD 0000000000000000000000000000000000000000\nbranch refs/heads/gone\n\n", 0, nil
		case len(args) >= 1 && args[0] == "ls-files":
			return "pkg/peer-wip.go\n", 0, nil
		}
		return "", 0, nil
	}

	rep, actions := Sweep(context.Background(), run, Options{
		RepoRoot:  root,
		Now:       now,
		LocksOnly: true,
	}, true)

	for _, c := range calls {
		if strings.HasPrefix(c, "worktree") {
			t.Fatalf("LocksOnly issued a worktree command (%q); the unattended tick must never prune a peer's worktree", c)
		}
	}
	if len(rep.Worktrees) != 0 {
		t.Fatalf("LocksOnly populated Worktrees (%d) — Sweep's prune loop would have something to remove", len(rep.Worktrees))
	}
	if len(rep.WIP) != 0 {
		t.Fatalf("LocksOnly populated the WIP inventory (%d files); that whole-tree walk should not be paid", len(rep.WIP))
	}

	// The lock half is untouched by the narrowing: the residue was still diagnosed and
	// swept. A LocksOnly mode that also skipped the locks would be a silent no-op tick.
	if len(rep.LockResidue) != 1 {
		t.Fatalf("LockResidue = %d, want the 1 aside-renamed index.lock", len(rep.LockResidue))
	}
	if len(actions) != 1 || !strings.Contains(actions[0], "orphan lock residue") {
		t.Fatalf("actions = %v, want the residue sweep", actions)
	}
	if _, err := os.Stat(residue); !os.IsNotExist(err) {
		t.Fatalf("residue survived the LocksOnly sweep: %v", err)
	}
}

// TestFullSweepStillSeesWorktrees is the negative control: the interactive
// `fak tree-doctor` path leaves LocksOnly false and still classifies worktrees, so the
// narrowing above is scoped to the unattended caller rather than a global regression.
func TestFullSweepStillSeesWorktrees(t *testing.T) {
	root, _ := residueGitDir(t)
	var sawWorktreeCall bool
	run := func(_ context.Context, _ string, args ...string) (string, int, error) {
		if len(args) >= 1 && args[0] == "worktree" {
			sawWorktreeCall = true
		}
		return "", 0, nil
	}
	Diagnose(context.Background(), run, Options{RepoRoot: root, Now: time.Now()})
	if !sawWorktreeCall {
		t.Fatal("full Diagnose never asked git about worktrees; LocksOnly leaked into the default path")
	}
}
