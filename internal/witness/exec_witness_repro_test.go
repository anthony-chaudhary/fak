// Package witness symptom witness for the exec-witness scratch leak.
//
// This file is deliberately written against symbols that already exist at the
// parent commit (NewExecutionVerifier + scratchWorktree) so it compiles before
// and after the fix. At the parent commit nothing sweeps orphaned scratch
// directories, so the planted legacy dir survives and the test fails; after the
// fix scratchWorktree sweeps its parent before creating its own dir and the
// planted dir is collected.
package witness

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestExecWitnessScratchCollectsKilledOwnerBeforeCreate plants a stale,
// ownerless scratch directory beside a scratch-worktree repo root and asserts
// that a scratch checkout sweeps it before creating its own directory.
//
// The fixture repo commits a go.work that references a sibling module so the
// checkout takes the sibling-parent path: scratchWorktree materializes beside
// the repository root, i.e. inside the parent directory holding the planted dir.
func TestExecWitnessScratchCollectsKilledOwnerBeforeCreate(t *testing.T) {
	ctx := context.Background()

	root := newExecutionRepo(t)
	// A committed workspace with a `../x` use directive forces scratchWorktree
	// to place its checkout beside the repo root (parent = filepath.Dir(root)).
	writeRepoFile(t, root, "go.work", "go 1.21\n\nuse (\n\t.\n\t../x\n)\n")
	gitIn(t, root, "add", "go.work")
	gitIn(t, root, "commit", "-q", "-m", "sibling workspace")

	rootAbs, err := filepath.Abs(root)
	if err != nil {
		t.Fatalf("abs root: %v", err)
	}
	parent := filepath.Dir(rootAbs)

	// A legacy, ownerless scratch dir (bare MkdirTemp-style suffix) killed long
	// enough ago to be classified stale by the sweep's legacy-age rule.
	planted := filepath.Join(parent, "fak-exec-witness-999999")
	if err := os.Mkdir(planted, 0o755); err != nil {
		t.Fatalf("mkdir planted: %v", err)
	}
	old := time.Now().Add(-3 * time.Hour)
	if err := os.Chtimes(planted, old, old); err != nil {
		t.Fatalf("chtimes planted: %v", err)
	}

	v := NewExecutionVerifier(root)
	scratchDir, cleanup, err := v.scratchWorktree(ctx, "HEAD")
	if err != nil {
		t.Fatalf("scratchWorktree: %v", err)
	}
	if cleanup != nil {
		cleanup()
	}
	if scratchDir == "" {
		t.Fatal("scratchWorktree returned an empty directory")
	}

	if _, statErr := os.Stat(planted); !os.IsNotExist(statErr) {
		t.Fatalf("orphaned scratch dir %s survived the checkout sweep (stat err=%v)", planted, statErr)
	}
}
