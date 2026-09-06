package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"

	"github.com/anthony-chaudhary/fak/internal/commitlifecycle"
)

func TestCommitLifecycleQueueMapsCheckpointStatesToExecutableActions(t *testing.T) {
	dir, _ := wipTestRepo(t)
	fresh := filepath.Join(dir, "fresh.txt")
	if err := os.WriteFile(fresh, []byte("base line\npark me\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(context.Background(), dir, "parked-session", true, 1000); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if err := os.Remove(fresh); err != nil {
		t.Fatal(err)
	}

	rows, err := commitLifecycleQueue(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) < 1 {
		t.Fatalf("rows = %+v, want checkpoint row", rows)
	}
	got := rows[0]
	// The named command must be one that EXISTS: an unclaimed RECLAIM checkpoint is
	// advanced by adopting it (#5998), which claims it first so a second reader of this
	// same queue cannot start the same recovery.
	if got.State != commitlifecycle.LandReady || commitLifecycleActionText(got.Action) != "fak wip reconcile adopt parked-session" {
		t.Fatalf("row = %+v, want LAND_READY with the executable adoption action", got)
	}
}

// TestCommitLifecycleQueueParksACheckpointAnotherSuccessorHolds is the queue half of the
// #5998 done-condition: once a peer's live claim exists, this row stops being offered as
// work and becomes an operator gate naming the holder. A queue that kept printing "adopt"
// here is precisely how one checkpoint gets recovered twice.
func TestCommitLifecycleQueueParksACheckpointAnotherSuccessorHolds(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)
	fresh := filepath.Join(dir, "fresh.txt")
	if err := os.WriteFile(fresh, []byte("base line\nheld work\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "held-session", true, 1000); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if err := os.Remove(fresh); err != nil {
		t.Fatal(err)
	}
	if _, _, err := wipAdoptRun(ctx, dir, wipAdoptOptions{
		Session: "held-session", Successor: "peer-successor", Now: time.Now(),
	}); err != nil {
		t.Fatalf("peer adoption: %v", err)
	}

	rows, err := commitLifecycleQueue(ctx, dir)
	if err != nil {
		t.Fatal(err)
	}
	got := rows[0]
	if got.State != commitlifecycle.Parked || !got.Action.NeedsOperator {
		t.Fatalf("row = %+v, want a PARKED operator gate while a peer holds the claim", got)
	}
	if !strings.Contains(got.Action.Reason, "peer-successor") {
		t.Fatalf("gate must name the holder, got %q", got.Action.Reason)
	}
}

func TestCommitLifecycleActionTextPreservesOperatorGate(t *testing.T) {
	got := commitLifecycleActionText(commitlifecycle.Action{NeedsOperator: true, Reason: "conflict needs review"})
	if got != "operator: conflict needs review" {
		t.Fatalf("action text = %q", got)
	}
}

func TestCommitLifecycleQueueIncludesExactWorkerLandAction(t *testing.T) {
	dir, _ := wipTestRepo(t)
	baseOut, err := gitWipOut(context.Background(), dir, nil, "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	base := strings.TrimSpace(baseOut)
	prep := workerworktree.Prepare(dir, "cmd", "5994", base, filepath.Join(t.TempDir(), "workers"), nil)
	if !prep.OK {
		t.Fatalf("prepare = %+v", prep)
	}
	t.Cleanup(func() { _ = workerworktree.Reap(dir, prep.Path, nil) })
	message := "feat(workerworktree): queue detached land (#5994) (fak workerworktree)"
	if err := workerworktree.SaveIntent(prep.Path, base, message, []string{"note.txt"}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(prep.Path, "note.txt"), []byte("worker edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rows, err := commitLifecycleQueue(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	var got *commitlifecycle.Row
	for i := range rows {
		if rows[i].State == commitlifecycle.LandReady && len(rows[i].Action.Args) > 2 && rows[i].Action.Args[0] == "worktree" {
			got = &rows[i]
			break
		}
	}
	if got == nil {
		t.Fatalf("rows = %+v, want worker LAND_READY", rows)
	}
	text := commitLifecycleActionText(got.Action)
	for _, want := range []string{"fak worktree worker land", "--worktree " + prep.Path, "--base-sha " + base, "--msg-file", "--paths note.txt"} {
		if !strings.Contains(text, want) {
			t.Fatalf("action %q lacks %q", text, want)
		}
	}
}
