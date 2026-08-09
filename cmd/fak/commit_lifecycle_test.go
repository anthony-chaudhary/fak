package main

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitlifecycle"
)

func TestCommitLifecycleQueueMapsCheckpointStatesToExecutableActions(t *testing.T) {
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("base line\npark me\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(context.Background(), dir, "parked-session", true, 1000); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if _, err := gitWipOut(context.Background(), dir, nil, "checkout", "--", "."); err != nil {
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
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("base line\nheld work\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "held-session", true, 1000); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "checkout", "--", "."); err != nil {
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
