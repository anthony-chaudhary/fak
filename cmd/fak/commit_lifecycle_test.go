package main

import (
	"context"
	"os"
	"testing"

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
	if got.State != commitlifecycle.LandReady || commitLifecycleActionText(got.Action) != "fak wip land parked-session --apply" {
		t.Fatalf("row = %+v, want LAND_READY with executable land action", got)
	}
}

func TestCommitLifecycleActionTextPreservesOperatorGate(t *testing.T) {
	got := commitLifecycleActionText(commitlifecycle.Action{NeedsOperator: true, Reason: "conflict needs review"})
	if got != "operator: conflict needs review" {
		t.Fatalf("action text = %q", got)
	}
}
