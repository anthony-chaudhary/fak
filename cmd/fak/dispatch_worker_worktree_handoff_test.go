package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestDispatchWorktreeOwnerHandoffProtectsSpawnedWorker(t *testing.T) {
	parent := t.TempDir()
	wt := filepath.Join(parent, workerworktree.WorktreeMarker+"-spawn")
	if err := os.Mkdir(wt, 0o755); err != nil {
		t.Fatal(err)
	}
	stamp := workerworktree.OwnerStamp{Schema: "fak-worker-worktree-owner/1", PID: 101, LeaseID: "lease", CreatedAt: time.Now().UTC()}
	raw, err := json.Marshal(stamp)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(workerworktree.OwnerStampPath(wt)), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(workerworktree.OwnerStampPath(wt), raw, 0o600); err != nil {
		t.Fatal(err)
	}

	payload := map[string]any{"worker_worktree": wt}
	if err := handoffDispatchWorktreeOwner(payload, 202); err != nil {
		t.Fatal(err)
	}
	updated, err := os.ReadFile(workerworktree.OwnerStampPath(wt))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(updated, &stamp); err != nil {
		t.Fatal(err)
	}
	if stamp.PID != 202 || payload["worker_owner_pid"] != 202 {
		t.Fatalf("stamp=%+v payload=%v, want spawned pid ownership", stamp, payload)
	}
}
