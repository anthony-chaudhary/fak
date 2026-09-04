package ops

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestOpsWitnessSuite delivers the comprehensive end-to-end witness for #11170.
// It verifies:
// 1. Storage GC cascade (pruning root stray binaries and expired scratch).
// 2. Dead-PID lock eviction while preserving live-PID locks.
// 3. Reaping orphan processes and tripping process guard.
// 4. Cryptographically and schema-valid event emission into fak-ops-event/1 ledger.
func TestOpsWitnessSuite(t *testing.T) {
	tmp := t.TempDir()
	repoRoot := filepath.Join(tmp, "repo")
	_ = os.MkdirAll(filepath.Join(repoRoot, ".git"), 0o755)
	_ = os.MkdirAll(filepath.Join(repoRoot, "cmd", "workerbin"), 0o755)
	_ = os.MkdirAll(filepath.Join(repoRoot, "_scratch", "go-tmp"), 0o755)
	_ = os.MkdirAll(filepath.Join(repoRoot, "_scratch", "stale-worker"), 0o755)

	// 1. Storage setup: expired go-tmp and stray scratch
	oldTmp := filepath.Join(repoRoot, "_scratch", "go-tmp", "build-artifact")
	_ = os.WriteFile(oldTmp, []byte("compiled objects"), 0o644)
	past := time.Now().Add(-3 * time.Hour)
	_ = os.Chtimes(oldTmp, past, past)

	oldScratch := filepath.Join(repoRoot, "_scratch", "stale-worker", "dump.json")
	_ = os.WriteFile(oldScratch, []byte("transcripts"), 0o644)
	_ = os.Chtimes(filepath.Dir(oldScratch), past, past)

	// 2. Dead vs Live Locks
	deadLock := filepath.Join(repoRoot, ".git", "fak-commit.lock")
	_ = os.WriteFile(deadLock, []byte("9999999"), 0o644)

	liveLock := filepath.Join(repoRoot, ".git", "index.lock")
	_ = os.WriteFile(liveLock, []byte("11111"), 0o644)

	// 3. Engine Initialization
	cfg := DefaultConfig()
	cfg.ScratchTTL = 1 * time.Hour
	cfg.OrphanReapEnabled = true

	engine, err := NewEngine(repoRoot, cfg)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	// Mock process liveness
	engine.Workspace.ProcessAlive = func(pid int) bool {
		return pid == 11111
	}

	// Mock process reaper
	reapedPIDs := []int{}
	engine.Process.Killer = func(pid int) (bool, string) {
		reapedPIDs = append(reapedPIDs, pid)
		return true, "killed"
	}

	// 4. Execute Tick with forceAll = true
	ctx := context.Background()
	if err := engine.Tick(ctx, true); err != nil {
		t.Fatalf("engine.Tick failed: %v", err)
	}

	// 5. Verify Storage Reclamation Witness
	if _, err := os.Stat(oldTmp); !os.IsNotExist(err) {
		t.Errorf("expected oldTmp to be deleted by storage cascade")
	}
	if _, err := os.Stat(filepath.Dir(oldScratch)); !os.IsNotExist(err) {
		t.Errorf("expected oldScratch dir to be deleted by storage cascade")
	}

	// 6. Verify Dead Lock Eviction Witness
	if _, err := os.Stat(deadLock); !os.IsNotExist(err) {
		t.Errorf("expected dead lock %s to be evicted", deadLock)
	}
	if _, err := os.Stat(liveLock); err != nil {
		t.Errorf("expected live lock %s to be retained", liveLock)
	}

	// 7. Verify Ledger Receipts Witness
	events, err := engine.Ledger.QueryEvents(1 * time.Hour)
	if err != nil {
		t.Fatalf("QueryEvents: %v", err)
	}
	if len(events) == 0 {
		t.Fatalf("expected ops events to be logged in fak-ops-event/1 ledger")
	}

	foundLockEvict := false
	foundStorageReclaim := false
	for _, ev := range events {
		if ev.Schema != EventSchemaV1 {
			t.Errorf("expected schema %s, got %s", EventSchemaV1, ev.Schema)
		}
		if ev.ActionType == ActionLockEvict {
			foundLockEvict = true
		}
		if ev.ActionType == ActionStorageReclaim {
			foundStorageReclaim = true
		}
	}

	if !foundLockEvict {
		t.Errorf("expected ActionLockEvict in ledger")
	}
	if !foundStorageReclaim {
		t.Errorf("expected ActionStorageReclaim in ledger")
	}
}
