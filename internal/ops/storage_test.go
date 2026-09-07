package ops

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestStorageManagerReclaimCascade(t *testing.T) {
	tmp := t.TempDir()

	// Setup fake repo root
	repoRoot := filepath.Join(tmp, "repo")
	_ = os.MkdirAll(filepath.Join(repoRoot, "cmd", "testbin"), 0o755)
	_ = os.MkdirAll(filepath.Join(repoRoot, "_scratch", "go-tmp"), 0o755)
	_ = os.MkdirAll(filepath.Join(repoRoot, "_scratch", "old-producer"), 0o755)

	// Create a dummy compiler temp older than 2 hours
	oldTmp := filepath.Join(repoRoot, "_scratch", "go-tmp", "item1")
	_ = os.WriteFile(oldTmp, []byte("temp file contents 1234567890"), 0o644)
	oldTime := time.Now().Add(-3 * time.Hour)
	_ = os.Chtimes(oldTmp, oldTime, oldTime)

	// Create an expired scratch dir
	expiredScratch := filepath.Join(repoRoot, "_scratch", "old-producer", "data.txt")
	_ = os.WriteFile(expiredScratch, []byte("producer junk 12345"), 0o644)
	_ = os.Chtimes(filepath.Dir(expiredScratch), oldTime, oldTime)

	cfg := DefaultConfig()
	cfg.ScratchTTL = 1 * time.Hour
	sm := NewStorageManager(repoRoot, cfg)

	// Dry run first
	dryRes, err := sm.ReclaimCascade(context.Background(), true)
	if err != nil {
		t.Fatalf("ReclaimCascade dryRun: %v", err)
	}
	if dryRes.TotalBytes == 0 {
		t.Errorf("expected reclaimable bytes in dry run, got 0")
	}

	// Real apply
	res, err := sm.ReclaimCascade(context.Background(), false)
	if err != nil {
		t.Fatalf("ReclaimCascade apply: %v", err)
	}
	if res.TotalBytes == 0 {
		t.Errorf("expected reclaimed bytes in apply, got 0")
	}

	// Verify oldTmp and expiredScratch are deleted
	if _, err := os.Stat(oldTmp); !os.IsNotExist(err) {
		t.Errorf("expected oldTmp to be deleted, stat err: %v", err)
	}
	if _, err := os.Stat(filepath.Dir(expiredScratch)); !os.IsNotExist(err) {
		t.Errorf("expected expiredScratch dir to be deleted, stat err: %v", err)
	}
}

func TestEvaluateWatermark(t *testing.T) {
	cfg := DefaultConfig()
	sm := NewStorageManager("", cfg)

	w, r := sm.EvaluateWatermark(10 * 1024 * 1024 * 1024)
	if w || r {
		t.Errorf("expected no warning or refuse on 10GB free")
	}

	w, r = sm.EvaluateWatermark(3 * 1024 * 1024 * 1024)
	if !w || r {
		t.Errorf("expected warning but not refuse on 3GB free")
	}

	w, r = sm.EvaluateWatermark(1 * 1024 * 1024 * 1024)
	if !w || !r {
		t.Errorf("expected warning and refuse on 1GB free")
	}
}

func TestStorageManagerActiveScratchPreserved(t *testing.T) {
	tmp := t.TempDir()
	repoRoot := filepath.Join(tmp, "repo")
	scratchDir := filepath.Join(repoRoot, "_scratch")
	_ = os.MkdirAll(scratchDir, 0o755)

	oldTime := time.Now().Add(-3 * time.Hour)

	// 1. Expired scratch without locks or PIDs -> should be reclaimed
	expiredDir := filepath.Join(scratchDir, "old-expired")
	_ = os.MkdirAll(expiredDir, 0o755)
	_ = os.WriteFile(filepath.Join(expiredDir, "data.txt"), []byte("data"), 0o644)
	_ = os.Chtimes(expiredDir, oldTime, oldTime)

	// 2. Expired dir but has an active PID file -> should be preserved
	activePidDir := filepath.Join(scratchDir, "active-pid")
	_ = os.MkdirAll(activePidDir, 0o755)
	_ = os.WriteFile(filepath.Join(activePidDir, "worker.pid"), []byte("12345"), 0o644)
	_ = os.Chtimes(activePidDir, oldTime, oldTime)

	// 3. Expired dir with a dead PID file -> should be reclaimed
	deadPidDir := filepath.Join(scratchDir, "dead-pid")
	_ = os.MkdirAll(deadPidDir, 0o755)
	_ = os.WriteFile(filepath.Join(deadPidDir, "worker.pid"), []byte("99999"), 0o644)
	_ = os.Chtimes(deadPidDir, oldTime, oldTime)

	// 4. Expired dir with a recent lock file -> should be preserved
	activeLockDir := filepath.Join(scratchDir, "active-lock")
	_ = os.MkdirAll(activeLockDir, 0o755)
	_ = os.WriteFile(filepath.Join(activeLockDir, "sync.lock"), []byte("lock content"), 0o644)
	_ = os.Chtimes(activeLockDir, oldTime, oldTime)

	// 5. Expired dir with an active lease.json -> should be preserved
	activeLeaseDir := filepath.Join(scratchDir, "active-lease")
	_ = os.MkdirAll(activeLeaseDir, 0o755)
	_ = os.WriteFile(filepath.Join(activeLeaseDir, "lease.json"), []byte(`{"pid":12345}`), 0o644)
	_ = os.Chtimes(activeLeaseDir, oldTime, oldTime)

	// 6. Worker worktree directory -> should be skipped by reclaimTier1
	workerWtDir := filepath.Join(scratchDir, "fak-worker-wt-lane-123")
	_ = os.MkdirAll(workerWtDir, 0o755)
	_ = os.WriteFile(filepath.Join(workerWtDir, "code.go"), []byte("package main"), 0o644)
	_ = os.Chtimes(workerWtDir, oldTime, oldTime)

	cfg := DefaultConfig()
	cfg.ScratchTTL = 1 * time.Hour
	sm := NewStorageManager(repoRoot, cfg)
	sm.ProcessAlive = func(pid int) bool {
		return pid == 12345
	}

	bytesReclaimed, count, _, err := sm.reclaimTier1(context.Background(), false)
	if err != nil {
		t.Fatalf("reclaimTier1: %v", err)
	}
	if bytesReclaimed == 0 || count == 0 {
		t.Errorf("expected some bytes and files reclaimed from dead and expired dirs, got bytes=%d count=%d", bytesReclaimed, count)
	}

	// Verify old-expired was deleted
	if _, err := os.Stat(expiredDir); !os.IsNotExist(err) {
		t.Errorf("expected old-expired to be deleted")
	}

	// Verify dead-pid was deleted
	if _, err := os.Stat(deadPidDir); !os.IsNotExist(err) {
		t.Errorf("expected dead-pid to be deleted")
	}

	// Verify active-pid was preserved
	if _, err := os.Stat(activePidDir); err != nil {
		t.Errorf("expected active-pid to be preserved: %v", err)
	}

	// Verify active-lock was preserved
	if _, err := os.Stat(activeLockDir); err != nil {
		t.Errorf("expected active-lock to be preserved: %v", err)
	}

	// Verify active-lease was preserved
	if _, err := os.Stat(activeLeaseDir); err != nil {
		t.Errorf("expected active-lease to be preserved: %v", err)
	}

	// Verify worker worktree was preserved
	if _, err := os.Stat(workerWtDir); err != nil {
		t.Errorf("expected worker worktree to be preserved: %v", err)
	}
}

func TestStorageManagerCheckDiskSpace(t *testing.T) {
	cfg := DefaultConfig()
	sm := NewStorageManager("C:\\fake\\root", cfg)

	// 10 GB free: healthy
	sm.DiskFree = func(string) (int64, error) {
		return 10 * 1024 * 1024 * 1024, nil
	}
	free, w, r, err := sm.CheckDiskSpace()
	if err != nil || w || r || free != 10*1024*1024*1024 {
		t.Errorf("expected healthy 10GB: free=%d w=%v r=%v err=%v", free, w, r, err)
	}

	// 3 GB free: warning
	sm.DiskFree = func(string) (int64, error) {
		return 3 * 1024 * 1024 * 1024, nil
	}
	free, w, r, err = sm.CheckDiskSpace()
	if err != nil || !w || r || free != 3*1024*1024*1024 {
		t.Errorf("expected warning 3GB: free=%d w=%v r=%v err=%v", free, w, r, err)
	}

	// 1 GB free: refuse
	sm.DiskFree = func(string) (int64, error) {
		return 1 * 1024 * 1024 * 1024, nil
	}
	free, w, r, err = sm.CheckDiskSpace()
	if err != nil || !w || !r || free != 1*1024*1024*1024 {
		t.Errorf("expected refuse 1GB: free=%d w=%v r=%v err=%v", free, w, r, err)
	}
}
