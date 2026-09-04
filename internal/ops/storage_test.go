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
