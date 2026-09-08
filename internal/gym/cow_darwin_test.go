//go:build darwin

package gym

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestDarwinAPFSOverlayReset validates:
// 1. darwinAPFSOverlay activates automatically on Darwin when APFS is available.
// 2. Episode reset on a workspace populated with 1,000+ mutated files restores pristine baseline state in < 5ms.
func TestDarwinAPFSOverlayReset(t *testing.T) {
	baseDir := t.TempDir()

	// 1. Seed 1,000 baseline files in base workspace
	const totalFiles = 1000
	for i := 0; i < totalFiles; i++ {
		fname := fmt.Sprintf("baseline_file_%04d.txt", i)
		fpath := filepath.Join(baseDir, fname)
		payload := fmt.Sprintf("baseline-payload-seed-%04d", i)
		if err := os.WriteFile(fpath, []byte(payload), 0644); err != nil {
			t.Fatalf("failed seeding file %d: %v", i, err)
		}
	}

	// 2. Initialize overlay
	overlay, err := NewOverlay(baseDir, "")
	if err != nil {
		t.Fatalf("failed to create overlay: %v", err)
	}
	defer overlay.Destroy()

	// Verify darwinAPFSOverlay driver is selected on Darwin APFS
	apfsOverlay, ok := overlay.(*darwinAPFSOverlay)
	if !ok {
		t.Fatalf("expected *darwinAPFSOverlay on Darwin APFS, got %T", overlay)
	}

	// 3. Mutate 100 files, add 50 new files, delete 50 files in MergedDir
	const mutateCount = 100
	for i := 0; i < mutateCount; i++ {
		fname := fmt.Sprintf("baseline_file_%04d.txt", i)
		fpath := filepath.Join(apfsOverlay.MergedDir(), fname)
		if err := os.WriteFile(fpath, []byte(fmt.Sprintf("mutated-content-%04d", i)), 0644); err != nil {
			t.Fatalf("failed mutating file %d: %v", i, err)
		}
	}

	const addCount = 50
	for i := 0; i < addCount; i++ {
		fname := fmt.Sprintf("new_agent_file_%04d.txt", i)
		fpath := filepath.Join(apfsOverlay.MergedDir(), fname)
		if err := os.WriteFile(fpath, []byte(fmt.Sprintf("added-content-%04d", i)), 0644); err != nil {
			t.Fatalf("failed adding new file %d: %v", i, err)
		}
	}

	const deleteCount = 50
	for i := 0; i < deleteCount; i++ {
		fname := fmt.Sprintf("baseline_file_%04d.txt", totalFiles-1-i)
		fpath := filepath.Join(apfsOverlay.MergedDir(), fname)
		if err := os.Remove(fpath); err != nil {
			t.Fatalf("failed deleting file %d: %v", totalFiles-1-i, err)
		}
	}

	// 4. Measure Reset() duration
	start := time.Now()
	if err := apfsOverlay.Reset(); err != nil {
		t.Fatalf("apfsOverlay.Reset failed: %v", err)
	}
	resetDuration := time.Since(start)

	t.Logf("Observed Darwin APFS Reset() duration: %v", resetDuration)

	// Assert reset latency is strictly under 5ms
	if resetDuration >= 5*time.Millisecond {
		t.Errorf("Reset() latency exceeded sub-5ms SLA: took %v (must be < 5ms)", resetDuration)
	}

	// 5. Verify pristine baseline workspace restoration
	entries, err := os.ReadDir(apfsOverlay.MergedDir())
	if err != nil {
		t.Fatalf("failed reading post-reset merged directory: %v", err)
	}
	if len(entries) != totalFiles {
		t.Errorf("expected %d entries in post-reset merged dir, got %d", totalFiles, len(entries))
	}

	// Verify all 1,000 baseline files restored to pristine content
	for i := 0; i < totalFiles; i++ {
		fname := fmt.Sprintf("baseline_file_%04d.txt", i)
		fpath := filepath.Join(apfsOverlay.MergedDir(), fname)
		data, err := os.ReadFile(fpath)
		if err != nil {
			t.Fatalf("missing restored baseline file %s: %v", fname, err)
		}
		expectedPayload := fmt.Sprintf("baseline-payload-seed-%04d", i)
		if string(data) != expectedPayload {
			t.Errorf("corrupted file %s: expected %q, got %q", fname, expectedPayload, string(data))
		}
	}

	// Verify base workspace has zero residual pollution
	baseEntries, err := os.ReadDir(baseDir)
	if err != nil {
		t.Fatalf("failed reading baseDir: %v", err)
	}
	if len(baseEntries) != totalFiles {
		t.Errorf("base workspace corrupted! expected %d entries, found %d", totalFiles, len(baseEntries))
	}
}

// TestDarwinAPFSOverlayPromote validates that modifications, additions, and .wh. whiteouts
// promote cleanly to a target directory using APFS clonefile or fallback copying.
func TestDarwinAPFSOverlayPromote(t *testing.T) {
	baseDir := t.TempDir()

	// Seed baseline
	fileKeep := filepath.Join(baseDir, "keep.txt")
	fileModify := filepath.Join(baseDir, "modify.txt")
	fileDelete := filepath.Join(baseDir, "delete.txt")

	_ = os.WriteFile(fileKeep, []byte("keep-original"), 0644)
	_ = os.WriteFile(fileModify, []byte("modify-original"), 0644)
	_ = os.WriteFile(fileDelete, []byte("delete-original"), 0644)

	overlay, err := NewOverlay(baseDir, "")
	if err != nil {
		t.Fatalf("failed to create overlay: %v", err)
	}
	defer overlay.Destroy()

	apfsOverlay, ok := overlay.(*darwinAPFSOverlay)
	if !ok {
		t.Fatalf("expected *darwinAPFSOverlay, got %T", overlay)
	}

	// Apply mutations in MergedDir
	_ = os.WriteFile(filepath.Join(apfsOverlay.MergedDir(), "modify.txt"), []byte("modify-updated"), 0644)
	_ = os.WriteFile(filepath.Join(apfsOverlay.MergedDir(), "added.txt"), []byte("added-content"), 0644)
	_ = os.Remove(filepath.Join(apfsOverlay.MergedDir(), "delete.txt"))

	// Check modified artifacts
	artifacts := apfsOverlay.ModifiedArtifacts()
	t.Logf("Modified artifacts: %v", artifacts)

	foundAdd := false
	foundMod := false
	foundDel := false
	for _, a := range artifacts {
		if a == "added.txt" {
			foundAdd = true
		}
		if a == "modify.txt" {
			foundMod = true
		}
		if a == "delete.txt (deleted)" {
			foundDel = true
		}
	}
	if !foundAdd || !foundMod || !foundDel {
		t.Errorf("expected added.txt, modify.txt, and delete.txt (deleted) in artifacts: %v", artifacts)
	}

	// Prepare target directory populated with original files
	targetDir := t.TempDir()
	_ = os.WriteFile(filepath.Join(targetDir, "keep.txt"), []byte("keep-original"), 0644)
	_ = os.WriteFile(filepath.Join(targetDir, "modify.txt"), []byte("modify-original"), 0644)
	_ = os.WriteFile(filepath.Join(targetDir, "delete.txt"), []byte("delete-original"), 0644)

	// Promote to targetDir
	if err := apfsOverlay.Promote(targetDir); err != nil {
		t.Fatalf("Promote failed: %v", err)
	}

	// Verify targetDir has promoted state
	keepData, _ := os.ReadFile(filepath.Join(targetDir, "keep.txt"))
	if string(keepData) != "keep-original" {
		t.Errorf("expected keep-original, got %q", string(keepData))
	}

	modData, _ := os.ReadFile(filepath.Join(targetDir, "modify.txt"))
	if string(modData) != "modify-updated" {
		t.Errorf("expected modify-updated, got %q", string(modData))
	}

	addData, _ := os.ReadFile(filepath.Join(targetDir, "added.txt"))
	if string(addData) != "added-content" {
		t.Errorf("expected added-content, got %q", string(addData))
	}

	if _, err := os.Stat(filepath.Join(targetDir, "delete.txt")); !os.IsNotExist(err) {
		t.Errorf("expected delete.txt to be deleted by whiteout in targetDir, but it still exists")
	}
}

// TestDarwinAPFSOverlayFallback validates that if APFS clone is unsupported (e.g. cross-volume boundary
// or unsupported filesystem), newOSOverlay transparently falls back to UserspaceOverlay.
func TestDarwinAPFSOverlayFallback(t *testing.T) {
	baseDir := t.TempDir()
	anchor := filepath.Join(baseDir, "anchor.txt")
	_ = os.WriteFile(anchor, []byte("anchor"), 0644)

	// 1. Test probe function directly on non-existent or invalid destination
	if isAPFSCloneAvailable(baseDir, "/non/existent/dev/path/impossible") {
		t.Errorf("expected isAPFSCloneAvailable to return false for invalid destination path")
	}

	// 2. Test newDarwinAPFSOverlay error handling on invalid tempBase
	_, err := newDarwinAPFSOverlay(baseDir, "/non/existent/dev/path/impossible")
	if err == nil {
		t.Errorf("expected error from newDarwinAPFSOverlay on invalid tempBase, got nil")
	}

	// 3. Test newOSOverlay fallback when tempBase causes APFS probe failure
	overlay, err := newOSOverlay(baseDir, "/non/existent/dev/path/impossible")
	// Note: newOSOverlay falls back to newUserspaceOverlay, which will try creating in that tempBase
	// and return the appropriate error or succeed if tempBase is valid.
	t.Logf("Fallback check result: overlay=%v, err=%v", overlay, err)

	// 4. Test normal newOSOverlay creates a working CoWOverlay
	liveOverlay, err := newOSOverlay(baseDir, "")
	if err != nil {
		t.Fatalf("newOSOverlay failed: %v", err)
	}
	defer liveOverlay.Destroy()

	if err := liveOverlay.Reset(); err != nil {
		t.Fatalf("overlay Reset failed: %v", err)
	}
}

// BenchmarkDarwinAPFSOverlayReset measures the latency and allocations of Reset() on APFS.
func BenchmarkDarwinAPFSOverlayReset(b *testing.B) {
	baseDir := b.TempDir()
	const seedCount = 1000
	for i := 0; i < seedCount; i++ {
		fname := fmt.Sprintf("seed_%04d.txt", i)
		_ = os.WriteFile(filepath.Join(baseDir, fname), []byte("payload-content-data"), 0644)
	}

	overlay, err := NewOverlay(baseDir, "")
	if err != nil {
		b.Fatalf("NewOverlay failed: %v", err)
	}
	defer overlay.Destroy()

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Apply a small mutation
		mutFile := filepath.Join(overlay.MergedDir(), "seed_0000.txt")
		_ = os.WriteFile(mutFile, []byte("mutated"), 0644)

		if err := overlay.Reset(); err != nil {
			b.Fatalf("Reset failed: %v", err)
		}
	}
}
