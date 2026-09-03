package benchsnapshot

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// TestPartialWriteInvisibility verifies that temporary and staging files (e.g. *.tmp.*)
// created before rename are never visible to SnapshotWatcher.ScanCommitted().
func TestPartialWriteInvisibility(t *testing.T) {
	tmpDir := t.TempDir()
	runID := "run-invis-001"
	phase := "eval"

	watcher := NewWatcher(tmpDir, runID, phase)
	writer, err := NewWriter(tmpDir, runID, phase)
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}

	phaseDir := filepath.Join(tmpDir, runID, phase)
	if err := os.MkdirAll(phaseDir, 0755); err != nil {
		t.Fatalf("failed to create phase dir: %v", err)
	}

	// Create various temporary and staging files that must be invisible.
	stagingFiles := []string{
		"000001.tmp.abcd1234ef56",
		"000002.tmp.99999",
		"000003.tmp",
		"000004.json.tmp",
		"staging.tmp",
		"random.txt",
		"not_a_json_file",
		"00001.json", // only 5 digits zero-padded (not 6 digits)
	}
	for _, sf := range stagingFiles {
		p := filepath.Join(phaseDir, sf)
		if err := os.WriteFile(p, []byte(`{"partial": true}`), 0644); err != nil {
			t.Fatalf("failed to create staging file %s: %v", sf, err)
		}
	}

	// Watcher must see 0 committed snapshots despite staging files existing.
	entries, err := watcher.ScanCommitted()
	if err != nil {
		t.Fatalf("ScanCommitted failed: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected 0 committed snapshots, got %d: %+v", len(entries), entries)
	}

	// Now write an actual atomic snapshot.
	validPayload := []byte(`{"status": "committed", "step": 1}`)
	receipt, err := writer.WriteSnapshot(validPayload)
	if err != nil {
		t.Fatalf("WriteSnapshot failed: %v", err)
	}
	if receipt.Sequence != 1 {
		t.Fatalf("expected sequence 1, got %d", receipt.Sequence)
	}

	// Watcher must now observe exactly 1 snapshot, ignoring all staging files.
	entries, err = watcher.ScanCommitted()
	if err != nil {
		t.Fatalf("ScanCommitted failed: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 committed snapshot, got %d", len(entries))
	}
	if entries[0].Sequence != 1 {
		t.Fatalf("expected entry sequence 1, got %d", entries[0].Sequence)
	}
	if entries[0].Path != receipt.Path {
		t.Fatalf("expected path %q, got %q", receipt.Path, entries[0].Path)
	}
}

// TestMonotonicSequentialOrdering proves that snapshots 1, 2, 3 are written and
// scanned in exact sequential order.
func TestMonotonicSequentialOrdering(t *testing.T) {
	tmpDir := t.TempDir()
	runID := "run-order-002"
	phase := "benchmark"

	writer, err := NewWriter(tmpDir, runID, phase)
	if err != nil {
		t.Fatalf("NewWriter failed: %v", err)
	}
	watcher := NewWatcher(tmpDir, runID, phase)

	payloads := [][]byte{
		[]byte(`{"step": 1, "accuracy": 0.81}`),
		[]byte(`{"step": 2, "accuracy": 0.86}`),
		[]byte(`{"step": 3, "accuracy": 0.92}`),
	}

	for i, payload := range payloads {
		receipt, err := writer.WriteSnapshot(payload)
		if err != nil {
			t.Fatalf("WriteSnapshot %d failed: %v", i+1, err)
		}
		expectedSeq := i + 1
		if receipt.Sequence != expectedSeq {
			t.Fatalf("expected receipt sequence %d, got %d", expectedSeq, receipt.Sequence)
		}
		expectedName := fmt.Sprintf("%06d.json", expectedSeq)
		if filepath.Base(receipt.Path) != expectedName {
			t.Fatalf("expected path to end in %q, got %q", expectedName, receipt.Path)
		}
	}

	entries, err := watcher.ScanCommitted()
	if err != nil {
		t.Fatalf("ScanCommitted failed: %v", err)
	}
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(entries))
	}

	for i, entry := range entries {
		expectedSeq := i + 1
		if entry.Sequence != expectedSeq {
			t.Errorf("entry %d: expected sequence %d, got %d", i, expectedSeq, entry.Sequence)
		}
		expectedName := fmt.Sprintf("%06d.json", expectedSeq)
		if filepath.Base(entry.Path) != expectedName {
			t.Errorf("entry %d: expected filename %q, got %q", i, expectedName, filepath.Base(entry.Path))
		}
		data, err := entry.ReadPayload()
		if err != nil {
			t.Errorf("entry %d: ReadPayload failed: %v", i, err)
		}
		if !bytes.Equal(data, payloads[i]) {
			t.Errorf("entry %d: payload mismatch. Got %s, want %s", i, string(data), string(payloads[i]))
		}
	}
}

// TestRestartCollisionHandling verifies all 3 restart policies:
// RestartRejectCollision (prevents overwrite), RestartResumeNext, and RestartFreshGeneration.
func TestRestartCollisionHandling(t *testing.T) {
	t.Run("RestartRejectCollision", func(t *testing.T) {
		tmpDir := t.TempDir()
		runID := "run-restart-001"
		phase := "train"

		w1, err := NewWriter(tmpDir, runID, phase)
		if err != nil {
			t.Fatalf("NewWriter w1 failed: %v", err)
		}

		origPayload := []byte(`{"initial": "run1"}`)
		r1, err := w1.WriteSnapshot(origPayload)
		if err != nil {
			t.Fatalf("WriteSnapshot w1 failed: %v", err)
		}
		if r1.Sequence != 1 {
			t.Fatalf("expected sequence 1, got %d", r1.Sequence)
		}

		// New writer on same runID/phase with default policy RestartRejectCollision.
		w2, err := NewWriter(tmpDir, runID, phase)
		if err != nil {
			t.Fatalf("NewWriter w2 failed: %v", err)
		}
		if w2.RestartMode != RestartRejectCollision {
			t.Fatalf("expected RestartRejectCollision default, got %v", w2.RestartMode)
		}

		// Attempting to write sequence 1 again must be rejected.
		_, err = w2.WriteSnapshot([]byte(`{"overwrite": "attempt"}`))
		if err == nil {
			t.Fatal("expected collision error from w2, got nil")
		}
		if !errors.Is(err, ErrSequenceCollision) {
			t.Fatalf("expected ErrSequenceCollision, got: %v", err)
		}

		// Ensure original file content was not overwritten.
		currentData, err := os.ReadFile(r1.Path)
		if err != nil {
			t.Fatalf("failed to read original snapshot: %v", err)
		}
		if !bytes.Equal(currentData, origPayload) {
			t.Fatalf("file content was modified! Got %s, want %s", string(currentData), string(origPayload))
		}
	})

	t.Run("RestartResumeNext", func(t *testing.T) {
		tmpDir := t.TempDir()
		runID := "run-restart-002"
		phase := "infer"

		w1, err := NewWriter(tmpDir, runID, phase)
		if err != nil {
			t.Fatalf("NewWriter w1 failed: %v", err)
		}
		if _, err := w1.WriteSnapshot([]byte(`{"p": 1}`)); err != nil {
			t.Fatalf("w1 write 1 failed: %v", err)
		}
		if _, err := w1.WriteSnapshot([]byte(`{"p": 2}`)); err != nil {
			t.Fatalf("w1 write 2 failed: %v", err)
		}

		// Create w2 with RestartResumeNext.
		w2, err := NewWriterWithMode(tmpDir, runID, phase, RestartResumeNext)
		if err != nil {
			t.Fatalf("NewWriter w2 failed: %v", err)
		}

		r3, err := w2.WriteSnapshot([]byte(`{"p": 3}`))
		if err != nil {
			t.Fatalf("w2 write resumed snapshot failed: %v", err)
		}
		if r3.Sequence != 3 {
			t.Fatalf("expected sequence 3 on resume, got %d", r3.Sequence)
		}

		watcher := NewWatcher(tmpDir, runID, phase)
		entries, err := watcher.ScanCommitted()
		if err != nil {
			t.Fatalf("ScanCommitted failed: %v", err)
		}
		if len(entries) != 3 {
			t.Fatalf("expected 3 committed snapshots, got %d", len(entries))
		}
		for i, e := range entries {
			if e.Sequence != i+1 {
				t.Fatalf("expected entry sequence %d, got %d", i+1, e.Sequence)
			}
		}
	})

	t.Run("RestartFreshGeneration", func(t *testing.T) {
		tmpDir := t.TempDir()
		runID := "run-restart-003"
		phase := "eval"

		w1, err := NewWriter(tmpDir, runID, phase)
		if err != nil {
			t.Fatalf("NewWriter w1 failed: %v", err)
		}
		if _, err := w1.WriteSnapshot([]byte(`{"gen1": true}`)); err != nil {
			t.Fatalf("w1 write failed: %v", err)
		}

		// Writer 2 with RestartFreshGeneration should allocate phase-gen2.
		w2, err := NewWriterWithMode(tmpDir, runID, phase, RestartFreshGeneration)
		if err != nil {
			t.Fatalf("NewWriter w2 failed: %v", err)
		}

		r2, err := w2.WriteSnapshot([]byte(`{"gen2": true}`))
		if err != nil {
			t.Fatalf("w2 write fresh generation failed: %v", err)
		}
		if r2.Phase != "eval-gen2" {
			t.Fatalf("expected phase 'eval-gen2', got %q", r2.Phase)
		}
		if r2.Sequence != 1 {
			t.Fatalf("expected sequence 1 for fresh generation, got %d", r2.Sequence)
		}

		// Writer 3 with RestartFreshGeneration should allocate phase-gen3.
		w3, err := NewWriterWithMode(tmpDir, runID, phase, RestartFreshGeneration)
		if err != nil {
			t.Fatalf("NewWriter w3 failed: %v", err)
		}
		r3, err := w3.WriteSnapshot([]byte(`{"gen3": true}`))
		if err != nil {
			t.Fatalf("w3 write fresh generation failed: %v", err)
		}
		if r3.Phase != "eval-gen3" {
			t.Fatalf("expected phase 'eval-gen3', got %q", r3.Phase)
		}
		if r3.Sequence != 1 {
			t.Fatalf("expected sequence 1 for fresh generation, got %d", r3.Sequence)
		}
	})
}

// TestDeterministicLexicographicReplay proves that scanning committed snapshots
// yields strictly lexicographic, monotonically increasing sequence order and
// deterministically replays identical results across scans.
func TestDeterministicLexicographicReplay(t *testing.T) {
	tmpDir := t.TempDir()
	runID := "run-replay-003"
	phase := "sweep"

	writer, err := NewWriter(tmpDir, runID, phase)
	if err != nil {
		t.Fatalf("NewWriter failed: %v", err)
	}
	watcher := NewWatcher(tmpDir, runID, phase)

	count := 25
	for i := 1; i <= count; i++ {
		payload := []byte(fmt.Sprintf(`{"index": %d, "data": "payload_%04d"}`, i, i))
		if _, err := writer.WriteSnapshot(payload); err != nil {
			t.Fatalf("WriteSnapshot %d failed: %v", i, err)
		}
	}

	// Perform multiple scans to test deterministic replay.
	for run := 0; run < 3; run++ {
		entries, err := watcher.ScanCommitted()
		if err != nil {
			t.Fatalf("scan run %d failed: %v", run, err)
		}
		if len(entries) != count {
			t.Fatalf("scan run %d: expected %d entries, got %d", run, count, len(entries))
		}

		for i := 0; i < count; i++ {
			expectedSeq := i + 1
			if entries[i].Sequence != expectedSeq {
				t.Fatalf("scan %d, item %d: expected seq %d, got %d", run, i, expectedSeq, entries[i].Sequence)
			}
			expectedName := fmt.Sprintf("%06d.json", expectedSeq)
			if filepath.Base(entries[i].Path) != expectedName {
				t.Fatalf("scan %d, item %d: expected filename %q, got %q", run, i, expectedName, filepath.Base(entries[i].Path))
			}
			if i > 0 {
				if !(entries[i-1].Path < entries[i].Path) {
					t.Fatalf("lexicographic order broken: %q not < %q", entries[i-1].Path, entries[i].Path)
				}
				if !(entries[i-1].Sequence < entries[i].Sequence) {
					t.Fatalf("monotonic sequence order broken: %d not < %d", entries[i-1].Sequence, entries[i].Sequence)
				}
			}
		}
	}
}

// TestIntegrityDigestMatchesPayload proves that SHA-256 digests in receipts and
// scanned entries match the cryptographic hash of the payload bytes.
func TestIntegrityDigestMatchesPayload(t *testing.T) {
	tmpDir := t.TempDir()
	runID := "run-digest-004"
	phase := "audit"

	writer, err := NewWriter(tmpDir, runID, phase)
	if err != nil {
		t.Fatalf("NewWriter failed: %v", err)
	}
	watcher := NewWatcher(tmpDir, runID, phase)

	testCases := []struct {
		name    string
		payload []byte
	}{
		{
			name:    "json-object",
			payload: []byte(`{"metric": "latency", "p99": 14.25, "unit": "ms"}`),
		},
		{
			name:    "empty-bytes",
			payload: []byte{},
		},
		{
			name:    "binary-payload",
			payload: []byte{0x00, 0x01, 0x02, 0xFE, 0xFF, 0x42, 0x13, 0x37},
		},
		{
			name:    "large-payload",
			payload: bytes.Repeat([]byte("benchmark-evidence-data-chunk;"), 2048),
		},
	}

	for i, tc := range testCases {
		h := sha256.Sum256(tc.payload)
		wantSHA := hex.EncodeToString(h[:])
		wantBytes := int64(len(tc.payload))

		receipt, err := writer.WriteSnapshot(tc.payload)
		if err != nil {
			t.Fatalf("case %q: WriteSnapshot failed: %v", tc.name, err)
		}

		if receipt.SHA256 != wantSHA {
			t.Errorf("case %q: receipt SHA mismatch: got %q, want %q", tc.name, receipt.SHA256, wantSHA)
		}
		if receipt.ByteCount != wantBytes {
			t.Errorf("case %q: receipt ByteCount mismatch: got %d, want %d", tc.name, receipt.ByteCount, wantBytes)
		}
		if receipt.Sequence != i+1 {
			t.Errorf("case %q: sequence mismatch: got %d, want %d", tc.name, receipt.Sequence, i+1)
		}
	}

	entries, err := watcher.ScanCommitted()
	if err != nil {
		t.Fatalf("ScanCommitted failed: %v", err)
	}
	if len(entries) != len(testCases) {
		t.Fatalf("expected %d entries, got %d", len(testCases), len(entries))
	}

	for i, tc := range testCases {
		h := sha256.Sum256(tc.payload)
		wantSHA := hex.EncodeToString(h[:])
		wantBytes := int64(len(tc.payload))

		entry := entries[i]
		if entry.SHA256 != wantSHA {
			t.Errorf("case %q: entry SHA mismatch: got %q, want %q", tc.name, entry.SHA256, wantSHA)
		}
		if entry.ByteCount != wantBytes {
			t.Errorf("case %q: entry ByteCount mismatch: got %d, want %d", tc.name, entry.ByteCount, wantBytes)
		}

		readPayload, err := entry.ReadPayload()
		if err != nil {
			t.Errorf("case %q: ReadPayload failed: %v", tc.name, err)
		}
		if !bytes.Equal(readPayload, tc.payload) {
			t.Errorf("case %q: payload bytes mismatch", tc.name)
		}
	}
}

// TestConcurrentWritesAndWatches ensures that concurrent calls to WriteSnapshot
// and ScanCommitted produce no race conditions and maintain full data integrity.
func TestConcurrentWritesAndWatches(t *testing.T) {
	tmpDir := t.TempDir()
	runID := "run-concurrent-005"
	phase := "concurrent"

	writer, err := NewWriter(tmpDir, runID, phase)
	if err != nil {
		t.Fatalf("NewWriter failed: %v", err)
	}
	watcher := NewWatcher(tmpDir, runID, phase)

	const totalWrites = 30
	var wg sync.WaitGroup
	wg.Add(1)

	// Background watcher loop verifying only committed snapshots are ever seen.
	stopWatcher := make(chan struct{})
	watcherErrCh := make(chan error, 1)

	go func() {
		defer close(watcherErrCh)
		for {
			select {
			case <-stopWatcher:
				return
			default:
				entries, err := watcher.ScanCommitted()
				if err != nil {
					watcherErrCh <- err
					return
				}
				// Verify strictly increasing sequence order.
				for i := 1; i < len(entries); i++ {
					if entries[i].Sequence <= entries[i-1].Sequence {
						watcherErrCh <- fmt.Errorf("non-monotonic sequence observed: %d <= %d",
							entries[i].Sequence, entries[i-1].Sequence)
						return
					}
				}
			}
		}
	}()

	// Writers writing snapshots concurrently.
	go func() {
		defer wg.Done()
		for i := 1; i <= totalWrites; i++ {
			payload := []byte(fmt.Sprintf(`{"worker": "main", "seq": %d}`, i))
			if _, err := writer.WriteSnapshot(payload); err != nil {
				t.Errorf("concurrent WriteSnapshot %d failed: %v", i, err)
				return
			}
		}
	}()

	wg.Wait()
	close(stopWatcher)

	if err := <-watcherErrCh; err != nil {
		t.Fatalf("watcher encountered error during concurrent writes: %v", err)
	}

	finalEntries, err := watcher.ScanCommitted()
	if err != nil {
		t.Fatalf("final ScanCommitted failed: %v", err)
	}
	if len(finalEntries) != totalWrites {
		t.Fatalf("expected %d final entries, got %d", totalWrites, len(finalEntries))
	}
}

// TestValidationAndEdgeCases tests argument validation and handling of non-existent directories.
func TestValidationAndEdgeCases(t *testing.T) {
	t.Run("InvalidArguments", func(t *testing.T) {
		if _, err := NewWriter("", "run", "phase"); err == nil {
			t.Error("expected error for empty baseDir")
		}
		if _, err := NewWriter("base", "", "phase"); err == nil {
			t.Error("expected error for empty runID")
		}
		if _, err := NewWriter("base", "run", ""); err == nil {
			t.Error("expected error for empty phase")
		}
	})

	t.Run("NonExistentDirScan", func(t *testing.T) {
		watcher := NewWatcher(t.TempDir(), "non-existent-run", "non-existent-phase")
		entries, err := watcher.ScanCommitted()
		if err != nil {
			t.Fatalf("expected no error for non-existent directory, got %v", err)
		}
		if len(entries) != 0 {
			t.Fatalf("expected 0 entries for non-existent dir, got %d", len(entries))
		}
	})

	t.Run("RestartModeString", func(t *testing.T) {
		if RestartRejectCollision.String() != "RestartRejectCollision" {
			t.Errorf("got %s", RestartRejectCollision.String())
		}
		if RestartResumeNext.String() != "RestartResumeNext" {
			t.Errorf("got %s", RestartResumeNext.String())
		}
		if RestartFreshGeneration.String() != "RestartFreshGeneration" {
			t.Errorf("got %s", RestartFreshGeneration.String())
		}
	})
}
