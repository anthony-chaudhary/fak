package sessionjournal

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// TestSegmentRotationWhenThresholdExceeded verifies segment rotation by entry count,
// byte size, and manual rotation trigger.
func TestSegmentRotationWhenThresholdExceeded(t *testing.T) {
	dir := t.TempDir()
	sj := NewSegmentedJournal(dir, WithMaxEntries(5), WithSyncOnAppend(false))

	// 1. Append 12 events. With MaxEntries=5:
	// - Events 1..5 in session-journal.000001.jsonl
	// - Events 6..10 in session-journal.000002.jsonl
	// - Events 11..12 in session-journal.active.jsonl
	for i := 1; i <= 12; i++ {
		ev := Event{
			ID:   fmt.Sprintf("session-%d", i),
			Kind: KindOpen,
			TS:   time.Now().UTC().Format(time.RFC3339Nano),
		}
		if err := sj.AppendEvent(ev); err != nil {
			t.Fatalf("AppendEvent %d: %v", i, err)
		}
	}

	sealed, err := sj.SealedSegments()
	if err != nil {
		t.Fatalf("SealedSegments: %v", err)
	}
	if len(sealed) != 2 {
		t.Fatalf("expected 2 sealed segments, got %d: %v", len(sealed), sealed)
	}

	expectedSeg1 := filepath.Join(dir, "session-journal.000001.jsonl")
	expectedSeg2 := filepath.Join(dir, "session-journal.000002.jsonl")
	if sealed[0] != expectedSeg1 || sealed[1] != expectedSeg2 {
		t.Fatalf("unexpected sealed segment names: %v", sealed)
	}

	if count := countLines(expectedSeg1); count != 5 {
		t.Fatalf("segment 1 expected 5 lines, got %d", count)
	}
	if count := countLines(expectedSeg2); count != 5 {
		t.Fatalf("segment 2 expected 5 lines, got %d", count)
	}
	if count := countLines(sj.ActivePath()); count != 2 {
		t.Fatalf("active segment expected 2 lines, got %d", count)
	}

	// 2. Test manual RotateActive
	if err := sj.RotateActive(); err != nil {
		t.Fatalf("RotateActive: %v", err)
	}
	sealed, err = sj.SealedSegments()
	if err != nil {
		t.Fatalf("SealedSegments after manual rotate: %v", err)
	}
	if len(sealed) != 3 {
		t.Fatalf("expected 3 sealed segments, got %d", len(sealed))
	}
	expectedSeg3 := filepath.Join(dir, "session-journal.000003.jsonl")
	if sealed[2] != expectedSeg3 {
		t.Fatalf("segment 3 expected %s, got %s", expectedSeg3, sealed[2])
	}
	if count := countLines(expectedSeg3); count != 2 {
		t.Fatalf("segment 3 expected 2 lines, got %d", count)
	}

	// Rotate on empty active segment should be a safe no-op
	if err := sj.RotateActive(); err != nil {
		t.Fatalf("RotateActive on empty: %v", err)
	}
	sealed, _ = sj.SealedSegments()
	if len(sealed) != 3 {
		t.Fatalf("rotate on empty should not create segment, got %d", len(sealed))
	}

	// 3. Test rotation by byte size
	sizeDir := t.TempDir()
	sizeSJ := NewSegmentedJournal(sizeDir, WithMaxSizeBytes(250), WithMaxEntries(10000), WithSyncOnAppend(false))
	for i := 1; i <= 6; i++ {
		ev := Event{
			ID:   fmt.Sprintf("size-session-%04d", i),
			Kind: KindOpen,
			TS:   time.Now().UTC().Format(time.RFC3339Nano),
			CWD:  "/workspace/project/subpath",
		}
		if err := sizeSJ.AppendEvent(ev); err != nil {
			t.Fatalf("AppendEvent size: %v", err)
		}
	}
	sizeSealed, err := sizeSJ.SealedSegments()
	if err != nil {
		t.Fatalf("SealedSegments size: %v", err)
	}
	if len(sizeSealed) < 1 {
		t.Fatalf("expected at least 1 sealed segment due to size threshold, got %d", len(sizeSealed))
	}
}

// TestCheckpointFoldingAndSub50msRecovery verifies that background compaction folds
// sealed segments into checkpoint.json and that FastFoldRecovery recovers in sub-50ms.
func TestCheckpointFoldingAndSub50msRecovery(t *testing.T) {
	dir := t.TempDir()
	// Rotate active segment every 200 entries
	sj := NewSegmentedJournal(dir, WithMaxEntries(200), WithSyncOnAppend(false))

	const numSessions = 500
	const eventsPerSession = 20
	// 500 * 20 = 10,000 events total
	baseTime := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)

	for i := 0; i < numSessions; i++ {
		sessionID := fmt.Sprintf("bench-session-%04d", i)
		// Open event
		tOpen := baseTime.Add(time.Duration(i) * time.Second)
		_ = sj.AppendEvent(Event{
			ID:    sessionID,
			Kind:  KindOpen,
			TS:    tOpen.Format(time.RFC3339Nano),
			Model: "claude-3-7-sonnet",
			PID:   1000 + i,
		})

		// Heartbeats
		for b := 1; b <= eventsPerSession-2; b++ {
			tBeat := tOpen.Add(time.Duration(b) * 10 * time.Millisecond)
			_ = sj.AppendEvent(Event{
				ID:   sessionID,
				Kind: KindBeat,
				TS:   tBeat.Format(time.RFC3339Nano),
				Drive: &DriveCarry{
					TurnsLeft: int64(100 - b),
				},
			})
		}

		// Close event (half closed, half live)
		if i%2 == 0 {
			tClose := tOpen.Add(time.Duration(eventsPerSession) * 10 * time.Millisecond)
			_ = sj.AppendEvent(Event{
				ID:     sessionID,
				Kind:   KindClose,
				TS:     tClose.Format(time.RFC3339Nano),
				Reason: "done",
			})
		}
	}

	sealed, err := sj.SealedSegments()
	if err != nil {
		t.Fatalf("SealedSegments: %v", err)
	}
	if len(sealed) < 40 {
		t.Fatalf("expected at least 40 sealed segments, got %d", len(sealed))
	}

	// Execute background compaction
	if err := sj.CompactCheckpoint(); err != nil {
		t.Fatalf("CompactCheckpoint: %v", err)
	}

	cp, err := sj.CurrentCheckpoint()
	if err != nil {
		t.Fatalf("CurrentCheckpoint: %v", err)
	}
	if cp == nil || cp.CompactedSeq == 0 {
		t.Fatalf("checkpoint invalid: %+v", cp)
	}
	if len(cp.Sessions) == 0 {
		t.Fatalf("checkpoint should have folded sessions, got 0")
	}

	// Add 50 new events to the active segment post-compaction
	for i := 0; i < 50; i++ {
		_ = sj.AppendEvent(Event{
			ID:   fmt.Sprintf("post-compact-%04d", i),
			Kind: KindOpen,
			TS:   time.Now().UTC().Format(time.RFC3339Nano),
		})
	}

	// Measure FastFoldRecovery latency
	start := time.Now()
	recovered, err := sj.FastFoldRecovery()
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("FastFoldRecovery: %v", err)
	}

	t.Logf("FastFoldRecovery recovered %d sessions across 10,050 total events in %v", len(recovered), elapsed)

	if elapsed >= 50*time.Millisecond {
		t.Fatalf("FastFoldRecovery exceeded 50ms budget: took %v", elapsed)
	}

	// Verify session states
	expectedTotalSessions := numSessions + 50
	if len(recovered) != expectedTotalSessions {
		t.Fatalf("expected %d sessions, got %d", expectedTotalSessions, len(recovered))
	}

	// Check that closed vs open sessions are accurate
	closedCount := 0
	liveCount := 0
	for _, s := range recovered {
		if s.Closed {
			closedCount++
			if s.CloseReason != "done" {
				t.Errorf("session %s expected close reason 'done', got %q", s.ID, s.CloseReason)
			}
		} else {
			liveCount++
		}
	}

	expectedClosed := numSessions / 2
	expectedLive := (numSessions - expectedClosed) + 50
	if closedCount != expectedClosed {
		t.Errorf("expected %d closed sessions, got %d", expectedClosed, closedCount)
	}
	if liveCount != expectedLive {
		t.Errorf("expected %d live sessions, got %d", expectedLive, liveCount)
	}
}

// TestFastFoldRecoveryWithRetainedHistoricalSegments verifies that FastFoldRecovery
// does NOT scan historical segments even when they remain on disk (RetainCompacted=true).
func TestFastFoldRecoveryWithRetainedHistoricalSegments(t *testing.T) {
	dir := t.TempDir()
	sj := NewSegmentedJournal(dir, WithMaxEntries(100), WithRetainCompacted(true), WithSyncOnAppend(false))

	// Generate 2,000 events
	for i := 0; i < 2000; i++ {
		_ = sj.AppendEvent(Event{
			ID:   fmt.Sprintf("retain-sess-%d", i%50),
			Kind: KindBeat,
			TS:   time.Now().UTC().Format(time.RFC3339Nano),
		})
	}

	if err := sj.CompactCheckpoint(); err != nil {
		t.Fatalf("CompactCheckpoint: %v", err)
	}

	// Verify sealed segments still exist on disk because RetainCompacted is true
	sealed, err := sj.SealedSegments()
	if err != nil {
		t.Fatalf("SealedSegments: %v", err)
	}
	if len(sealed) < 18 {
		t.Fatalf("expected >= 18 retained sealed segments, got %d", len(sealed))
	}

	// Append a few more events to active segment
	_ = sj.AppendEvent(Event{
		ID:   "active-only-session",
		Kind: KindOpen,
		TS:   time.Now().UTC().Format(time.RFC3339Nano),
	})

	// Fast recovery should skip the retained historical segments
	start := time.Now()
	recovered, err := sj.FastFoldRecovery()
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("FastFoldRecovery: %v", err)
	}
	if elapsed >= 50*time.Millisecond {
		t.Fatalf("FastFoldRecovery took %v with retained segments, expected < 50ms", elapsed)
	}

	foundActive := false
	for _, s := range recovered {
		if s.ID == "active-only-session" {
			foundActive = true
			break
		}
	}
	if !foundActive {
		t.Fatalf("active-only-session not found in recovery")
	}
}

// TestConcurrentAppendsAndRaceFreedom verifies thread safety and lack of race conditions
// under concurrent appends, active rotations, and background compactions.
func TestConcurrentAppendsAndRaceFreedom(t *testing.T) {
	dir := t.TempDir()
	sj := NewSegmentedJournal(dir, WithMaxEntries(20), WithSyncOnAppend(false))

	const numWorkers = 16
	const eventsPerWorker = 50
	var wg sync.WaitGroup

	// Background worker rotating and compacting periodically
	stopCompactor := make(chan struct{})
	compactorDone := make(chan struct{})
	go func() {
		defer close(compactorDone)
		ticker := time.NewTicker(5 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stopCompactor:
				return
			case <-ticker.C:
				_ = sj.RotateActive()
				_ = sj.CompactCheckpoint()
			}
		}
	}()

	// Launch concurrent appenders
	wg.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		workerID := w
		go func() {
			defer wg.Done()
			for i := 0; i < eventsPerWorker; i++ {
				ev := Event{
					ID:   fmt.Sprintf("concurrent-sess-%d", workerID),
					Kind: KindBeat,
					TS:   time.Now().UTC().Format(time.RFC3339Nano),
					PID:  2000 + workerID,
					Drive: &DriveCarry{
						TurnsLeft: int64(i),
					},
				}
				if err := sj.AppendEvent(ev); err != nil {
					t.Errorf("worker %d append %d: %v", workerID, i, err)
					return
				}
			}
		}()
	}

	wg.Wait()
	close(stopCompactor)
	<-compactorDone

	// Final compaction and recovery
	if err := sj.CompactCheckpoint(); err != nil {
		t.Fatalf("final compact: %v", err)
	}

	recovered, err := sj.FastFoldRecovery()
	if err != nil {
		t.Fatalf("FastFoldRecovery: %v", err)
	}

	if len(recovered) != numWorkers {
		t.Fatalf("expected %d distinct sessions recovered, got %d", numWorkers, len(recovered))
	}
}

// TestFastFoldRecoveryReopenAcrossCheckpoint verifies that a session closed before
// compaction is cleanly reopened by an event in the active segment.
func TestFastFoldRecoveryReopenAcrossCheckpoint(t *testing.T) {
	dir := t.TempDir()
	sj := NewSegmentedJournal(dir, WithSyncOnAppend(false))

	t1 := time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC)
	t2 := time.Date(2026, 9, 1, 10, 30, 0, 0, time.UTC)
	t3 := time.Date(2026, 9, 1, 11, 0, 0, 0, time.UTC)

	// Open and close session-reopen in initial active segment
	_ = sj.AppendEvent(Event{
		ID:   "session-reopen",
		Kind: KindOpen,
		TS:   t1.Format(time.RFC3339Nano),
		CWD:  "/work/v1",
	})
	_ = sj.AppendEvent(Event{
		ID:     "session-reopen",
		Kind:   KindClose,
		TS:     t2.Format(time.RFC3339Nano),
		Reason: "initial-done",
	})

	// Rotate and compact into checkpoint
	if err := sj.RotateActive(); err != nil {
		t.Fatalf("RotateActive: %v", err)
	}
	if err := sj.CompactCheckpoint(); err != nil {
		t.Fatalf("CompactCheckpoint: %v", err)
	}

	// Verify checkpoint has it closed
	cp, err := sj.CurrentCheckpoint()
	if err != nil || len(cp.Sessions) != 1 || !cp.Sessions[0].Closed {
		t.Fatalf("checkpoint session should be closed: %+v", cp)
	}

	// Reopen in new active segment
	_ = sj.AppendEvent(Event{
		ID:   "session-reopen",
		Kind: KindOpen,
		TS:   t3.Format(time.RFC3339Nano),
		CWD:  "/work/v2",
	})

	recovered, err := sj.FastFoldRecovery()
	if err != nil {
		t.Fatalf("FastFoldRecovery: %v", err)
	}
	if len(recovered) != 1 {
		t.Fatalf("expected 1 session, got %d", len(recovered))
	}

	s := recovered[0]
	if s.Closed {
		t.Fatalf("reopened session should not be closed")
	}
	if s.CloseReason != "" {
		t.Fatalf("reopened session should have empty close reason, got %q", s.CloseReason)
	}
	if s.CWD != "/work/v2" {
		t.Fatalf("expected updated CWD /work/v2, got %q", s.CWD)
	}
	if !s.StartedAt.Equal(t3) {
		t.Fatalf("expected StartedAt %v, got %v", t3, s.StartedAt)
	}
}

// TestFastFoldRecoveryToleratesTornTail verifies recovery tolerance when the active segment
// ends with a torn line.
func TestFastFoldRecoveryToleratesTornTail(t *testing.T) {
	dir := t.TempDir()
	sj := NewSegmentedJournal(dir, WithSyncOnAppend(false))

	_ = sj.AppendEvent(Event{
		ID:   "sess-valid",
		Kind: KindOpen,
		TS:   time.Now().UTC().Format(time.RFC3339Nano),
	})

	// Simulate torn row append
	f, err := os.OpenFile(sj.ActivePath(), os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("open active: %v", err)
	}
	_, _ = f.Write([]byte(`{"schema":"fak.sessionjournal.v1","kind":"open","id":"sess-torn","ts":"`))
	_ = f.Close()

	recovered, err := sj.FastFoldRecovery()
	if err != nil {
		t.Fatalf("FastFoldRecovery failed on torn active segment: %v", err)
	}

	if len(recovered) != 1 || recovered[0].ID != "sess-valid" {
		t.Fatalf("expected 1 recovered valid session, got: %+v", recovered)
	}
}

// TestBackgroundCompactor verifies that StartBackgroundCompactor periodically
// folds sealed segments.
func TestBackgroundCompactor(t *testing.T) {
	dir := t.TempDir()
	sj := NewSegmentedJournal(dir, WithMaxEntries(5), WithSyncOnAppend(false))

	stop := sj.StartBackgroundCompactor(20 * time.Millisecond)
	defer stop()

	for i := 0; i < 15; i++ {
		_ = sj.AppendEvent(Event{
			ID:   fmt.Sprintf("bg-sess-%d", i),
			Kind: KindOpen,
			TS:   time.Now().UTC().Format(time.RFC3339Nano),
		})
	}

	// Give background compactor time to trigger
	deadline := time.Now().Add(2 * time.Second)
	var cp *Checkpoint
	var err error
	for time.Now().Before(deadline) {
		cp, err = sj.CurrentCheckpoint()
		if err == nil && cp != nil && cp.CompactedSeq > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	if cp == nil || cp.CompactedSeq == 0 {
		t.Fatalf("background compactor did not produce checkpoint in time")
	}
}

// TestStartupFoldLatency500kHistoricalEvents proves that startup fold recovery at 500,000
// historical events finishes under the required 50ms budget (#11181).
func TestStartupFoldLatency500kHistoricalEvents(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 500k fold latency test in short mode")
	}

	dir := t.TempDir()
	sj := NewSegmentedJournal(dir, WithMaxEntries(10000), WithSyncOnAppend(false))

	// Construct a checkpoint representing 500,000 historical events folded into 25,000 unique sessions.
	const historicalSessions = 25000
	sessions := make([]Session, historicalSessions)
	now := time.Now().UTC()
	for i := 0; i < historicalSessions; i++ {
		sessions[i] = Session{
			ID:          fmt.Sprintf("hist-sess-%06d", i),
			StartedAt:   now.Add(-24 * time.Hour),
			LastSeen:    now.Add(-12 * time.Hour),
			Closed:      i%2 == 0,
			CloseReason: "done",
			Model:       "claude-3-7-sonnet",
			PID:         10000 + (i % 1000),
		}
	}

	// Write checkpoint.json
	cp := &Checkpoint{
		Schema:       CheckpointSchema,
		CompactedSeq: 50, // 50 sealed segments folded
		UpdatedAt:    now,
		Sessions:     sessions,
	}
	rawCP, err := json.Marshal(cp)
	if err != nil {
		t.Fatalf("marshal checkpoint: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, DefaultCheckpointName), rawCP, 0644); err != nil {
		t.Fatalf("write checkpoint: %v", err)
	}

	// Add 500 active events to session-journal.active.jsonl
	for i := 0; i < 500; i++ {
		_ = sj.AppendEvent(Event{
			ID:   fmt.Sprintf("active-sess-%04d", i),
			Kind: KindOpen,
			TS:   now.Format(time.RFC3339Nano),
		})
	}

	// Benchmark recovery latency
	start := time.Now()
	recovered, err := sj.FastFoldRecovery()
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("FastFoldRecovery: %v", err)
	}

	t.Logf("FastFoldRecovery for 500k event history completed in %v (recovered %d sessions)", elapsed, len(recovered))

	if elapsed >= 50*time.Millisecond {
		t.Fatalf("FastFoldRecovery exceeded 50ms budget: took %v", elapsed)
	}
	if len(recovered) != historicalSessions+500 {
		t.Fatalf("expected %d sessions, got %d", historicalSessions+500, len(recovered))
	}
}
