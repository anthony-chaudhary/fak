package main

import (
	"bytes"
	"context"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// snapshotSyncDrainSeams saves the drain seams and restores them at test end, so each test can
// inject a window/stranded/flush/clock without leaking state into its siblings.
func snapshotSyncDrainSeams(t *testing.T) {
	t.Helper()
	win, strand, flush, now, cap := syncDrainWindow, syncDrainStranded, syncDrainFlush, syncDrainNow, syncCaptureSource
	syncCaptureSource = func(string) (string, error) {
		return "0f1e2d3c4b5a60718293a4b5c6d7e8f90a1b2c3d", nil
	}
	t.Cleanup(func() {
		syncDrainWindow, syncDrainStranded, syncDrainFlush, syncDrainNow, syncCaptureSource = win, strand, flush, now, cap
	})
}

func decodeSyncDrainReport(t *testing.T, b []byte) syncDrainReport {
	t.Helper()
	var rep syncDrainReport
	if err := json.Unmarshal(b, &rep); err != nil {
		t.Fatalf("drain JSON did not decode: %v\n%s", err, b)
	}
	return rep
}

// A red pre-push window must QUEUE the stranded commit and NEVER push it.
func TestSyncDrainRedWindowQueuesNotPushed(t *testing.T) {
	snapshotSyncDrainSeams(t)
	qp := filepath.Join(t.TempDir(), "queue.json")

	flushCalls := 0
	syncDrainWindow = func(_ context.Context, _ syncDrainConfig) syncDrainWindowVerdict {
		return syncDrainWindowVerdict{
			Green:        false,
			Reason:       "trunk build not green: TRUNK_WOULD_NOT_COMPILE",
			BuildVerdict: "TRUNK_WOULD_NOT_COMPILE",
			PeerState:    "ahead",
		}
	}
	syncDrainStranded = func(_ context.Context, _ syncDrainConfig) ([]syncDrainEntry, error) {
		return []syncDrainEntry{{SHA: "abc1231111111111111111111111111111111111", Subject: "fix: hermetic thing"}}, nil
	}
	syncDrainFlush = func(_ context.Context, _ syncDrainConfig) (safesync.PushResult, error) {
		flushCalls++
		return safesync.PushResult{Pushed: true}, nil
	}
	syncDrainNow = func() int64 { return 1000 }

	var out, errb bytes.Buffer
	code := runSyncDrain(&out, &errb, syncDrainConfig{queuePath: qp, asJSON: true})
	if code != syncExitRefused {
		t.Fatalf("exit = %d, want %d (refused/held); stderr=%s", code, syncExitRefused, errb.String())
	}
	if flushCalls != 0 {
		t.Fatalf("flush push called %d times on a RED window, want 0", flushCalls)
	}

	rep := decodeSyncDrainReport(t, out.Bytes())
	if rep.Verdict != "QUEUED" {
		t.Fatalf("verdict = %q, want QUEUED", rep.Verdict)
	}
	if len(rep.Queued) != 1 || rep.Queued[0].SHA != "abc1231111111111111111111111111111111111" {
		t.Fatalf("queued = %+v, want the one stranded commit", rep.Queued)
	}
	if rep.Queued[0].RefusalReason == "" {
		t.Fatalf("queued entry recorded no refusal reason: %+v", rep.Queued[0])
	}
	if rep.Attempts != 1 {
		t.Fatalf("attempts = %d, want 1", rep.Attempts)
	}
	if rep.NextRetryUnix != 1000+syncDrainBackoffBaseSec {
		t.Fatalf("next_retry_unix = %d, want %d", rep.NextRetryUnix, 1000+syncDrainBackoffBaseSec)
	}

	// The queue must have been persisted, not just reported.
	persisted, err := loadSyncDrainQueue(qp)
	if err != nil {
		t.Fatalf("reload queue: %v", err)
	}
	if len(persisted.Entries) != 1 {
		t.Fatalf("persisted queue has %d entries, want 1", len(persisted.Entries))
	}
}

// A green window must FLUSH the queue in exactly one push and clear it.
func TestSyncDrainGreenWindowSingleFlush(t *testing.T) {
	snapshotSyncDrainSeams(t)
	qp := filepath.Join(t.TempDir(), "queue.json")

	flushCalls := 0
	syncDrainWindow = func(_ context.Context, _ syncDrainConfig) syncDrainWindowVerdict {
		return syncDrainWindowVerdict{Green: true, BuildVerdict: "OK", PeerState: "ahead"}
	}
	syncDrainStranded = func(_ context.Context, _ syncDrainConfig) ([]syncDrainEntry, error) {
		return []syncDrainEntry{
			{SHA: "aaa0000000000000000000000000000000000000", Subject: "fix: one"},
			{SHA: "bbb0000000000000000000000000000000000000", Subject: "feat: two"},
		}, nil
	}
	syncDrainFlush = func(_ context.Context, _ syncDrainConfig) (safesync.PushResult, error) {
		flushCalls++
		return safesync.PushResult{Pushed: true, Attempts: 1}, nil
	}
	syncDrainNow = func() int64 { return 2000 }

	var out, errb bytes.Buffer
	code := runSyncDrain(&out, &errb, syncDrainConfig{queuePath: qp, asJSON: true})
	if code != syncExitOK {
		t.Fatalf("exit = %d, want %d (ok); stderr=%s", code, syncExitOK, errb.String())
	}
	if flushCalls != 1 {
		t.Fatalf("flush push called %d times on a GREEN window, want exactly 1", flushCalls)
	}

	rep := decodeSyncDrainReport(t, out.Bytes())
	if rep.Verdict != "FLUSHED" {
		t.Fatalf("verdict = %q, want FLUSHED", rep.Verdict)
	}
	if len(rep.Flushed) != 2 {
		t.Fatalf("flushed = %d commits, want 2", len(rep.Flushed))
	}

	// A flush clears the queue: nothing left stranded.
	persisted, err := loadSyncDrainQueue(qp)
	if err != nil {
		t.Fatalf("reload queue: %v", err)
	}
	if len(persisted.Entries) != 0 || persisted.Attempts != 0 {
		t.Fatalf("queue not cleared after flush: %+v", persisted)
	}
}

// TestSyncDrainRedThenGreenLandsQueuedCommit tests the multi-tick transition:
// Round 1 (RED): a stranded commit is queued and push is withheld.
// Round 2 (GREEN): the queued commit is flushed in one push and cleared from the queue.
func TestSyncDrainRedThenGreenLandsQueuedCommit(t *testing.T) {
	snapshotSyncDrainSeams(t)
	qp := filepath.Join(t.TempDir(), "queue.json")

	isGreen := false
	syncDrainWindow = func(_ context.Context, _ syncDrainConfig) syncDrainWindowVerdict {
		if !isGreen {
			return syncDrainWindowVerdict{
				Green:        false,
				Reason:       "trunk build not green: TRUNK_WOULD_NOT_COMPILE",
				BuildVerdict: "TRUNK_WOULD_NOT_COMPILE",
				PeerState:    "behind",
			}
		}
		return syncDrainWindowVerdict{Green: true, BuildVerdict: "OK", PeerState: "in-sync"}
	}

	strandedList := []syncDrainEntry{{SHA: "abc1231111111111111111111111111111111111", Subject: "fix: queued commit"}}
	syncDrainStranded = func(_ context.Context, _ syncDrainConfig) ([]syncDrainEntry, error) {
		return strandedList, nil
	}

	flushCalls := 0
	syncDrainFlush = func(_ context.Context, _ syncDrainConfig) (safesync.PushResult, error) {
		flushCalls++
		return safesync.PushResult{Pushed: true, Attempts: 1}, nil
	}

	// Round 1: RED window -> commit must be queued, no push.
	syncDrainNow = func() int64 { return 1000 }
	var out1, err1 bytes.Buffer
	code1 := runSyncDrain(&out1, &err1, syncDrainConfig{queuePath: qp, asJSON: true})
	if code1 != syncExitRefused {
		t.Fatalf("round1 exit = %d, want %d (refused/queued); stderr=%s", code1, syncExitRefused, err1.String())
	}
	if flushCalls != 0 {
		t.Fatalf("round1 flushCalls = %d, want 0 on red window", flushCalls)
	}
	rep1 := decodeSyncDrainReport(t, out1.Bytes())
	if rep1.Verdict != "QUEUED" || len(rep1.Queued) != 1 {
		t.Fatalf("round1 verdict = %q queued = %+v, want QUEUED with 1 entry", rep1.Verdict, rep1.Queued)
	}

	// Round 2: Window turns GREEN, no newly stranded commits -> queued commit is flushed.
	isGreen = true
	strandedList = nil
	syncDrainNow = func() int64 { return 2000 }
	var out2, err2 bytes.Buffer
	code2 := runSyncDrain(&out2, &err2, syncDrainConfig{queuePath: qp, asJSON: true})
	if code2 != syncExitOK {
		t.Fatalf("round2 exit = %d, want %d (ok/flushed); stderr=%s", code2, syncExitOK, err2.String())
	}
	if flushCalls != 1 {
		t.Fatalf("round2 flushCalls = %d, want exactly 1 flush", flushCalls)
	}
	rep2 := decodeSyncDrainReport(t, out2.Bytes())
	if rep2.Verdict != "FLUSHED" || len(rep2.Flushed) != 1 || rep2.Flushed[0].SHA != "abc1231111111111111111111111111111111111" {
		t.Fatalf("round2 verdict = %q flushed = %+v, want FLUSHED with landed commit", rep2.Verdict, rep2.Flushed)
	}

	// Queue must now be cleared.
	persisted, err := loadSyncDrainQueue(qp)
	if err != nil {
		t.Fatalf("reload queue: %v", err)
	}
	if len(persisted.Entries) != 0 {
		t.Fatalf("persisted queue not empty after green flush: %+v", persisted)
	}
}

// TestSyncDrainFlushPinsCapturedSource is #4221's drain-side witness: a green-window flush
// pushes the CAPTURED source object, threaded through as cfg.sourceSHA, so the queue is cleared
// only for commits reachable from that immutable SHA. HEAD moving between the stranded read and
// the flush cannot make drain publish (or clear) a commit that merely rode along on the mutable
// branch tip — the flush is pinned to the object drain captured, not to whatever HEAD is now.
func TestSyncDrainFlushPinsCapturedSource(t *testing.T) {
	snapshotSyncDrainSeams(t)
	const captured = "0f1e2d3c4b5a60718293a4b5c6d7e8f90a1b2c3d"

	oldCapture := syncCaptureSource
	syncCaptureSource = func(string) (string, error) { return captured, nil }
	t.Cleanup(func() { syncCaptureSource = oldCapture })

	flushSource := "<never-called>"
	syncDrainWindow = func(context.Context, syncDrainConfig) syncDrainWindowVerdict {
		return syncDrainWindowVerdict{Green: true, BuildVerdict: "OK", PeerState: "ahead"}
	}
	syncDrainStranded = func(context.Context, syncDrainConfig) ([]syncDrainEntry, error) {
		return []syncDrainEntry{{SHA: "aaa0000000000000000000000000000000000000", Subject: "fix: one"}}, nil
	}
	syncDrainFlush = func(_ context.Context, cfg syncDrainConfig) (safesync.PushResult, error) {
		flushSource = cfg.sourceSHA
		return safesync.PushResult{Pushed: true, Attempts: 1}, nil
	}
	syncDrainNow = func() int64 { return 3000 }

	var out, errb bytes.Buffer
	code := runSyncDrain(&out, &errb, syncDrainConfig{queuePath: filepath.Join(t.TempDir(), "queue.json"), asJSON: true})
	if code != syncExitOK {
		t.Fatalf("exit=%d stderr=%q, want ok", code, errb.String())
	}
	if flushSource != captured {
		t.Fatalf("drain flush sourceSHA = %q, want the captured object %q — the flush is not pinned to the immutable source", flushSource, captured)
	}
}

// With nothing stranded and an empty queue, drain is a clean idle no-op — no push, no backoff,
// regardless of the window color.
func TestSyncDrainNoWorkIsIdle(t *testing.T) {
	snapshotSyncDrainSeams(t)
	qp := filepath.Join(t.TempDir(), "queue.json")

	flushCalls := 0
	syncDrainWindow = func(_ context.Context, _ syncDrainConfig) syncDrainWindowVerdict {
		// Even a RED window must not fabricate a QUEUED verdict when there is nothing to hold.
		return syncDrainWindowVerdict{Green: false, Reason: "trunk build not green: TRUNK_WOULD_NOT_COMPILE", BuildVerdict: "TRUNK_WOULD_NOT_COMPILE", PeerState: "in-sync"}
	}
	syncDrainStranded = func(_ context.Context, _ syncDrainConfig) ([]syncDrainEntry, error) {
		return nil, nil
	}
	syncDrainFlush = func(_ context.Context, _ syncDrainConfig) (safesync.PushResult, error) {
		flushCalls++
		return safesync.PushResult{Pushed: true}, nil
	}
	syncDrainNow = func() int64 { return 3000 }

	var out, errb bytes.Buffer
	code := runSyncDrain(&out, &errb, syncDrainConfig{queuePath: qp, asJSON: true})
	if code != syncExitOK {
		t.Fatalf("exit = %d, want %d (ok); stderr=%s", code, syncExitOK, errb.String())
	}
	if flushCalls != 0 {
		t.Fatalf("flush push called %d times with no work, want 0", flushCalls)
	}
	rep := decodeSyncDrainReport(t, out.Bytes())
	if rep.Verdict != "IDLE" {
		t.Fatalf("verdict = %q, want IDLE", rep.Verdict)
	}
	if rep.Attempts != 0 || rep.NextRetryUnix != 0 {
		t.Fatalf("idle tick bumped backoff: attempts=%d next=%d", rep.Attempts, rep.NextRetryUnix)
	}
}

// While the window stays red, the backoff must GROW (not blind-retry) and no push must fire.
func TestSyncDrainBackoffGrowsWhileRed(t *testing.T) {
	snapshotSyncDrainSeams(t)
	qp := filepath.Join(t.TempDir(), "queue.json")

	flushCalls := 0
	syncDrainWindow = func(_ context.Context, _ syncDrainConfig) syncDrainWindowVerdict {
		return syncDrainWindowVerdict{Green: false, Reason: "peer merge in flight: behind", BuildVerdict: "OK", PeerState: "behind"}
	}
	// Same stranded commit both rounds — also exercises SHA-dedup.
	syncDrainStranded = func(_ context.Context, _ syncDrainConfig) ([]syncDrainEntry, error) {
		return []syncDrainEntry{{SHA: "ccc0000000000000000000000000000000000000", Subject: "fix: stranded"}}, nil
	}
	syncDrainFlush = func(_ context.Context, _ syncDrainConfig) (safesync.PushResult, error) {
		flushCalls++
		return safesync.PushResult{Pushed: true}, nil
	}

	// Round 1.
	syncDrainNow = func() int64 { return 1000 }
	var out1, err1 bytes.Buffer
	if code := runSyncDrain(&out1, &err1, syncDrainConfig{queuePath: qp, asJSON: true}); code != syncExitRefused {
		t.Fatalf("round1 exit = %d, want %d; stderr=%s", code, syncExitRefused, err1.String())
	}
	rep1 := decodeSyncDrainReport(t, out1.Bytes())

	// Round 2 (still red, later clock).
	syncDrainNow = func() int64 { return 2000 }
	var out2, err2 bytes.Buffer
	if code := runSyncDrain(&out2, &err2, syncDrainConfig{queuePath: qp, asJSON: true}); code != syncExitRefused {
		t.Fatalf("round2 exit = %d, want %d; stderr=%s", code, syncExitRefused, err2.String())
	}
	rep2 := decodeSyncDrainReport(t, out2.Bytes())

	if flushCalls != 0 {
		t.Fatalf("flush push fired %d times across two RED rounds, want 0", flushCalls)
	}
	if rep2.Attempts <= rep1.Attempts {
		t.Fatalf("attempts did not grow: round1=%d round2=%d", rep1.Attempts, rep2.Attempts)
	}
	if rep2.BackoffSeconds <= rep1.BackoffSeconds {
		t.Fatalf("backoff did not grow: round1=%ds round2=%ds (want strictly larger, not a blind immediate retry)",
			rep1.BackoffSeconds, rep2.BackoffSeconds)
	}
	// Dedup: the same stranded SHA must not be double-queued.
	if len(rep2.Queued) != 1 {
		t.Fatalf("queued = %d, want 1 (SHA dedup across rounds)", len(rep2.Queued))
	}
}
