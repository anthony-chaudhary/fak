package session

import (
	"context"
	"testing"
	"time"
)

// TestWaitResumeWarmSplice proves the live-resume loop: a Paused session BLOCKS in
// WaitResume, an operator Transition(Paused->Running) wakes it, and the wired WarmKVSplicer
// makes the resume WARM (the resumed turn reuses KV instead of cold re-prefilling). This is the
// #916 acceptance: a Paused session blocks, Transition(Paused->Running) re-admits it, and warm
// KV is spliced when available.
func TestWaitResumeWarmSplice(t *testing.T) {
	tbl := NewTable()
	const trace = "gw-resume"

	var splicedFor string
	tbl.WatchResumeSplice(func(st State) SpliceResult {
		splicedFor = st.TraceID
		return SpliceResult{Warm: true} // warm KV available and reattached
	})

	// Park the session in Paused, then block on it in a goroutine.
	if _, ok := tbl.Transition(trace, Paused, "operator-hold"); !ok {
		t.Fatal("pause transition rejected")
	}
	verdicts := make(chan ResumeVerdict, 1)
	go func() { verdicts <- tbl.WaitResume(context.Background(), trace) }()

	// The waiter must still be parked (the session is held). Give the goroutine a moment to
	// register, then confirm no verdict has arrived.
	select {
	case v := <-verdicts:
		t.Fatalf("WaitResume returned %+v while still Paused; it must block on the hold", v)
	case <-time.After(20 * time.Millisecond):
	}

	// Resume: the operator flips Paused->Running. This must wake the parked waiter.
	if _, ok := tbl.Transition(trace, Running, ""); !ok {
		t.Fatal("resume transition rejected")
	}
	select {
	case v := <-verdicts:
		if !v.Resumed {
			t.Fatalf("verdict = %+v, want Resumed", v)
		}
		if v.Mode != ResumeWarm {
			t.Fatalf("verdict mode = %v, want warm (the splicer reattached KV)", v.Mode)
		}
		if splicedFor != trace {
			t.Fatalf("splicer ran for %q, want %q", splicedFor, trace)
		}
	case <-time.After(time.Second):
		t.Fatal("WaitResume did not wake on Paused->Running")
	}
}

// TestWaitResumeColdFallback proves the degrade-safe path: with NO splicer wired (or a splicer
// that declines), a resume re-admits COLD — the caller cold re-prefills, exactly as today, so
// correctness never depends on warm KV being present.
func TestWaitResumeColdFallback(t *testing.T) {
	tbl := NewTable()
	const trace = "gw-cold"
	tbl.Transition(trace, Paused, "hold")

	verdicts := make(chan ResumeVerdict, 1)
	go func() { verdicts <- tbl.WaitResume(context.Background(), trace) }()
	time.Sleep(10 * time.Millisecond)
	tbl.Transition(trace, Running, "")

	select {
	case v := <-verdicts:
		if !v.Resumed || v.Mode != ResumeCold {
			t.Fatalf("verdict = %+v, want Resumed cold (no splicer wired)", v)
		}
	case <-time.After(time.Second):
		t.Fatal("WaitResume did not wake on resume")
	}

	// A splicer that DECLINES (warm KV evicted while paused) also falls back to cold.
	tbl.WatchResumeSplice(func(State) SpliceResult { return SpliceResult{} })
	tbl.Transition(trace, Paused, "hold-2")
	go func() { verdicts <- tbl.WaitResume(context.Background(), trace) }()
	time.Sleep(10 * time.Millisecond)
	tbl.Transition(trace, Running, "")
	select {
	case v := <-verdicts:
		if !v.Resumed || v.Mode != ResumeCold {
			t.Fatalf("declining-splicer verdict = %+v, want Resumed cold", v)
		}
	case <-time.After(time.Second):
		t.Fatal("WaitResume did not wake on the second resume")
	}
}

// TestWaitResumeNotPausedReturnsImmediately proves the wait is for the HOLD, not a turn: a live
// session returns Resumed at once, and a terminal session returns not-resumed (the caller ends
// the loop instead of re-admitting).
func TestWaitResumeNotPausedReturnsImmediately(t *testing.T) {
	tbl := NewTable()

	// A never-paused (default Running) session resumes immediately.
	v := tbl.WaitResume(context.Background(), "gw-live")
	if !v.Resumed {
		t.Fatalf("live-session verdict = %+v, want immediate Resumed", v)
	}

	// A stopped session does NOT resume — the loop ends.
	const term = "gw-term"
	tbl.Transition(term, Stopped, ReasonStopped)
	v = tbl.WaitResume(context.Background(), term)
	if v.Resumed {
		t.Fatalf("stopped-session verdict = %+v, want not Resumed", v)
	}
}

// TestWaitResumePausedToStopEndsLoop proves a Paused->Stopped move wakes the parked waiter with
// a NON-resume verdict (the held session was killed, not resumed) — a waiter is never orphaned
// when a paused session is stopped instead of resumed.
func TestWaitResumePausedToStopEndsLoop(t *testing.T) {
	tbl := NewTable()
	const trace = "gw-killed"
	tbl.Transition(trace, Paused, "hold")

	verdicts := make(chan ResumeVerdict, 1)
	go func() { verdicts <- tbl.WaitResume(context.Background(), trace) }()
	time.Sleep(10 * time.Millisecond)
	tbl.Transition(trace, Stopped, ReasonStopped)

	select {
	case v := <-verdicts:
		if v.Resumed {
			t.Fatalf("verdict = %+v, want not Resumed (session stopped, not resumed)", v)
		}
		if v.Reason != ReasonStopped {
			t.Fatalf("verdict reason = %q, want %q", v.Reason, ReasonStopped)
		}
	case <-time.After(time.Second):
		t.Fatal("WaitResume did not wake on Paused->Stopped")
	}
}

// TestWaitResumeContextCancel proves a cancelled context releases a parked waiter (no leak) with
// a RESUME_CANCELLED reason — the caller did not get a resume and ends the loop.
func TestWaitResumeContextCancel(t *testing.T) {
	tbl := NewTable()
	const trace = "gw-cancel"
	tbl.Transition(trace, Paused, "hold")

	ctx, cancel := context.WithCancel(context.Background())
	verdicts := make(chan ResumeVerdict, 1)
	go func() { verdicts <- tbl.WaitResume(ctx, trace) }()
	time.Sleep(10 * time.Millisecond)
	cancel()

	select {
	case v := <-verdicts:
		if v.Resumed || v.Reason != ReasonResumeCancelled {
			t.Fatalf("verdict = %+v, want not Resumed with reason %s", v, ReasonResumeCancelled)
		}
	case <-time.After(time.Second):
		t.Fatal("WaitResume did not release on context cancel")
	}

	// The cancelled waiter must be dropped, not leaked.
	tbl.mu.Lock()
	left := len(tbl.resumeWaiters[trace])
	tbl.mu.Unlock()
	if left != 0 {
		t.Fatalf("resumeWaiters[%q] = %d after cancel, want 0 (no leak)", trace, left)
	}
}

// TestWaitResumeCASResume proves the operator --if-rev (CompareAndSet) resume path also wakes a
// parked waiter — a Paused->Running flip applied with a Rev guard is a resume too.
func TestWaitResumeCASResume(t *testing.T) {
	tbl := NewTable()
	const trace = "gw-cas"
	paused, ok := tbl.Transition(trace, Paused, "hold")
	if !ok {
		t.Fatal("pause rejected")
	}

	verdicts := make(chan ResumeVerdict, 1)
	go func() { verdicts <- tbl.WaitResume(context.Background(), trace) }()
	time.Sleep(10 * time.Millisecond)

	// Resume via CAS at the paused Rev.
	if _, ok := tbl.CompareAndSet(trace, paused.Rev, State{Run: Running}); !ok {
		t.Fatal("CAS resume rejected")
	}
	select {
	case v := <-verdicts:
		if !v.Resumed {
			t.Fatalf("CAS resume verdict = %+v, want Resumed", v)
		}
	case <-time.After(time.Second):
		t.Fatal("WaitResume did not wake on a CAS resume")
	}
}

// TestTableResumeAllWakesAllPausedSessions proves Table.ResumeAll transitions every paused session
// to Running, wakes each WaitResume waiter, preserves non-paused sessions, and behaves idempotently.
func TestTableResumeAllWakesAllPausedSessions(t *testing.T) {
	tbl := NewTable()
	traces := []string{"sess-1", "sess-2", "sess-3"}
	for _, tr := range traces {
		if _, ok := tbl.Transition(tr, Paused, "hold"); !ok {
			t.Fatalf("pause %s rejected", tr)
		}
	}
	// Add an unpaused session and a stopped session.
	if _, ok := tbl.Transition("sess-running", Running, ""); !ok {
		t.Fatal("running transition rejected")
	}
	if _, ok := tbl.Transition("sess-stopped", Stopped, "finished"); !ok {
		t.Fatal("stopped transition rejected")
	}

	verdicts := make(map[string]chan ResumeVerdict)
	for _, tr := range traces {
		ch := make(chan ResumeVerdict, 1)
		verdicts[tr] = ch
		trCopy := tr
		go func() { ch <- tbl.WaitResume(context.Background(), trCopy) }()
	}
	time.Sleep(15 * time.Millisecond)

	// Resume all.
	resumed := tbl.ResumeAll("operator batch resume")
	if len(resumed) != len(traces) {
		t.Fatalf("ResumeAll returned %d sessions, want %d", len(resumed), len(traces))
	}
	for i, st := range resumed {
		if st.TraceID != traces[i] {
			t.Errorf("resumed[%d].TraceID = %q, want %q", i, st.TraceID, traces[i])
		}
		if st.Run != Running {
			t.Errorf("resumed[%d].Run = %v, want Running", i, st.Run)
		}
		if st.Reason != "" {
			t.Errorf("resumed[%d].Reason = %q, want empty", i, st.Reason)
		}
	}

	// Verify each parked WaitResume was woken.
	for _, tr := range traces {
		select {
		case v := <-verdicts[tr]:
			if !v.Resumed {
				t.Errorf("%s: verdict = %+v, want Resumed=true", tr, v)
			}
			if v.State.Run != Running {
				t.Errorf("%s: state run = %v, want Running", tr, v.State.Run)
			}
		case <-time.After(time.Second):
			t.Fatalf("%s: WaitResume did not wake on ResumeAll", tr)
		}
	}

	// Verify unpaused and stopped sessions were untouched.
	if cur := tbl.Get("sess-running"); cur.Run != Running {
		t.Errorf("sess-running run = %v, want Running", cur.Run)
	}
	if cur := tbl.Get("sess-stopped"); cur.Run != Stopped {
		t.Errorf("sess-stopped run = %v, want Stopped", cur.Run)
	}

	// Calling ResumeAll again when no sessions are paused returns empty.
	second := tbl.ResumeAll("")
	if len(second) != 0 {
		t.Fatalf("second ResumeAll returned %d sessions, want 0", len(second))
	}
}
