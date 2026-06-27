package session

import "context"

// resume.go — the live-resume loop for a PAUSED session (issue #916, epic #912 "one
// machine"). In the unified agent-session / serving-admission model a Paused session is a
// PREEMPTED sequence: its KV should be swapped back WARM on resume, not cold re-prefilled.
// Today the swap-in is a pure analogy — RunState(Paused) exists, Transition flips it back to
// Running, but nothing BLOCKS on Paused and re-admits at a boundary, and nothing carries the
// warm-vs-cold decision to whoever owns the KV mover.
//
// This file is the in-package half of that loop: the BLOCK-on-Paused / re-admit-at-boundary
// primitive plus the typed warm/cold resume decision. It deliberately does NOT call the KV
// mover — internal/model's KVCache.Clone and internal/cachemeta's MoveTo(KVRestore) live
// outside this foundation leaf (which imports only the stdlib + internal/lifecycle). The host
// wires a WarmKVSplicer that performs the actual splice; this loop decides WHEN to splice and
// degrades to cold re-prefill when the splice is unavailable, so correctness never depends on
// warm KV being present.
//
// THE CONTRACT.
//   - WaitResume(ctx, trace) blocks while the session is Paused and returns the instant it is
//     transitioned back to Running (or the session is stopped, or ctx is cancelled). A session
//     that is already non-Paused returns immediately — the wait is for the HOLD, not a turn.
//   - On the Paused->Running edge it consults the wired WarmKVSplicer: if it reports warm KV is
//     available and spliced, the resumed turn REUSES it; otherwise the verdict says cold, and
//     the caller re-prefills exactly as today.
//   - Re-admission is at a turn BOUNDARY: WaitResume returns the verdict, and the caller's next
//     Decide is the boundary at which the (warm or cold) session re-enters. The loop never
//     resumes mid-decode.

// ResumeMode names how a Paused->Running resume re-admits a session. Warm means the wired
// splicer reported the session's KV was reattached (KVCache.Clone / MoveTo(KVRestore) on the
// host) so the resumed turn reuses it; Cold means no warm KV was available and the caller
// re-prefills as it does today. The zero value is Cold — the safe default, so a resume with no
// splicer wired (or a splicer that declines) is always the correct, if slower, cold path.
type ResumeMode uint8

const (
	// ResumeCold is the fallback: warm KV was unavailable (no splicer, the splicer declined,
	// or the session was not actually held), so the caller cold re-prefills. The zero value.
	ResumeCold ResumeMode = iota
	// ResumeWarm means the wired WarmKVSplicer reattached the session's KV, so the resumed
	// turn reuses it instead of re-prefilling.
	ResumeWarm
)

// String renders a ResumeMode as its lowercase wire token; an out-of-range value renders
// "unknown" rather than panicking.
func (m ResumeMode) String() string {
	switch m {
	case ResumeWarm:
		return "warm"
	case ResumeCold:
		return "cold"
	}
	return "unknown"
}

// ResumeVerdict is what WaitResume returns once a Paused session leaves the hold. Resumed is
// true when the session was transitioned back to a live (Running/Throttled) state and should
// re-admit at the next boundary; it is false when the wait ended because the session was
// STOPPED or the context was cancelled (the caller ends the loop instead of re-admitting).
// Mode is warm/cold (meaningful only when Resumed); State is the drive record observed at the
// resume edge. Reason carries why a non-resume wait ended (a closed token).
type ResumeVerdict struct {
	Resumed bool
	Mode    ResumeMode
	State   State
	Reason  string
}

// WarmKVSplicer is the host-wired seam that performs the actual warm-KV reattach on a
// Paused->Running resume. The session package never imports the KV mover (internal/model /
// internal/cachemeta); the host implements this to call KVCache.Clone / MoveTo(KVRestore) and
// returns true iff warm KV was available AND spliced, so the resumed turn may reuse it. It
// returns false to decline (no warm KV held, eviction happened while paused, or any error) —
// the loop then degrades to cold re-prefill. It is given the resume-edge State so it can key
// the splice on the trace / continuation lineage. A nil splicer always resumes Cold.
type WarmKVSplicer func(State) bool

// WatchResumeSplice wires the warm-KV splice seam. splicer==nil clears it (every resume is
// then Cold — the byte-identical pre-splice path). Safe to call on a live table; a nil
// receiver is a no-op.
func (t *Table) WatchResumeSplice(splicer WarmKVSplicer) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.spliceFn = splicer
	t.mu.Unlock()
}

// WaitResume blocks while trace is Paused and returns when it is transitioned back to a live
// state, stopped, or ctx is cancelled. It is the live-resume loop a served session runs at a
// turn boundary: instead of cold re-attaching, it parks on the Paused hold and is woken the
// instant an operator resumes it, then re-admits — warm if the wired splicer reattached KV,
// cold otherwise.
//
// A session that is NOT Paused returns immediately with Resumed=true (Running/Throttled) or
// Resumed=false (terminal) — the wait is for the operator HOLD, never for a turn. A nil
// receiver returns an immediate cold-resume default so a loop with no table wired behaves
// byte-identically to the pre-resume path.
func (t *Table) WaitResume(ctx context.Context, trace string) ResumeVerdict {
	if t == nil {
		return ResumeVerdict{Resumed: true, Mode: ResumeCold, State: DefaultState(trace)}
	}
	for {
		t.mu.Lock()
		cur := t.getLocked(trace)
		switch cur.Run {
		case Paused:
			// Register a one-shot waiter and block OUTSIDE the lock. A resume transition (or a
			// stop) closes the channel; the loop then re-reads the live state and decides.
			ch := t.registerResumeWaiterLocked(trace)
			t.mu.Unlock()
			select {
			case <-ch:
				// Woken by a transition off Paused (resume or stop) — loop and re-read.
				continue
			case <-ctx.Done():
				t.dropResumeWaiter(trace, ch)
				return ResumeVerdict{Resumed: false, State: cur, Reason: ReasonResumeCancelled}
			}
		case Stopped, Draining:
			// The hold ended in a stop, not a resume: end the loop, do not re-admit.
			t.mu.Unlock()
			return ResumeVerdict{Resumed: false, State: cur, Reason: cur.stopReasonOr(canonicalReason(cur.Run))}
		default:
			// Running/Throttled: live. Decide warm vs cold via the wired splicer, then re-admit
			// at the next boundary (the caller's next Decide). The splicer runs after the lock
			// is released so a slow KV mover never stalls other sessions.
			splicer := t.spliceFn
			t.mu.Unlock()
			mode := ResumeCold
			if splicer != nil && splicer(cur) {
				mode = ResumeWarm
			}
			return ResumeVerdict{Resumed: true, Mode: mode, State: cur}
		}
	}
}

// registerResumeWaiterLocked adds a one-shot wake channel for trace and returns it. Caller
// holds the lock. The channel is closed by signalResumeLocked when the session transitions off
// Paused, waking every parked WaitResume for that trace exactly once.
func (t *Table) registerResumeWaiterLocked(trace string) chan struct{} {
	if t.resumeWaiters == nil {
		t.resumeWaiters = map[string][]chan struct{}{}
	}
	ch := make(chan struct{})
	t.resumeWaiters[trace] = append(t.resumeWaiters[trace], ch)
	return ch
}

// signalResumeLocked wakes (and clears) every waiter parked on trace by closing each channel.
// Caller holds the lock. It is called from the transition write path whenever a session leaves
// Paused — to Running/Throttled (a resume) OR to a stop (so a parked waiter is not orphaned on
// a paused->stopped move). Idempotent: a trace with no waiters is a cheap map miss.
func (t *Table) signalResumeLocked(trace string) {
	ws := t.resumeWaiters[trace]
	if len(ws) == 0 {
		return
	}
	for _, ch := range ws {
		close(ch)
	}
	delete(t.resumeWaiters, trace)
}

// dropResumeWaiter removes one cancelled waiter channel from trace's list (the ctx-cancelled
// path) so a never-signalled waiter does not leak in the map. Takes the lock itself — it runs
// after the select, off the registration path. A channel already cleared by signalResumeLocked
// is a no-op.
func (t *Table) dropResumeWaiter(trace string, ch chan struct{}) {
	t.mu.Lock()
	defer t.mu.Unlock()
	ws := t.resumeWaiters[trace]
	for i, w := range ws {
		if w == ch {
			t.resumeWaiters[trace] = append(ws[:i], ws[i+1:]...)
			break
		}
	}
	if len(t.resumeWaiters[trace]) == 0 {
		delete(t.resumeWaiters, trace)
	}
}
