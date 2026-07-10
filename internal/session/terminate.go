package session

// terminate.go — the forceful-stop half of the drain/terminate pair (#2758, epic
// #2753). `fak session stop` (OpCancel) DRAINS: the write parks the session at
// Draining and the loop takes the stop at its next TURN BOUNDARY, so the in-flight
// turn always runs to completion. Terminate is the typed, deliberately-different
// verb: the session parks at Terminating, and the running arm is WOKEN mid-turn via
// the level-triggered signal below — it cancels the in-flight model call's context
// and dispatches no further tool call. The distinction is first-class (two
// run-states, two closed stop reasons: DRAINING vs TERMINATED), never an ambiguous
// single "stop".

// TerminateSignal returns the channel that is CLOSED the moment trace enters
// Terminating — the loop-side wake-up runArm selects on to cancel in-flight work at
// the next safe point. It is level-triggered, not one-shot: a trace already
// Terminating (or already finalized Stopped with the TERMINATED reason) gets an
// already-closed channel, so a late registration never blocks on a signal that
// fired before it arrived. Successive calls for a live trace return the SAME
// channel. A nil receiver returns nil — a nil channel blocks forever in a select,
// so a loop with no table wired behaves byte-identically to the pre-terminate path.
func (t *Table) TerminateSignal(trace string) <-chan struct{} {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)
	if cur.Run == Terminating || (cur.Run == Stopped && cur.Reason == ReasonTerminated) {
		ch := make(chan struct{})
		close(ch)
		return ch
	}
	if t.terminateSignals == nil {
		t.terminateSignals = map[string]chan struct{}{}
	}
	ch, ok := t.terminateSignals[trace]
	if !ok {
		ch = make(chan struct{})
		t.terminateSignals[trace] = ch
	}
	return ch
}

// signalTerminateLocked closes (and clears) trace's terminate channel. Caller holds
// the lock. It is called from the transition write paths (Transition, CompareAndSet)
// when a session enters Terminating. Idempotent: the map delete makes a second
// signal a cheap miss, and late TerminateSignal callers are handed a pre-closed
// channel by the level-trigger check above.
func (t *Table) signalTerminateLocked(trace string) {
	if ch, ok := t.terminateSignals[trace]; ok {
		close(ch)
		delete(t.terminateSignals, trace)
	}
}
