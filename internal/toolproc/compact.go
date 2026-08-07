package toolproc

// CompactJournal returns a bounded subset of events safe to persist as the new
// journal — and, critically, one that Fold still accepts. It keeps every
// un-exited spawn (a live process whose liveness a future hook firing must still
// resolve, and which can be arbitrarily old) plus the last tailKeep events as a
// recent margin, preserving original order. Events of fully terminal CallIDs (a
// spawn paired with an exit or kill) that lie ENTIRELY outside the tail window
// are dropped: they cannot affect any future identity, liveness, or bg-bridge
// resolution. This bounds the journal so the per-firing full parse stays
// O(bounded), not O(all-history) — the root cause behind the guard-hook O(n^2)
// latency regression (#3032).
//
// The tail window can begin partway through a call's history — an exit, kill, or
// pulse whose spawn fell just before the cut. Keeping that trailing event while
// dropping its spawn would orphan it, and Fold rejects an orphaned event ("exit
// for unknown call"), refusing the very journal this function promises is safe
// to persist. So a spawn is retained not only when its call is still live but
// whenever the call has ANY event inside the tail window — a spawn always
// precedes its call's other events, so retaining it restores fold-cleanliness
// across the boundary while still dropping calls that lie wholly behind it.
func CompactJournal(events []Event, tailKeep int) []Event {
	if tailKeep < 0 {
		tailKeep = 0
	}
	if tailKeep >= len(events) {
		return events
	}
	terminal := make(map[string]bool)
	for _, ev := range events {
		if ev.Kind == EvExit || ev.Kind == EvKill {
			terminal[ev.CallID] = true
		}
	}
	tailStart := len(events) - tailKeep
	// CallIDs represented in the tail window: their spawn must survive even if it
	// sits before the cut, or the tail's trailing event is orphaned (Fold-fatal).
	inTail := make(map[string]bool)
	for _, ev := range events[tailStart:] {
		if ev.CallID != "" {
			inTail[ev.CallID] = true
		}
	}
	// Sessions that own a surviving un-exited spawn. Such a spawn is a leak: a
	// process still RUNNING after its session ended, which Fold flags
	// TOOL_ORPHANED — but ONLY if the session's session_end row is still present
	// to define the orphan boundary. That row is not a spawn and carries no
	// CallID, so neither keep-clause above retains it once it ages out of the
	// tail; dropping it would re-fold the preserved leak as a healthy RUNNING
	// proc, silently erasing the very orphan signal this subsystem exists to
	// surface. Retain the session_end alongside the spawn it orphaned. Order-safe:
	// the spawn precedes its session's end, so a kept spawn never trips Fold's
	// "spawn from an already-ended session" refusal.
	liveSessions := make(map[string]bool)
	for _, ev := range events {
		if ev.Kind == EvSpawn && !terminal[ev.CallID] && ev.Session != "" {
			liveSessions[ev.Session] = true
		}
	}
	out := make([]Event, 0, len(events))
	for i, ev := range events {
		switch {
		case i >= tailStart:
			out = append(out, ev)
		case ev.Kind == EvSpawn && (!terminal[ev.CallID] || inTail[ev.CallID]):
			out = append(out, ev)
		case (ev.Kind == EvExit || ev.Kind == EvKill) && inTail[ev.CallID]:
			// A call represented in the tail keeps its spawn (above) so the tail
			// event is not orphaned — but it must ALSO keep its terminal events, or
			// Fold recomputes the wrong state. When a call's exit/kill fell before
			// the cut while a LATER straddling event (a late pulse, or the exit that
			// follows a kill) sits in the tail, dropping the terminal marker resurrects
			// a DONE proc as RUNNING or downgrades a KILLED one to DONE, silently losing
			// the KILLED / TOOL_RESULT_AFTER_KILL verdict. (Pre-cut pulses of a kept
			// call are still dropped: they change pulse counts, not RUNNING/DONE/KILLED
			// state, so their loss is lossy-but-faithful.)
			out = append(out, ev)
		case ev.Kind == EvSessionEnd && liveSessions[ev.Session]:
			out = append(out, ev)
		case ev.Kind == EvSessionResume && liveSessions[ev.Session]:
			// Retain a retraction on exactly the same terms as the boundary it
			// retracts (#3152). The pair is only meaningful together: keeping the
			// session_end while dropping the session_resume that follows it re-arms a
			// boundary the journal already withdrew, and the next retained spawn for
			// that session then refuses the very journal this function promises is
			// safe to persist. Order-safe by construction — a resume is written ahead
			// of the spawn that refuted the end, so it can never be retained without
			// its own end also being retained or already dropped as harmless.
			out = append(out, ev)
		}
	}
	return out
}
