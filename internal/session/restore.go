package session

// restore.go — the durable-resume write: load a full drive record VERBATIM into the
// table, Rev and all. It is the load-time inverse of Snapshot (the dump-time read):
// Snapshot serializes every live session's State; Restore re-attaches ONE persisted
// State so a process restart — or a session OFFLOADED to another host, user, instance,
// or VM and brought back — resumes at the budget / priority / run-state / pace it held,
// not at a default. This is the §5 "persistence" rung of the design note
// docs/notes/SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md, and the drive half of
// the portable session image (internal/sessionimage): a STOPPED session reloads as
// STOPPED (with its closed reason token), never silently resurrected as RUNNING.
//
// Unlike the live control verbs (Transition / SetBudget / SetPace / SetPriority),
// Restore deliberately differs on two points, because a LOAD is not a MUTATION:
//
//   - It does NOT bump Rev. The persisted Rev IS the record's revision lineage; a
//     resumed session must report the same Rev it was dumped at, so an operator UI that
//     held an If-Rev across the offload still composes. A round-trip
//     Snapshot -> Restore is therefore the identity (Rev included).
//   - It does NOT enforce the terminal guard. The control verbs refuse to write a
//     terminal (Stopped) session — "you start a new session, you do not un-stop one."
//     But re-attaching a dumped image is exactly the case where a terminal session MUST
//     be re-established faithfully: a Stopped image restores Stopped. Restore is the one
//     write that may set a terminal record, precisely because resume must be honest.
//
// The TraceID in st is replaced by the trace key, so a State read under one key restores
// under whatever key the caller chooses (a session can be re-homed to a new id on a new
// host). Trust is the caller's concern: Restore writes the bytes as given. The image
// layer (internal/sessionimage) verifies a dumped image's integrity (a sha256 over every
// part, fail-closed) BEFORE it calls here, so an offloaded session's drive cannot be
// silently tampered en route.

// Restore loads a full drive record verbatim under trace, preserving its Rev, and
// returns the stored record. It is the durable-resume inverse of Snapshot: a Stopped
// image restores AS Stopped (Restore is the only write that re-establishes a terminal
// session), and Snapshot followed by Restore is the identity. The empty TraceID in st
// is replaced by trace.
func (t *Table) Restore(trace string, st State) State {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.ensureLocked()
	st.TraceID = trace
	t.state[trace] = st
	t.touchLocked(trace)
	t.trimLocked()
	return st
}
