package toolproc

// sessionresume.go — the toolproc/DOS journal lifetime split (#3152, gap 2).
//
// THE DEFECT. The workspace toolproc journal is fed by the harness hooks
// `fak guard` installs (cmd/fak/guard_toolproc_hooks.go): PreToolUse -> spawn,
// PostToolUse -> exit, SessionEnd -> session_end. `session_end` is the ORPHAN
// BOUNDARY: Fold treats a spawn arriving after it as the leaked-child class and
// refuses the whole table. That is sound only if `session_end` means the session
// really ended. It does not. The harness fires SessionEnd at events that are
// terminal for a TURN or a transcript, not for the session id — and the session
// then goes on calling tools under the SAME id.
//
// THE WITNESS (measured on this workspace's live .fak/toolproc/journal.jsonl,
// 29,525 rows, 2,344 sessions, 2026-07-16 .. 2026-08-07):
//
//   - 224 of 1,517 session_end rows (14.8%) are followed by a further spawn in
//     the same session. The shortest gap between "this session ended" and that
//     session spawning again is 20 ms; the median is 6.4 minutes.
//   - 140 sessions carry MORE THAN ONE session_end row (one carries 35). A
//     session cannot end thirty-five times; the row is not describing the
//     session's lifetime, it is describing a hook firing.
//   - Consequence: Fold refuses at row 100 of 29,525 — 0.34% into the file. The
//     documented reader, `fak toolproc ps --events .fak/toolproc/journal.jsonl`,
//     has returned nothing but that refusal since 2026-07-16. #3152 reports the
//     per-tool ledger reading ~2% of its baseline; the true figure is 0%.
//
// So the row's lifetime is decoupled from the thing it claims to describe. That
// is the whole bug, and it is on the PRODUCER side.
//
// WHY A WRITTEN ROW AND NOT A LOOSER READER. The obvious cure — let Fold tolerate
// a spawn from an ended session — is the one to refuse. The refusal exists for a
// real class (children forked from a dead parent, #3032) and #5524 settled the
// precedent for exactly this shape of defect: cure the writer, never weaken the
// check. So the journal is append-only and the correction is appended.
//
// A session_end is a CLAIM about the future ("this session will not spawn
// again"). The producer cannot check it when it writes it. But at the next
// firing it CAN: a spawn it is about to journal for that session is direct
// evidence the claim was false. hookSessionResume emits `session_resume` at that
// moment, ahead of the spawn, and Fold consumes it by disarming the boundary.
//
// This keeps the check's teeth where they belong. The retraction is only ever
// written by the hook that OBSERVED the session alive. A child forked from a
// genuinely dead parent does not fire that session's PreToolUse hook, and a
// supervisor-brokered spawn (ArmMonitor, monitor.go) never routes through
// HookEvents at all — both still hit Fold's refusal untouched. What changes is
// only the case where the journal itself carries first-hand, timestamped
// evidence that the boundary was wrong.
//
// It is also not silent: Counts.SessionsResumed reports how many sessions ended
// prematurely, so a workspace whose harness does this can see that it does.
//
// SCOPE. This repairs journals written from here on. It does not retro-fix rows
// already on disk: a journal that already contains a bare post-end spawn still
// refuses, correctly, because nothing in it retracts the boundary. Compaction
// (CompactJournal) reclaims that history as the calls involved go terminal.

// hookSessionResume returns the session_resume row a firing must write BEFORE
// its own events, and whether one is needed.
//
// It is needed when this firing is about to journal a spawn for a session that
// `existing` shows as ended and not since resumed — i.e. exactly when the
// firing's own evidence refutes a boundary still standing in the journal. It is
// a pure function: no clock, no IO, and it reads only rows already durable.
//
// The check runs over the events the firing produced rather than over the hook
// kind, so it covers every spawn-producing path in HookEvents — the PreToolUse
// spawn and the post-hook background-launch bridge alike.
func hookSessionResume(out []Event, nowMS int64, existing []Event) (Event, bool) {
	session := ""
	for _, ev := range out {
		if ev.Kind == EvSpawn && ev.Session != "" {
			session = ev.Session
			break
		}
	}
	if session == "" {
		return Event{}, false
	}
	if !sessionEndedUnretracted(session, existing) {
		return Event{}, false
	}
	return Event{Kind: EvSessionResume, Session: session, AtMS: nowMS}, true
}

// sessionEndedUnretracted reports whether the journal's last lifetime row for
// session is a session_end (armed boundary) rather than a session_resume
// (already retracted) or nothing at all. Last-wins, matching Fold: a session may
// legitimately end, resume, and end again, and only the standing boundary is
// what a new spawn would collide with.
func sessionEndedUnretracted(session string, existing []Event) bool {
	ended := false
	for _, ev := range existing {
		if ev.Session != session {
			continue
		}
		switch ev.Kind {
		case EvSessionEnd:
			ended = true
		case EvSessionResume:
			ended = false
		}
	}
	return ended
}
