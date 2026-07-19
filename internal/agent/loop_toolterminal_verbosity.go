package agent

// loop_toolterminal_verbosity.go — the verbosity axis of the background-process
// completion watcher (#2932). The wake PATH itself (supervisor terminal sink →
// ToolTerminalWakeQueue → the loop's turn-boundary splice) landed with #2400;
// what it lacked was Hermes' `display.background_process_notifications` control,
// so every terminal transition woke a turn at full detail with no way to say
// "only wake me on failures" or "don't wake me at all".
//
// Two axes, deliberately separated:
//
//   - ADMISSION — does this terminal transition wake a turn at all? `off` never
//     wakes; `error` wakes only on a FAILURE terminal (KILLED, or DONE with a
//     non-ok exit status); `result`/`all` wake on every terminal.
//   - DETAIL — how much of the folded verdict rides into the resumed turn.
//     `result` carries the outcome-only projection (did it finish, how did it
//     go); `all` and `error` carry the full verdict, because a caller who asked
//     to be woken for a diagnosis needs the findings to act on.
//
// The zero value ("") is `all` — byte-for-byte the pre-#2932 wake — so every
// existing construction site keeps its historical behavior.
//
// What this file does NOT do, on purpose: re-adjudicate the completion result.
// toolproc.Proc is already the kernel's FOLDED verdict — closed Reason tokens,
// deterministic Detail lines, caller-supplied identifiers. The background
// process's own stdout/stderr never reaches this struct, so the bytes spliced
// into the resumed turn are adjudicated by construction, not by a check bolted
// on here. TestToolTerminalWakeCarriesNoRawProcessOutput pins that invariant.

import "github.com/anthony-chaudhary/fak/internal/toolproc"

// ToolTerminalVerbosity is the closed verbosity vocabulary for background-process
// completion wakes — the fak twin of Hermes' display.background_process_notifications.
type ToolTerminalVerbosity string

const (
	// ToolTerminalVerbosityAll wakes on every terminal transition, full verdict.
	ToolTerminalVerbosityAll ToolTerminalVerbosity = "all"
	// ToolTerminalVerbosityResult wakes on every terminal transition, outcome-only.
	ToolTerminalVerbosityResult ToolTerminalVerbosity = "result"
	// ToolTerminalVerbosityError wakes only on a failure terminal, full verdict.
	ToolTerminalVerbosityError ToolTerminalVerbosity = "error"
	// ToolTerminalVerbosityOff never wakes a turn on a background completion.
	ToolTerminalVerbosityOff ToolTerminalVerbosity = "off"
)

// ParseToolTerminalVerbosity maps a configured string onto the closed vocabulary.
// An empty string resolves to the default (all). An unrecognized value is refused
// rather than silently defaulting, so a typo'd setting is a loud misconfiguration
// instead of an unexpectedly chatty — or unexpectedly silent — session.
func ParseToolTerminalVerbosity(s string) (ToolTerminalVerbosity, bool) {
	switch ToolTerminalVerbosity(s) {
	case "":
		return ToolTerminalVerbosityAll, true
	case ToolTerminalVerbosityAll:
		return ToolTerminalVerbosityAll, true
	case ToolTerminalVerbosityResult:
		return ToolTerminalVerbosityResult, true
	case ToolTerminalVerbosityError:
		return ToolTerminalVerbosityError, true
	case ToolTerminalVerbosityOff:
		return ToolTerminalVerbosityOff, true
	}
	return "", false
}

// isFailure reports whether a terminal verdict is a FAILURE terminal: the
// supervisor killed it, or it exited with a non-ok status. A DONE proc with an
// empty ExitStatus is treated as success — the fold only leaves it empty when
// nothing reported otherwise.
func isFailure(p toolproc.Proc) bool {
	return p.State == toolproc.StateKilled || (p.ExitStatus != "" && p.ExitStatus != "ok")
}

// admits reports whether this verbosity level wakes a turn for verdict p.
func (v ToolTerminalVerbosity) admits(p toolproc.Proc) bool {
	switch v {
	case ToolTerminalVerbosityOff:
		return false
	case ToolTerminalVerbosityError:
		return isFailure(p)
	}
	return true
}

// project returns the verdict this verbosity level carries into the resumed turn.
// `result` strips the diagnostic surface — findings, liveness/pulse counters, and
// the scheduling fields — down to the outcome a fire-and-forget caller asked for:
// which call, which tool, and how it ended. Every other level carries the verdict
// whole.
func (v ToolTerminalVerbosity) project(p toolproc.Proc) toolproc.Proc {
	if v != ToolTerminalVerbosityResult {
		return p
	}
	return toolproc.Proc{
		CallID:     p.CallID,
		Tool:       p.Tool,
		Session:    p.Session,
		State:      p.State,
		ExitStatus: p.ExitStatus,
		KillReason: p.KillReason,
		EndMS:      p.EndMS,
		RuntimeMS:  p.RuntimeMS,
	}
}

// resolve returns the effective level, mapping the zero value onto the default.
func (v ToolTerminalVerbosity) resolve() ToolTerminalVerbosity {
	if v == "" {
		return ToolTerminalVerbosityAll
	}
	return v
}

// recordSuppressed journals a terminal verdict the verbosity withheld. A
// suppressed wake is RECORDED, never silently dropped: "my background job
// finished and nothing happened" must be answerable from the journal, otherwise
// a misconfigured verbosity is indistinguishable from a lost wake.
func (q *ToolTerminalWakeQueue) recordSuppressed(p toolproc.Proc) {
	w := ToolTerminalWake{Kind: ToolTerminalWakeKind, TraceID: p.CallID, Session: p.Session, Verdict: p}
	q.mu.Lock()
	q.records = append(q.records, ToolTerminalWakeRecord{Wake: w, Status: "SUPPRESSED"})
	q.mu.Unlock()
}

// TerminalWakeSink returns the toolprocgate.Supervisor.SetTerminalSink function
// that applies verbosity v before q sees a completion (#2932).
//
// Verbosity is a property of the WIRING, not of the mailbox: the same queue can
// be fed by a sink at any level, and `sup.SetTerminalSink(q.Enqueue)` — the
// pre-#2932 wiring — keeps its exact historical behavior because that path is
// untouched. TerminalWakeSink(q, "") is that same default, spelled explicitly.
func TerminalWakeSink(q *ToolTerminalWakeQueue, v ToolTerminalVerbosity) func(toolproc.Proc) {
	level := v.resolve()
	return func(p toolproc.Proc) {
		if q == nil {
			return
		}
		// Ownership and terminality are the queue's own admission rules; re-checked
		// here so a suppressed-vs-foreign verdict is never mis-journaled as ours.
		if p.Session != q.trace || (p.State != toolproc.StateDone && p.State != toolproc.StateKilled) {
			return
		}
		if !level.admits(p) {
			q.recordSuppressed(p)
			return
		}
		q.Enqueue(level.project(p))
	}
}
