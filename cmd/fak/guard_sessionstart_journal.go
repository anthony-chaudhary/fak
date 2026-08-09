package main

import (
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

// guard_sessionstart_journal.go — C3 of the crash-journal epic (#3787 / epic #3784):
// UNCONDITIONAL register-on-start.
//
// The boot-epoch fold (C1) can only classify sessions somebody actually RECORDED, and until
// now nothing recorded the common case. The two stores that existed both miss it:
//
//   - `session-registry.json` (#1197) is opt-in — it needs guardDurabilityWanted, so a plain
//     attended `fak guard -- claude` publishes no descriptor at all;
//   - the SessionStart hook itself wrote the uuid<->trace identity join (#4112) and nothing
//     else, so the session's own START — cwd, boot epoch, driver pid — went unrecorded.
//
// So a machine-wide crash (a Windows-update reboot, the 0xc0000005 that killed every terminal
// on 2026-07-09) erased exactly the sessions nobody had opted in for. This writes the `open`
// row for EVERY SessionStart, which is what makes `fak sessionjournal report` able to say
// "these died in the reboot" about an interactive session rather than only a fleet one.
//
// Ordering and posture match the identity join it rides beside: it runs BEFORE the affordance
// mode check (the FAK_GUARD_AFFORDANCE_MODE=off knob governs the injected hint, not a durable
// store) and it is fail-open — a bad path or a write error is a silent no-op, never a wedged
// session start. Its own kill switch is FAK_SESSION_JOURNAL_REGISTER=off, which is an opt-OUT:
// the default is on, so registration is unconditional in the sense the DoD asks for (no config
// opt-in gates it) while a lean harness can still turn it off.

const (
	// guardSessionJournalEnvMode is the register-on-start kill switch. Default (empty) is ON.
	guardSessionJournalEnvMode = "FAK_SESSION_JOURNAL_REGISTER"
	guardSessionJournalModeOff = "off"
	// guardSessionJournalAgent is the wrapped agent this hook can only ever belong to: the
	// SessionStart hook is installed exclusively for a claude child (guardPreCompactIsClaudeCommand),
	// so the value is witnessed by the install rule rather than guessed from the environment.
	guardSessionJournalAgent = "claude"
)

// recordGuardSessionStartJournal appends one boot-stamped `open` row for this session to the
// crash-survivable journal. driverPID is the pid the identity join already witnessed (0 when
// none was) — a pid is recorded only when it was WITNESSED, matching that store's contract,
// because a wrong pid is strictly worse than none for a later liveness fold.
//
// Best-effort at every step, per the hook's contract: an off knob, an unidentifiable session,
// or a write error all return silently.
func recordGuardSessionStartJournal(traceID string, driverPID int) {
	if strings.EqualFold(strings.TrimSpace(os.Getenv(guardSessionJournalEnvMode)), guardSessionJournalModeOff) {
		return
	}
	id := guardSessionJournalID(traceID)
	if id == "" {
		return // nothing identifies this session; ParseEvents would skip an id-less row anyway
	}
	now := time.Now().UTC()
	bt, _ := sessionjournal.BootTime(now)
	cwd, _ := os.Getwd()
	_ = sessionjournal.Append("", sessionjournal.Event{
		Kind:  sessionjournal.KindOpen,
		ID:    id,
		TS:    now.Format(time.RFC3339),
		Boot:  sessionjournal.BootID(bt),
		PID:   driverPID,
		Host:  sessionJournalHost(),
		CWD:   cwd,
		Agent: guardSessionJournalAgent,
	})
}

// guardSessionJournalID picks the fold key for the row. The transcript UUID comes FIRST and
// that ordering is load-bearing: resolveGuardSessionID hands an ordinary attended launch the
// CONSTANT "guard" as its trace, so keying on the trace would fold every interactive session
// on the host into one row — precisely the case C3 exists to record. The trace is the fallback
// for a resumed child, which has CLAUDE_CODE_SESSION_ID stripped; there the trace is a real
// per-session id (a resume is launched with an explicit --session-id). Empty when neither
// exists, which records nothing rather than an anonymous row.
func guardSessionJournalID(traceID string) string {
	if uuid := strings.TrimSpace(os.Getenv("CLAUDE_CODE_SESSION_ID")); uuid != "" {
		return uuid
	}
	return strings.TrimSpace(traceID)
}
