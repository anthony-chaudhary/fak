package main

import (
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
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

func recordGuardSessionStartJournalFor(traceID, sessionID, agent string, driverPID int) {
	if strings.EqualFold(strings.TrimSpace(os.Getenv(guardSessionJournalEnvMode)), guardSessionJournalModeOff) {
		return
	}
	id := guardSessionJournalIDFor(traceID, sessionID)
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
		Agent: strings.TrimSpace(agent),
	})
}

// recordGuardProviderSessionClose closes the provider session bound to traceID
// before a clear-origin child is opened. The identity ledger is the witnessed
// trace<->provider-id join; without a join this records nothing rather than close
// a guessed journal key.
func recordGuardProviderSessionClose(traceID, reason string) {
	if strings.EqualFold(strings.TrimSpace(os.Getenv(guardSessionJournalEnvMode)), guardSessionJournalModeOff) {
		return
	}
	_, sessionByTrace := resume.LoadIdentity(resolveSweepRegDir(""))
	id := strings.TrimSpace(sessionByTrace[strings.TrimSpace(traceID)])
	if id == "" {
		return
	}
	now := time.Now().UTC()
	bt, _ := sessionjournal.BootTime(now)
	_ = sessionjournal.Append("", sessionjournal.Event{
		Kind: sessionjournal.KindClose, ID: id, TS: now.Format(time.RFC3339),
		Boot: sessionjournal.BootID(bt), Reason: strings.TrimSpace(reason),
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
	return guardSessionJournalIDFor(traceID, os.Getenv("CLAUDE_CODE_SESSION_ID"))
}

func guardSessionJournalIDFor(traceID, sessionID string) string {
	if uuid := strings.TrimSpace(sessionID); uuid != "" {
		return uuid
	}
	return strings.TrimSpace(traceID)
}
