// outcome.go — the PROVE-THE-RESUME-TOOK half of the resume process (#1146). scan.go's
// Diagnose answers "which sessions crashed and never resumed?"; this file answers the
// question that comes AFTER a watchdog fires `claude --resume`: "did that resume actually
// take?" — from transcript evidence, never from the launcher's own ledger row.
//
// # The gap it closes
//
// A resume ledger records "launched", not "took". A resumed session can progress cleanly,
// re-strand on the same usage-limit wall two seconds later, hit an auth wall no retry can
// fix, or silently produce zero new turns — and the ledger row is identical in all four
// cases. Telling them apart today takes manual transcript forensics. This file is the
// deterministic fold that turns typed transcript facts + typed ledger facts into (a) the
// closed Outcome of the last attempt, (b) the outcome-aware RetryGate a relauncher
// self-gates on (the port of fleet_resume_watchdog.resume_blocked), and (c) the one
// ResumeState label an operator reads per session (pending / launched / took /
// re-stranded / gave-up / settled).
//
// # Content-blind by construction
//
// Like Diagnose, everything here reasons over a closed set of FACTS: the shell
// (cmd/fak/resume_status.go) reads the transcript's terminal turn and classifies its text
// against the wall vocabularies into a TerminalSignal; it parses ledger JSONL rows into
// Attempts. This leaf never sees a byte of transcript content or a ledger string beyond
// the closed phase/action tokens — same facts in, same verdict out, no clock, no I/O.
package resume

import (
	"fmt"
	"strings"
)

// Outcome is how a session's LAST (resumed) turn actually ended, read from the
// transcript's TERMINAL turn — ground truth, never a self-report, and never an earlier
// turn a later one superseded.
type Outcome string

const (
	// OutcomeProgressed: the terminal turn is a normal/clean turn — the resume took.
	OutcomeProgressed Outcome = "progressed"
	// OutcomeRecoverable: the terminal turn is a usage-limit wall (resumable after the
	// named reset) or a transient API error (overloaded/529) — another attempt is warranted.
	OutcomeRecoverable Outcome = "recoverable"
	// OutcomeUnrecoverable: the terminal turn is an auth/login/credit/access wall — a
	// re-resume cannot fix it; it needs a human.
	OutcomeUnrecoverable Outcome = "unrecoverable"
	// OutcomeUnknown: no transcript / no readable terminal turn. Treated as progressed by
	// the retry gate (conservative burn-once: never loop blindly on a session we cannot read).
	OutcomeUnknown Outcome = "unknown"
)

// TerminalSignal is the closed set of facts about a transcript's terminal user/assistant
// turn. The shell matches the turn's text against the wall vocabularies (auth walls,
// "limit … resets …" banners, overloaded/529 transients) and hands this leaf only the bits.
type TerminalSignal struct {
	// Found: a terminal user/assistant turn with text was located. False means no
	// transcript, an unreadable one, or a text-less terminal record.
	Found bool `json:"found"`
	// AuthWall: the text matched the auth/login/credit/access-wall vocabulary.
	AuthWall bool `json:"auth_wall,omitempty"`
	// LimitWall: the text carried a usage-limit banner with a reset window.
	LimitWall bool `json:"limit_wall,omitempty"`
	// TransientAPIError: the text matched the overloaded/529/rate transient vocabulary.
	TransientAPIError bool `json:"transient_api_error,omitempty"`
}

// ClassifyOutcome folds a terminal signal into the closed Outcome. Precedence follows
// remediation cost, mirroring the terminal_failure discipline: an auth wall (needs a
// human) outranks a limit/transient wall (wait and retry), which outranks a clean turn —
// so the most expensive-to-recover wall is never masked by a cheaper reading.
func ClassifyOutcome(sig TerminalSignal) Outcome {
	switch {
	case !sig.Found:
		return OutcomeUnknown
	case sig.AuthWall:
		return OutcomeUnrecoverable
	case sig.LimitWall, sig.TransientAPIError:
		return OutcomeRecoverable
	default:
		return OutcomeProgressed
	}
}

// Attempt is one durable resume-ledger row, reduced to the typed facts the gate reasons
// over. The shell parses the JSONL; unknown fields are dropped, not trusted.
type Attempt struct {
	// UnixSeconds is the row's timestamp; zero when the row carried none.
	UnixSeconds int64 `json:"unix_seconds,omitempty"`
	// Phase is the row's lifecycle token ("launched", "deferred", …). Empty is a launch:
	// the watchdog's launched rows and other launchers' phase-less rows both record a spawn.
	Phase string `json:"phase,omitempty"`
	// Action is the row's operator token; a "consolidate…" prefix marks a manual settle.
	Action string `json:"action,omitempty"`
	// ManualOverride marks an operator-authored row; authoritative, honored forever.
	ManualOverride bool `json:"manual_override,omitempty"`
}

// IsLaunch reports whether this row records a fired resume — the same rule the admit
// gate's launch-rate window uses: a deferral/consideration/skip is bookkeeping, not an
// attempt, so counting it would burn a session's attempt budget on rows where nothing ran.
func (a Attempt) IsLaunch() bool {
	switch strings.ToLower(strings.TrimSpace(a.Phase)) {
	case "deferred", "considered", "skipped":
		return false
	default:
		return true
	}
}

// settled reports whether this row is an operator override (a manual consolidate) — an
// operator settled the session by hand, which is authoritative over any automatic verdict.
func (a Attempt) settled() bool {
	return a.ManualOverride || strings.HasPrefix(strings.ToLower(strings.TrimSpace(a.Action)), "consolidate")
}

// DefaultMaxResumeAttempts is the give-up cap on automatic resumes of one session,
// matching the watchdog's FAK_MAX_ATTEMPTS default.
const DefaultMaxResumeAttempts = 8

// CountAttempts is the number of fired launches in a session's ledger history.
func CountAttempts(history []Attempt) int {
	n := 0
	for _, a := range history {
		if a.IsLaunch() {
			n++
		}
	}
	return n
}

// LastLaunchUnix is the timestamp of the most recent fired launch, or zero when the
// history holds none (or none carried a timestamp).
func LastLaunchUnix(history []Attempt) int64 {
	var last int64
	for _, a := range history {
		if a.IsLaunch() && a.UnixSeconds > last {
			last = a.UnixSeconds
		}
	}
	return last
}

// NewTurnsAfter counts the real model turns that landed strictly after sinceUnix — the
// evidence a resume produced progress. A zero/negative sinceUnix (no launch on record)
// yields zero: "new turns since a resume" is meaningless before any resume fired.
func NewTurnsAfter(turnUnixTimes []int64, sinceUnix int64) int {
	if sinceUnix <= 0 {
		return 0
	}
	n := 0
	for _, t := range turnUnixTimes {
		if t > sinceUnix {
			n++
		}
	}
	return n
}

// RetryDecision is the outcome-aware once-gate verdict: whether a NEW automatic resume of
// this session is blocked, and the closed human reason.
type RetryDecision struct {
	Blocked bool   `json:"blocked"`
	Reason  string `json:"reason"`
}

// RetryGate decides whether a new automatic resume is blocked, given the session's prior
// ledger rows (oldest first) and the Outcome of the last attempt. It replaces "any ledger
// row ⇒ never again" with "blocked unless the last attempt failed recoverably and we are
// under the attempt cap" — so a resume that immediately re-hit a usage-limit wall is
// retried past the reset instead of being permanently stranded, while a clean finish or
// an auth wall stays burned. maxAttempts <= 0 takes DefaultMaxResumeAttempts.
func RetryGate(history []Attempt, outcome Outcome, maxAttempts int) RetryDecision {
	if maxAttempts <= 0 {
		maxAttempts = DefaultMaxResumeAttempts
	}
	if len(history) == 0 {
		return RetryDecision{Blocked: false, Reason: "first resume"}
	}
	for _, a := range history {
		if a.settled() {
			return RetryDecision{Blocked: true, Reason: "operator-settled (manual ledger override)"}
		}
	}
	attempts := CountAttempts(history)
	if attempts == 0 {
		attempts = len(history)
	}
	if attempts >= maxAttempts {
		return RetryDecision{Blocked: true, Reason: fmt.Sprintf("attempt cap reached (%d/%d)", attempts, maxAttempts)}
	}
	switch outcome {
	case OutcomeRecoverable:
		return RetryDecision{Blocked: false,
			Reason: fmt.Sprintf("last resume failed recoverably; attempt %d/%d", attempts+1, maxAttempts)}
	case OutcomeUnrecoverable:
		return RetryDecision{Blocked: true, Reason: "last resume hit an auth/access wall — a re-resume cannot fix it"}
	default:
		// progressed / unknown: the resume took, or we cannot prove it didn't — burn once.
		return RetryDecision{Blocked: true, Reason: "already resumed once (resume took)"}
	}
}

// ResumeState is where a crashed session stands in its resume journey — the one label an
// operator reads per session (#1146's closed vocabulary, plus the operator-settled case).
type ResumeState string

const (
	// ResumePending: crashed, and no resume has been fired yet.
	ResumePending ResumeState = "pending"
	// ResumeLaunched: a resume fired, but the transcript cannot yet prove progress — the
	// silent case a ledger alone cannot distinguish from success.
	ResumeLaunched ResumeState = "launched"
	// ResumeTook: a resume fired AND the transcript shows new real turns with a clean
	// terminal turn — provably progressed.
	ResumeTook ResumeState = "took"
	// ResumeReStranded: a resume fired and the session is walled again (limit/transient)
	// — eligible for another attempt once the wall clears, per RetryGate.
	ResumeReStranded ResumeState = "re-stranded"
	// ResumeGaveUp: no automatic resume will fire again — the attempt cap is spent or the
	// wall is one a re-resume cannot fix (auth/access); a human owns it now.
	ResumeGaveUp ResumeState = "gave-up"
	// ResumeSettled: an operator settled the session by hand (manual ledger override).
	ResumeSettled ResumeState = "settled"
)

// ResumeFacts is the closed input to FoldResumeState: ledger facts plus transcript
// evidence, all typed, all shell-extracted.
type ResumeFacts struct {
	// Attempts is the fired-launch count from the ledger (CountAttempts).
	Attempts int `json:"attempts"`
	// MaxAttempts is the give-up cap; <= 0 takes DefaultMaxResumeAttempts.
	MaxAttempts int `json:"max_attempts,omitempty"`
	// OperatorSettled: any ledger row is a manual override/consolidate.
	OperatorSettled bool `json:"operator_settled,omitempty"`
	// NewTurns is the count of real model turns after the last launch (NewTurnsAfter).
	NewTurns int `json:"new_turns"`
	// Outcome is the terminal-turn classification of the last attempt (ClassifyOutcome).
	Outcome Outcome `json:"outcome"`
}

// FoldResumeState folds the facts into the one per-session label. Precedence, most
// authoritative first: no attempt yet → pending; an operator settle is final; proven
// progress (new turns + clean terminal) is took even at the attempt cap; an auth wall or
// a spent cap is gave-up; a re-hit wall is re-stranded; anything else stays launched —
// fired, unproven. Total over any input, never a panic.
func FoldResumeState(f ResumeFacts) ResumeState {
	max := f.MaxAttempts
	if max <= 0 {
		max = DefaultMaxResumeAttempts
	}
	switch {
	case f.Attempts <= 0:
		return ResumePending
	case f.OperatorSettled:
		return ResumeSettled
	case f.NewTurns > 0 && f.Outcome == OutcomeProgressed:
		return ResumeTook
	case f.Outcome == OutcomeUnrecoverable:
		return ResumeGaveUp
	case f.Attempts >= max:
		return ResumeGaveUp
	case f.Outcome == OutcomeRecoverable:
		return ResumeReStranded
	default:
		return ResumeLaunched
	}
}
