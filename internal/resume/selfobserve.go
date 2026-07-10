package resume

// selfobserve.go — the WORKER-FACING self-observation fold (#3804). `resume status`
// (cmd/fak/resume_status.go) is the OPERATOR readout: it walks a whole transcript store and
// folds every crashed session for a human triaging a batch. This file answers the narrower,
// first-person question a single guarded worker asks about ITSELF, without operator
// involvement: "did MY session get resumed, did that resume take, and will another attempt
// fire — or does a human own me now?"
//
// It is a THIN composition of the same closed folds `resume status` uses (CountAttempts,
// LastLaunchUnix, EarnedResumeBudget, RetryGate, FoldResumeState) over ONE session's ledger
// history plus the transcript-witnessed outcome/progress the shell supplies. Same facts in,
// same record out — no clock, no I/O, no transcript content. The shell halves (the CLI
// `fak resume self` and the fak_resume_history MCP hook) do the ledger/transcript reads; this
// leaf only folds.
//
// # Fail-closed on no evidence
//
// A session with no ledger rows is not an error and never a fabricated "took": HasHistory is
// false and the record folds to the honest floor (pending, zero attempts, no retry block).
// A worker that was never resumed reads exactly that — the same conservative posture the
// operator fold takes when a launch record is missing.

// SelfFacts is the closed, typed input to FoldSelfObservation: one session's ledger history
// (oldest first, as RetryGate expects) plus the transcript-witnessed progress the shell
// lifted. Every field is shell-extracted from a durable record; this leaf trusts none of it
// beyond the closed tokens.
type SelfFacts struct {
	// Session is the session id being observed — carried through to the record for display;
	// the fold takes no view on it.
	Session string
	// History is this session's fired + bookkeeping ledger rows, oldest first. Empty means
	// no launch record: the fail-closed floor.
	History []Attempt
	// Outcome is the terminal-turn classification of the last attempt (ClassifyOutcome over
	// the transcript's terminal signal). OutcomeUnknown when the shell could not read a
	// terminal turn — the conservative burn-once reading the gate already handles.
	Outcome Outcome
	// NewTurns is the count of real model turns that landed after the last launch
	// (NewTurnsAfter) — the transcript-witnessed proof a resume produced work. Zero until a
	// resume fires and a real turn lands.
	NewTurns int
	// MaxAttempts is the give-up cap; <= 0 means "use the progress-earned budget"
	// (EarnedResumeBudget), the same convention RetryGate and the operator fold take.
	MaxAttempts int
}

// SelfObservation is the worker's own witnessed recovery record: the leaf verdicts a guarded
// session reads about itself, plus the one-line self-advice hint. It is the return shape of
// both `fak resume self` and the fak_resume_history MCP tool.
type SelfObservation struct {
	Session string `json:"session,omitempty"`
	// HasHistory is false when no ledger row exists for this session — the fail-closed empty
	// record. A reader keys on this to tell "never resumed" from "resumed and settled".
	HasHistory bool `json:"has_history"`
	// Attempts is the fired-launch count (CountAttempts) — bookkeeping rows excluded.
	Attempts int `json:"attempts"`
	// LastLaunchUnix is the most recent fired launch's timestamp, zero when none carried one.
	LastLaunchUnix int64 `json:"last_launch_unix,omitempty"`
	// NewTurns is the real turns witnessed after the last launch (the progress proof).
	NewTurns int `json:"new_turns_since_resume"`
	// Outcome is the terminal-turn classification of the last attempt.
	Outcome Outcome `json:"outcome"`
	// State is the one resume-journey label (FoldResumeState): pending / launched / took /
	// re-stranded / gave-up / settled.
	State ResumeState `json:"resume_state"`
	// RetryBlocked / RetryReason is the outcome-aware once-gate verdict (RetryGate): whether a
	// NEW automatic resume of this session is blocked, and the closed human reason.
	RetryBlocked bool   `json:"retry_blocked"`
	RetryReason  string `json:"retry_reason"`
	// EarnedBudget is the progress-earned give-up cap this session's history earned
	// (EarnedResumeBudget) — what the automatic relauncher will ration it to when no explicit
	// cap is set. Surfaced so a worker sees how many attempts it has left, not just whether
	// the next one is blocked.
	EarnedBudget int `json:"earned_budget"`
	// OperatorSettled is true when any ledger row is a manual override/consolidate — a human
	// settled the session by hand, authoritative over any automatic verdict.
	OperatorSettled bool `json:"operator_settled,omitempty"`
	// NextHint is the one-line, closed-vocabulary self-advice keyed on State: what THIS worker
	// should understand about its own recovery posture. Advice-only; nothing acts on it.
	NextHint string `json:"next_hint"`
}

// FoldSelfObservation folds one session's facts into its self-observation record. It is a pure
// composition of the existing outcome folds, so `fak resume self`, the MCP hook, and the
// operator `resume status` table can never tell a worker a different story than they tell a
// human. Total over any input; empty history returns the fail-closed floor.
func FoldSelfObservation(f SelfFacts) SelfObservation {
	attempts := CountAttempts(f.History)
	settled := false
	for _, a := range f.History {
		if a.settled() {
			settled = true
			break
		}
	}
	gate := RetryGate(f.History, f.Outcome, f.MaxAttempts)
	state := FoldResumeState(ResumeFacts{
		Attempts:        attempts,
		MaxAttempts:     f.MaxAttempts,
		OperatorSettled: settled,
		NewTurns:        f.NewTurns,
		Outcome:         f.Outcome,
	})
	return SelfObservation{
		Session:         f.Session,
		HasHistory:      len(f.History) > 0,
		Attempts:        attempts,
		LastLaunchUnix:  LastLaunchUnix(f.History),
		NewTurns:        f.NewTurns,
		Outcome:         f.Outcome,
		State:           state,
		RetryBlocked:    gate.Blocked,
		RetryReason:     gate.Reason,
		EarnedBudget:    EarnedResumeBudget(f.History),
		OperatorSettled: settled,
		NextHint:        selfNextHint(state),
	}
}

// selfNextHint maps the resume-journey state to the one-line first-person advice a worker
// reads about itself. Closed over the ResumeState vocabulary; a state with no case (there is
// none — the switch is total over the six labels) folds to the neutral launched hint.
func selfNextHint(state ResumeState) string {
	switch state {
	case ResumePending:
		return "no resume attempt on record for this session — nothing to recover"
	case ResumeTook:
		return "a resume took — new turns landed on a clean terminal turn; you are progressing"
	case ResumeReStranded:
		return "re-stranded on a wall — another automatic attempt is warranted once it clears"
	case ResumeGaveUp:
		return "no automatic resume will fire again — the cap is spent or the wall needs a human"
	case ResumeSettled:
		return "an operator settled this session by hand — the automatic gate is closed"
	default: // ResumeLaunched
		return "a resume fired but progress is not yet witnessed — waiting for a real turn"
	}
}
