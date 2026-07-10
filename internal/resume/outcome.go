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
	// OutcomeCancelled: the terminal turn is a deliberate interrupt/stop — the operator
	// (or the harness on their behalf) chose to end the session. Relaunching would undo
	// an intentional decision, so the gate refuses it outright (#3354); the deny twin of
	// OutcomeRecoverable's transport-death relaunch arm.
	OutcomeCancelled Outcome = "cancelled"
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
	// Cancelled: the text matched the deliberate interrupt/stop vocabulary (an operator
	// interrupt, an explicit stop verb) — intent, not an environmental wall.
	Cancelled bool `json:"cancelled,omitempty"`
}

// ClassifyOutcome folds a terminal signal into the closed Outcome. Precedence follows
// remediation cost, mirroring the terminal_failure discipline: a deliberate cancel
// (intent — relaunch is forbidden, not merely useless) outranks an auth wall (needs a
// human) outranks a limit/transient wall (wait and retry), which outranks a clean turn —
// so the most expensive-to-recover reading is never masked by a cheaper one.
func ClassifyOutcome(sig TerminalSignal) Outcome {
	switch {
	case !sig.Found:
		return OutcomeUnknown
	case sig.Cancelled:
		return OutcomeCancelled
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
	// An operator-settled row (a manual consolidate, marked by Action not Phase) is an
	// override, not a fired launch — counting it would burn an attempt on a row where the
	// operator, not the watchdog, acted. RetryGate already blocks on settled(); keep the
	// launch count consistent so a phase-less consolidate row is not miscounted as a spawn.
	if a.settled() {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(a.Phase)) {
	case "deferred", "considered", "skipped", "gate_fail_open", "queued", "detected", "status", "tick", "snapshot", "progress", "settled", "operator_settled", "consolidated", "rearm":
		return false
	default:
		return true
	}
}

// isRearm reports whether this row is a re-arm reclaim marker (#2178): an operator/self-heal row
// that reclaims a session which burned its whole attempt budget on a KNOWN-transient infra fault
// (e.g. the managed-cache-1h-TTL 400 wave) rather than a real defect. RetryGate considers only the
// history AFTER the last such marker, so the reclaimed session resumes from a fresh attempt budget.
func (a Attempt) isRearm() bool {
	return strings.EqualFold(strings.TrimSpace(a.Phase), "rearm")
}

// afterLastRearm returns the suffix of history following the last re-arm marker (#2178), or the
// whole history when none is present. Because the marker zeroes the budget accrued before it, a
// later manual_override/unrecoverable row still wins (it lands in the returned suffix): last write
// wins, exactly like the .ps1 launch gate and the Python resume_blocked / planner counters.
func afterLastRearm(history []Attempt) []Attempt {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].isRearm() {
			return history[i+1:]
		}
	}
	return history
}

// settled reports whether this row is an operator override (a manual consolidate) — an
// operator settled the session by hand, which is authoritative over any automatic verdict.
func (a Attempt) settled() bool {
	return a.ManualOverride || strings.HasPrefix(strings.ToLower(strings.TrimSpace(a.Action)), "consolidate")
}

// DefaultMaxResumeAttempts is the give-up cap on automatic resumes of one session,
// matching the watchdog's FAK_MAX_ATTEMPTS default. It is the BASE the earned-budget
// curve (EarnedResumeBudget) pivots around: a session making real progress earns up to
// MaxEarnedResumeBudget attempts, a stalled one is cut down toward MinEarnedResumeBudget.
const DefaultMaxResumeAttempts = 8

// The earned-budget curve floor and ceiling. A thrashing session (every resume
// re-strands within ProgressGapSeconds, doing no work) is cut toward the floor so we
// stop burning attempts on a session that never moves; a session that keeps making real
// progress between resumes earns up to the ceiling. Both are clamps on
// EarnedResumeBudget, never on an explicit operator-set cap.
const (
	// MinEarnedResumeBudget is the fewest resumes a thrashing session can earn — enough
	// to survive a genuine but transient double-wall, but far below the flat 8 a
	// progress-blind cap would have handed it.
	MinEarnedResumeBudget = 2
	// MaxEarnedResumeBudget is the most a consistently-progressing session can earn above
	// the base — a session that keeps doing real work between resumes is worth resuming
	// past the flat cap rather than abandoning mid-progress.
	MaxEarnedResumeBudget = 12
)

// ProgressGapSeconds is the minimum wall-clock gap between two consecutive fired launches
// for the earlier launch to count as PROGRESS rather than THRASH. A resume that produced
// real work ran for a meaningful stretch before the session re-stranded and the next
// resume fired; a resume that re-hit its wall within this window did essentially nothing.
// Ten minutes is deliberately conservative — a usage-limit re-strand is typically
// seconds, real coding work is minutes-plus — so a borderline gap is read as progress,
// never as thrash.
const ProgressGapSeconds int64 = 600

// EarnedResumeBudget is the progress-earned give-up cap: instead of granting every
// session the flat DefaultMaxResumeAttempts, it reads the session's own launch history as
// evidence of whether prior resumes did real work, and returns a budget the session
// earned. The signal is purely the spacing of the fired launches already on the ledger —
// no transcript content, no clock, same history in, same budget out:
//
//   - Each consecutive pair of fired launches whose gap is >= ProgressGapSeconds is a
//     PROGRESS interval (the earlier resume ran productively before re-stranding) and
//     earns +1 above the base.
//   - Each pair whose gap is < ProgressGapSeconds is a THRASH interval (the resume
//     re-stranded almost immediately, doing no work) and costs -1 below the base.
//   - A gap we cannot measure (either launch missing a timestamp) is neutral: absence of
//     evidence never REDUCES a session's budget, so an untimestamped history folds back
//     to exactly the flat default.
//
// The result is clamped to [MinEarnedResumeBudget, MaxEarnedResumeBudget]. Fewer than two
// fired launches carries no interval evidence, so the base DefaultMaxResumeAttempts is
// returned unchanged — the first resumes of a fresh session are never rationed on
// evidence that does not exist yet.
func EarnedResumeBudget(history []Attempt) int {
	launches := launchUnixTimes(history)
	if len(launches) < 2 {
		return DefaultMaxResumeAttempts
	}
	budget := DefaultMaxResumeAttempts
	for i := 1; i < len(launches); i++ {
		prev, cur := launches[i-1], launches[i]
		if prev <= 0 || cur <= 0 {
			continue // unmeasurable gap: neutral, never a penalty
		}
		gap := cur - prev
		if gap < 0 {
			gap = -gap // ledger rows need not be strictly ordered; distance is what matters
		}
		if gap >= ProgressGapSeconds {
			budget++
		} else {
			budget--
		}
	}
	if budget < MinEarnedResumeBudget {
		budget = MinEarnedResumeBudget
	}
	if budget > MaxEarnedResumeBudget {
		budget = MaxEarnedResumeBudget
	}
	return budget
}

// launchUnixTimes is the ordered timestamps of the fired launches in a history — the raw
// signal EarnedResumeBudget folds. It keeps only rows that count as a launch (the same
// IsLaunch rule CountAttempts uses), preserving ledger order so the gaps between
// consecutive resumes are read as they actually happened.
func launchUnixTimes(history []Attempt) []int64 {
	var out []int64
	for _, a := range history {
		if a.IsLaunch() {
			out = append(out, a.UnixSeconds)
		}
	}
	return out
}

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

// ResumeModels attributes a model to a resume's progress. Given the real-turn timestamps
// and the parallel model of each turn (turnModels[i] ran the turn at turnUnixTimes[i]), it
// returns the LATEST model that ran strictly after sinceUnix — the one the resume is on
// now — and the distinct models seen after it in first-appearance order. A resume whose
// turns span more than one model (a re-home mid-flight) yields a multi-element distinct
// set, so a monitor can tell "took, on the intended model" from "took, but drifted." A
// zero/negative sinceUnix, mismatched slice lengths, or no post-launch turn yields the
// empty string and nil: there is no resume progress to attribute a model to. It answers
// only WHICH model the recovery turns ran on — never whether that is the RIGHT model; the
// expectation policy lives with the operator's caller, not this leaf.
func ResumeModels(turnUnixTimes []int64, turnModels []string, sinceUnix int64) (latest string, distinct []string) {
	if sinceUnix <= 0 || len(turnUnixTimes) != len(turnModels) {
		return "", nil
	}
	var latestUnix int64
	seen := map[string]bool{}
	for i, t := range turnUnixTimes {
		if t <= sinceUnix {
			continue
		}
		m := turnModels[i]
		if m == "" {
			continue
		}
		if !seen[m] {
			seen[m] = true
			distinct = append(distinct, m)
		}
		if t >= latestUnix {
			latestUnix = t
			latest = m
		}
	}
	return latest, distinct
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
// an auth wall stays burned.
//
// The cap is progress-EARNED, not flat (#3124): maxAttempts <= 0 means "use the earned
// budget", so the cap is EarnedResumeBudget(history) — a session making real progress
// between resumes earns more attempts, a thrashing one earns fewer, rather than every
// session getting the same blind DefaultMaxResumeAttempts. A positive maxAttempts is an
// explicit operator/caller override and is honored literally, un-earned.
func RetryGate(history []Attempt, outcome Outcome, maxAttempts int) RetryDecision {
	// #2178: a re-arm reclaim row zeroes the budget accrued before it — gate on the suffix after
	// the last marker so the earned budget, settled scan, and attempt count all start fresh.
	history = afterLastRearm(history)
	if maxAttempts <= 0 {
		maxAttempts = EarnedResumeBudget(history)
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
		return RetryDecision{Blocked: false, Reason: "first resume"}
	}
	if attempts >= maxAttempts {
		return RetryDecision{Blocked: true, Reason: fmt.Sprintf("attempt cap reached (%d/%d)", attempts, maxAttempts)}
	}
	switch outcome {
	case OutcomeRecoverable:
		return RetryDecision{Blocked: false,
			Reason: fmt.Sprintf("last resume failed recoverably; attempt %d/%d", attempts+1, maxAttempts)}
	case OutcomeCancelled:
		// The deny axis (#3354): a deliberate stop is intent, not a failure — relaunching
		// would override the operator's decision. Distinct from the operator-settled ledger
		// override above (a hand-written consolidate row) and from the auth wall below (an
		// environmental wall no retry can fix): this one the session's own terminal turn proves.
		return RetryDecision{Blocked: true, Reason: "intentionally cancelled — do not relaunch"}
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
	case f.Outcome == OutcomeUnrecoverable, f.Outcome == OutcomeCancelled:
		// Cancelled folds to gave-up for the same operator-facing meaning: no automatic
		// resume will fire again; a human owns whatever happens next (#3354).
		return ResumeGaveUp
	case f.Attempts >= max:
		return ResumeGaveUp
	case f.Outcome == OutcomeRecoverable:
		return ResumeReStranded
	default:
		return ResumeLaunched
	}
}
