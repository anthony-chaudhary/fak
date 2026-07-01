// Package attemptbudget is a pure fold over one issue's attempt history: given a
// bounded budget and the recorded attempts (each carrying the failure class it
// ended in), it decides whether the issue is still dispatchable, COOLING_DOWN
// under a failure-class-aware backoff window, or HELD for human triage -- so a
// repeatedly failing issue stops burning workers once it crosses the budget,
// instead of being re-offered forever (#1777), and so different kinds of
// failure cool down at different rates instead of all sharing one window
// (#1778). It never decides WHY an attempt failed; it only counts, classifies,
// and thresholds facts the caller already gathered. Pure: same Input in, same
// Decision out; zero I/O, zero clock reads -- the caller supplies "now" as
// data (Input.NowUnix), the same discipline internal/dispatchorder and
// internal/skipledger already use for clock-dependent folds.
package attemptbudget

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// Status is the closed dispatchability verdict for one issue.
type Status string

const (
	StatusDispatchable Status = "dispatchable"
	// StatusCoolingDown means the issue is still under its attempt budget but
	// its last failure's class-specific backoff window has not yet elapsed --
	// distinct from StatusHeld, which is a hard stop on attempt count.
	StatusCoolingDown Status = "cooling_down"
	StatusHeld        Status = "held"
)

// FailureClass is the closed, caller-facing backoff vocabulary this package
// assigns a window to. A caller's raw Attempt.FailureClass string is mapped
// onto one of these (see classify); an unrecognized string maps to
// FailureClassOther, so an unknown failure class never crashes or gets
// silently treated as the least cautious window.
type FailureClass string

const (
	// FailureClassAuth: the attempt failed on an authentication/authorization
	// problem (bad/expired credentials, permission denied). These often need a
	// human to rotate a secret or grant access, so they get the LONGEST
	// default backoff -- retrying fast just burns another worker on the same
	// wall.
	FailureClassAuth FailureClass = "auth"
	// FailureClassMerge: the attempt failed on a merge/rebase conflict against
	// trunk. A moderate backoff gives concurrent peers time to land, after
	// which trunk has likely moved and a retry may no longer conflict.
	FailureClassMerge FailureClass = "merge"
	// FailureClassTest: the attempt failed a test run. The shortest default
	// backoff -- test failures are the most likely to be flaky or fixed by a
	// quick follow-up, so the issue should come back around soonest.
	FailureClassTest FailureClass = "test"
	// FailureClassAmbiguousScope: the attempt failed because the issue's scope
	// was unclear or contested (e.g. it collided with a concurrent peer's
	// area, or the worker could not determine the target package). A long
	// backoff, close to auth's, since re-dispatching immediately just repeats
	// the same ambiguity.
	FailureClassAmbiguousScope FailureClass = "ambiguous_scope"
	// FailureClassOther: any failure class the caller supplies that this
	// package does not recognize. Gets the default/moderate window rather
	// than being coerced into one of the named classes above.
	FailureClassOther FailureClass = "other"
)

// classify maps a caller-supplied raw failure-class string onto the closed
// FailureClass vocabulary via case-insensitive substring checks, so callers
// already using descriptive tags (internal/timeoutphase-style, e.g.
// "test_failure", "auth_error", "merge_conflict") don't have to pre-normalize
// onto attemptbudget's exact tokens.
func classify(raw string) FailureClass {
	low := strings.ToLower(raw)
	switch {
	case strmatch.ContainsAny(low, "auth", "credential", "permission", "unauthorized", "forbidden"):
		return FailureClassAuth
	case strmatch.ContainsAny(low, "merge", "conflict", "rebase"):
		return FailureClassMerge
	case strmatch.ContainsAny(low, "test", "assert"):
		return FailureClassTest
	case strmatch.ContainsAny(low, "ambiguous", "scope"):
		return FailureClassAmbiguousScope
	default:
		return FailureClassOther
	}
}

// DefaultBackoffSeconds is the closed, total default policy: how long an
// issue cools down after its LAST recorded attempt, keyed by that attempt's
// classified FailureClass. Auth and ambiguous-scope failures usually need a
// human (rotate a credential, resolve a scope collision) so they cool down
// the longest; merge conflicts cool down long enough for a concurrent peer to
// land; test failures cool down the shortest, since they are the cheapest to
// retry and the most likely to be transient. Every FailureClass has an entry
// -- callers needing a different policy pass their own via Input.Backoff.
var DefaultBackoffSeconds = map[FailureClass]int64{
	FailureClassAuth:           4 * 3600, // 4h: needs a human to rotate/grant
	FailureClassMerge:          30 * 60,  // 30m: give trunk time to move
	FailureClassTest:           10 * 60,  // 10m: cheapest to retry, often flaky
	FailureClassAmbiguousScope: 2 * 3600, // 2h: needs a human to resolve scope
	FailureClassOther:          60 * 60,  // 1h: moderate default
}

// Attempt is one recorded try at an issue: the failure class it ended in (the
// caller's vocabulary -- e.g. "test_failure", "timeout", "merge_conflict") and
// when it happened. An attempt that SUCCEEDED should simply not be recorded
// here; this package only ever sees the failed history.
type Attempt struct {
	FailureClass string `json:"failure_class"`
	AtUnix       int64  `json:"at_unix"`
}

// Input is one issue's attempt-budget facts.
type Input struct {
	IssueID  string    `json:"issue_id"`
	Attempts []Attempt `json:"attempts"`
	// Budget is the maximum number of recorded (failed) attempts allowed
	// before the issue is held for triage. A Budget <= 0 means unlimited --
	// the issue is never held on attempt count alone.
	Budget int `json:"budget"`
	// NowUnix is the caller-supplied clock reading used for backoff math (is
	// the last attempt's class-specific cooldown window still open?). Zero
	// means the caller does not care about cooldown timing -- the Decision
	// still reports the backoff window that WOULD apply, but Status never
	// becomes StatusCoolingDown on a zero clock (there is no "now" to compare
	// against). This package never reads a clock itself.
	NowUnix int64 `json:"now_unix,omitempty"`
	// Backoff optionally overrides DefaultBackoffSeconds for this issue only
	// (e.g. an operator tuning one noisy issue's windows). A nil map uses
	// DefaultBackoffSeconds.
	Backoff map[FailureClass]int64 `json:"backoff,omitempty"`
}

// Decision is the verdict for one issue.
type Decision struct {
	IssueID          string `json:"issue_id"`
	Status           Status `json:"status"`
	AttemptCount     int    `json:"attempt_count"`
	Budget           int    `json:"budget"`
	LastFailureClass string `json:"last_failure_class,omitempty"`
	// BackoffClass is the closed FailureClass the LastFailureClass was
	// classified into (empty when there is no recorded attempt yet).
	BackoffClass FailureClass `json:"backoff_class,omitempty"`
	// BackoffSeconds is the cooldown window that failure class carries under
	// the effective policy (Input.Backoff, or DefaultBackoffSeconds).
	BackoffSeconds int64 `json:"backoff_seconds,omitempty"`
	// CooldownUntilUnix is the last attempt's AtUnix plus BackoffSeconds --
	// the earliest time this issue should be re-offered. Zero when there is
	// no recorded attempt.
	CooldownUntilUnix int64 `json:"cooldown_until_unix,omitempty"`
}

// Decide folds one issue's Input into a Decision, in this order: HELD once
// AttemptCount reaches Budget (Budget > 0) -- a hard stop that overrides
// cooldown; otherwise COOLING_DOWN when NowUnix is positive and still before
// the last attempt's class-specific CooldownUntilUnix; otherwise
// DISPATCHABLE. The Decision always carries the LAST recorded attempt's
// classified BackoffClass/BackoffSeconds/CooldownUntilUnix (when there is a
// recorded attempt) regardless of Status, so a report can show the policy
// even for a HELD issue.
func Decide(in Input) Decision {
	d := Decision{
		IssueID:      in.IssueID,
		Status:       StatusDispatchable,
		AttemptCount: len(in.Attempts),
		Budget:       in.Budget,
	}
	if len(in.Attempts) > 0 {
		last := in.Attempts[len(in.Attempts)-1]
		d.LastFailureClass = last.FailureClass
		d.BackoffClass = classify(last.FailureClass)
		d.BackoffSeconds = backoffSeconds(d.BackoffClass, in.Backoff)
		d.CooldownUntilUnix = last.AtUnix + d.BackoffSeconds
		if in.NowUnix > 0 && in.NowUnix < d.CooldownUntilUnix {
			d.Status = StatusCoolingDown
		}
	}
	if in.Budget > 0 && d.AttemptCount >= in.Budget {
		d.Status = StatusHeld
	}
	return d
}

// backoffSeconds resolves the effective window for class under an optional
// per-issue override map, falling back to DefaultBackoffSeconds and finally
// to FailureClassOther's default if the class is somehow missing from both
// (never zero, so a caller can't be left with "no cooldown at all" by an
// incomplete override map).
func backoffSeconds(class FailureClass, override map[FailureClass]int64) int64 {
	if override != nil {
		if s, ok := override[class]; ok {
			return s
		}
	}
	if s, ok := DefaultBackoffSeconds[class]; ok {
		return s
	}
	return DefaultBackoffSeconds[FailureClassOther]
}

// Report is the batch verdict over many issues.
type Report struct {
	Decisions         []Decision `json:"decisions"`
	DispatchableCount int        `json:"dispatchable_count"`
	CoolingDownCount  int        `json:"cooling_down_count"`
	HeldCount         int        `json:"held_count"`
}

// DecideAll folds a batch of issues, in the order given, into a Report.
func DecideAll(inputs []Input) Report {
	rep := Report{Decisions: make([]Decision, 0, len(inputs))}
	for _, in := range inputs {
		d := Decide(in)
		switch d.Status {
		case StatusHeld:
			rep.HeldCount++
		case StatusCoolingDown:
			rep.CoolingDownCount++
		default:
			rep.DispatchableCount++
		}
		rep.Decisions = append(rep.Decisions, d)
	}
	return rep
}
