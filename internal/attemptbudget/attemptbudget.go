// Package attemptbudget is a pure fold over one issue's attempt history: given a
// bounded budget and the recorded attempts (each carrying the failure class it
// ended in), it decides whether the issue is still dispatchable or should be
// HELD for human triage -- so a repeatedly failing issue stops burning workers
// once it crosses the budget, instead of being re-offered forever (#1777). It
// never decides WHY an attempt failed; it only counts and thresholds facts the
// caller already gathered. Pure: same Input in, same Decision out; zero I/O,
// zero clock reads.
package attemptbudget

// Status is the closed dispatchability verdict for one issue.
type Status string

const (
	StatusDispatchable Status = "dispatchable"
	StatusHeld         Status = "held"
)

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
}

// Decision is the verdict for one issue.
type Decision struct {
	IssueID          string `json:"issue_id"`
	Status           Status `json:"status"`
	AttemptCount     int    `json:"attempt_count"`
	Budget           int    `json:"budget"`
	LastFailureClass string `json:"last_failure_class,omitempty"`
}

// Decide folds one issue's Input into a Decision: HELD once AttemptCount
// reaches Budget (Budget > 0), carrying the failure class of the LAST recorded
// attempt so triage knows what kept failing; DISPATCHABLE otherwise.
func Decide(in Input) Decision {
	d := Decision{
		IssueID:      in.IssueID,
		Status:       StatusDispatchable,
		AttemptCount: len(in.Attempts),
		Budget:       in.Budget,
	}
	if len(in.Attempts) > 0 {
		d.LastFailureClass = in.Attempts[len(in.Attempts)-1].FailureClass
	}
	if in.Budget > 0 && d.AttemptCount >= in.Budget {
		d.Status = StatusHeld
	}
	return d
}

// Report is the batch verdict over many issues.
type Report struct {
	Decisions         []Decision `json:"decisions"`
	DispatchableCount int        `json:"dispatchable_count"`
	HeldCount         int        `json:"held_count"`
}

// DecideAll folds a batch of issues, in the order given, into a Report.
func DecideAll(inputs []Input) Report {
	rep := Report{Decisions: make([]Decision, 0, len(inputs))}
	for _, in := range inputs {
		d := Decide(in)
		if d.Status == StatusHeld {
			rep.HeldCount++
		} else {
			rep.DispatchableCount++
		}
		rep.Decisions = append(rep.Decisions, d)
	}
	return rep
}
