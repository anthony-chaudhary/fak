package session

import "time"

// timebudget.go — the WALL-CLOCK twin of Budget (issue #1584, epic #1570 "managed
// context"). Budget governs TOKEN axes (turns/output/context/clarification-queries);
// TimeBudget governs the ORTHOGONAL axis a long-lived managed run also needs bounded:
// real elapsed time, which keeps ticking whether or not the model is spending tokens.
//
// # The gap it closes
//
// A managed run can promise "keep going for up to N tokens" (Budget) but has no way to
// promise "keep going for up to N wall-clock minutes" — and a hidden context reset
// (DebitUsage's context-exhaustion path, Recontinue's fresh-trace re-arm) already
// carries budget/priority/pace forward across that reset, but drops elapsed time on the
// floor: a fresh child trace today starts as if no time had passed. That is the gap
// #1584 names: "govern wall-clock budgets across context resets" needs a runtime
// continuity contract, not just a token counter that happens to also be reset-aware.
//
// # The persistence discipline (mirrors internal/dormancy.Stamp, not a new pattern)
//
// A naive elapsed-time counter kept only in memory resets to zero on every hidden
// restart — exactly the failure internal/dormancy.Stamp was built to avoid for the
// "how long was this dormant" measurement. TimeBudget reuses that discipline instead of
// inventing a second one: elapsed time is derived from real UNIX-NANOSECOND timestamps
// (StartedAtUnixNano / PausedAtUnixNano) plus an accumulated ElapsedNanos carry-forward,
// not a live ticking counter. A value built this way survives a JSON round-trip and a
// process restart byte-for-byte, because nothing here depends on a monotonic clock
// reading that dies with the process — only on wall-clock instants, exactly like Stamp.
//
// # The clock-injection discipline (mirrors internal/resume, internal/ctxplan)
//
// Every method that needs "now" takes it as an explicit time.Time parameter — there is
// no time.Now() baked into any decision here. This matches the rest of the session
// package family: resume.Plan takes IdleSeconds from the caller, dormancy.Stamp.GapAt
// takes now, and cmd/fak's sessionDurability takes an injected `now func() time.Time`
// (defaulting to time.Now only at the process boundary). TimeBudget follows the same
// shape so accounting-across-a-simulated-restart is deterministically testable with no
// sleep and no wall-clock flakiness.
//
// # Where it lives on State
//
// TimeBudget rides on session.State exactly like Budget: a new, independent field
// (State.Time), advisory to nothing else, additive to the wire shape (omitzero keeps a
// pre-#1584 State marshaling byte-identically to today). Recontinue carries it forward
// the same way it carries Budget/Generation forward across a reset — see
// (*Table) RecontinueAt in table.go — so a hidden restart does not zero the clock.

// TimeUnbounded is the sentinel for "no wall-clock limit configured" — the time-axis
// analogue of Unbounded. A TimeBudget at its zero value is unbounded (LimitNanos <= 0
// means off), matching Budget's "unconfigured axis is permissive" default.
const TimeUnbounded = -1

// Wall-clock stop/observe reasons — closed tokens in the same family as decide.go's
// Reason* constants, so "why did the time budget stop this run" is a checkable field.
const (
	// ReasonTimeBudgetExhausted marks a session Draining because its wall-clock
	// envelope elapsed. Distinct from ReasonBudgetTokens/Turns/Context so an operator
	// or a supervisor can tell a time-out from a token exhaustion at a glance.
	ReasonTimeBudgetExhausted = "TIME_BUDGET_EXHAUSTED"
)

// TimeBudget is a session's wall-clock allotment: how much real elapsed time it may run
// across its whole lineage (the original trace plus every Recontinue'd child), tracked
// independently of token usage. The zero value is unbounded (no envelope configured) —
// a State with no TimeBudget behaves exactly as it did before this field existed.
//
// Accounting is timestamp-based, not counter-based: StartedAtUnixNano marks the instant
// the CURRENT run (since the last resume/restart) began ticking, and ElapsedNanos is the
// carried-forward total from every PRIOR run in this lineage. Elapsed(now) sums the two,
// so "how much time has this managed run actually consumed" survives however many hidden
// restarts happened in between — each restart calls Pause (which folds the just-ended
// run's duration into ElapsedNanos) and the next boundary calls Resume (which re-arms
// StartedAtUnixNano at the fresh now). A TimeBudget that is never paused/resumed (a
// single continuous run) still works: Elapsed(now) is just now - StartedAtUnixNano.
type TimeBudget struct {
	// LimitNanos is the total wall-clock envelope in nanoseconds across the whole
	// lineage. <= 0 means TimeUnbounded (no limit) — the query/decide paths treat a
	// zero-value TimeBudget as fully permissive, matching Budget's Unbounded convention.
	LimitNanos int64 `json:"limit_nanos,omitempty"`
	// ElapsedNanos is the accumulated wall-clock time already consumed by every run in
	// this lineage BEFORE the current one started — the carry-forward a hidden restart
	// must not drop. It is only advanced by Pause (which folds the just-ended run's
	// duration in); Elapsed(now) adds the current run's live duration on top without
	// mutating this field, so repeated queries are idempotent.
	ElapsedNanos int64 `json:"elapsed_nanos,omitempty"`
	// StartedAtUnixNano is the wall-clock instant (unix nanoseconds) the CURRENT run
	// began ticking — set by Start on a fresh TimeBudget and re-armed by Resume after a
	// restart. Zero means the current run has not been started/resumed (no live tick;
	// Elapsed(now) then reports ElapsedNanos alone).
	StartedAtUnixNano int64 `json:"started_at_unix_nano,omitempty"`
}

// NewTimeBudget builds an unbounded TimeBudget (no envelope, not yet started). Use
// WithLimit to set the envelope and Start to arm the clock.
func NewTimeBudget() TimeBudget { return TimeBudget{} }

// WithLimit returns a copy with the wall-clock envelope set to limit. A non-positive
// limit clears the envelope (TimeUnbounded) rather than being stored as a negative
// number, so a caller passing TimeUnbounded or any other <=0 value gets the same
// unbounded behavior.
func (b TimeBudget) WithLimit(limit time.Duration) TimeBudget {
	if limit <= 0 {
		b.LimitNanos = 0
	} else {
		b.LimitNanos = int64(limit)
	}
	return b
}

// Bounded reports whether this axis carries a real envelope. Mirrors Budget's
// contextBounded/tokensUnbounded naming.
func (b TimeBudget) Bounded() bool { return b.LimitNanos > 0 }

// Running reports whether the current run is live-ticking (StartedAtUnixNano set). A
// TimeBudget that has been Paused (or never Started) is not running: Elapsed(now) then
// reports only the carried-forward total, never accruing more from a stale now.
func (b TimeBudget) Running() bool { return b.StartedAtUnixNano > 0 }

// Start arms the clock at now for a fresh (or previously-paused) TimeBudget, marking the
// current run's beginning. Starting an already-running budget is a no-op (it does not
// reset StartedAtUnixNano out from under a live run) — call Pause first to fold the
// live duration into ElapsedNanos before re-Starting, or use Resume, which does both.
func (b TimeBudget) Start(now time.Time) TimeBudget {
	if b.Running() {
		return b
	}
	b.StartedAtUnixNano = now.UnixNano()
	return b
}

// Pause folds the current run's live duration into ElapsedNanos and clears
// StartedAtUnixNano — the "hidden restart is about to happen" write. This is the exact
// moment a hidden context reset (Recontinue) or a process shutdown must call, mirroring
// how a real pause/resume cycle works: the elapsed time BEFORE the pause is durably
// carried, so a subsequent Resume (even in a freshly restarted process, reading this
// value back from persisted State/Descriptor JSON) resumes accounting from the true
// total, not from zero. Pausing an already-paused (or never-started) budget is a safe
// no-op: it neither double-counts nor loses time. now earlier than StartedAtUnixNano (a
// backwards wall-clock) clamps the folded delta to zero, the same conservative rule
// dormancy.Stamp.GapAt applies.
func (b TimeBudget) Pause(now time.Time) TimeBudget {
	if !b.Running() {
		return b
	}
	delta := now.UnixNano() - b.StartedAtUnixNano
	if delta < 0 {
		delta = 0
	}
	b.ElapsedNanos += delta
	b.StartedAtUnixNano = 0
	return b
}

// Resume re-arms the clock at now after a hidden restart, WITHOUT losing the
// ElapsedNanos carried across the gap — the read side of the same contract Pause writes.
// It is exactly Start when the budget is already paused (the common case: a restarted
// process rehydrates a TimeBudget from persisted state, where StartedAtUnixNano is
// necessarily 0, and Resume arms it at the current wall-clock instant). Calling Resume
// on an already-running budget is a no-op, matching Start.
func (b TimeBudget) Resume(now time.Time) TimeBudget { return b.Start(now) }

// Elapsed returns the total wall-clock time this lineage has consumed as of now: the
// carried-forward ElapsedNanos plus the current run's live duration (zero if not
// running). It never mutates the receiver, so polling it repeatedly (a `fak session
// status` call, a per-turn query) is side-effect-free — only Pause advances
// ElapsedNanos. A now earlier than StartedAtUnixNano clamps the live component to zero,
// so a backwards wall-clock never reports negative or inflated elapsed time.
func (b TimeBudget) Elapsed(now time.Time) time.Duration {
	total := b.ElapsedNanos
	if b.Running() {
		live := now.UnixNano() - b.StartedAtUnixNano
		if live > 0 {
			total += live
		}
	}
	return time.Duration(total)
}

// Remaining returns how much wall-clock budget is left as of now: LimitNanos -
// Elapsed(now), floored at zero. An unbounded budget (Bounded()==false) returns
// (0, false) — mirroring Budget.contextBounded's "not configured" signal — so a caller
// never mistakes an absent envelope for a zero remaining allotment.
func (b TimeBudget) Remaining(now time.Time) (remaining time.Duration, ok bool) {
	if !b.Bounded() {
		return 0, false
	}
	left := b.LimitNanos - int64(b.Elapsed(now))
	if left < 0 {
		left = 0
	}
	return time.Duration(left), true
}

// Exceeded reports whether the wall-clock envelope has been reached or passed as of now.
// An unbounded budget is never exceeded.
func (b TimeBudget) Exceeded(now time.Time) bool {
	if !b.Bounded() {
		return false
	}
	return b.Elapsed(now) >= time.Duration(b.LimitNanos)
}

// TimeQueryVerdict is the read-only "how much wall-clock budget is left, and should this
// run stop" answer — the time-axis analogue of QueryBudgetVerdict. It is deliberately a
// pure query (unlike Decide, it takes no lock and mutates nothing): a caller may ask it
// as often as it likes — at a turn boundary, from an operator CLI, or from a supervisor
// loop deciding whether to even re-admit a session after a restart — without perturbing
// the accounted time. Stopping the run (folding the elapsed time and transitioning the
// session) is a separate, explicit act via Table.DecideTimeBudget.
type TimeQueryVerdict struct {
	// Bounded reports whether a wall-clock envelope is configured at all. False means
	// every other field is a zero/permissive default (Exceeded=false, Remaining=0,
	// Unbounded reads as "no opinion" rather than "zero time left").
	Bounded bool `json:"bounded"`
	// Exceeded is true when Elapsed >= the configured Limit — the caller should treat
	// this exactly like a token-budget exhaustion (stop at the next boundary).
	Exceeded bool `json:"exceeded"`
	// Elapsed is the total wall-clock time consumed so far across this lineage.
	Elapsed time.Duration `json:"elapsed"`
	// Remaining is Limit-Elapsed, floored at zero; meaningless (reported as 0) when
	// Bounded is false.
	Remaining time.Duration `json:"remaining"`
	// Limit echoes the configured envelope (0 when unbounded).
	Limit time.Duration `json:"limit"`
}

// restoredPaused is the load-time half of the durable wall-clock contract: it clears
// any live StartedAtUnixNano WITHOUT folding a guessed duration into ElapsedNanos. A
// HIDDEN restart (a process crash, a redeploy) is exactly a Pause the dying process
// never got to call — the persisted StartedAtUnixNano is a wall-clock instant from a
// now-dead process, and the true gap between "last persisted Update" and "this restart"
// is unknowable here (this package takes no clock of its own). The conservative choice
// mirrors dormancy.Stamp's "unknown gap" discipline in spirit but inverted for safety on
// THIS axis: rather than assume the worst (charge the whole downtime against the
// budget, which would let an infrastructure outage silently exhaust a user's time
// envelope) or the best (assume zero downtime, silently extending it), it drops the
// unknowable gap and preserves exactly the ElapsedNanos that WAS durably recorded as of
// the last persist. The caller then explicitly Resumes at the real current now via
// Table.ResumeTimeBudget, which arms a fresh, accurate StartedAtUnixNano — so the
// session's total accounted time is a true lower bound on real elapsed time, never an
// inflated or silently-frozen one.
func (b TimeBudget) restoredPaused() TimeBudget {
	b.StartedAtUnixNano = 0
	return b
}

// Query answers "how much wall-clock budget is left as of now, and is the envelope
// exceeded" without mutating anything — the pure read half of the wall-clock gate, safe
// to call from `fak session status`, a per-turn check, or a supervisor deciding whether
// to re-admit a session after a hidden restart.
func (b TimeBudget) Query(now time.Time) TimeQueryVerdict {
	if !b.Bounded() {
		return TimeQueryVerdict{Bounded: false}
	}
	elapsed := b.Elapsed(now)
	remaining, _ := b.Remaining(now)
	return TimeQueryVerdict{
		Bounded:   true,
		Exceeded:  b.Exceeded(now),
		Elapsed:   elapsed,
		Remaining: remaining,
		Limit:     time.Duration(b.LimitNanos),
	}
}
