// Package mutationbudget is the throttle guard for live GitHub mutations: it gates
// each planned burst of close/comment calls behind the remaining API budget so a
// parallel-agent fleet cannot exhaust the rate limit mid-batch and leave the work
// half-executed. It is one leaf of the "safe 400 GitHub issues/hour parallel-agent
// throughput" program (issue #1825, fleet-400iph).
//
// The GitHub REST API meters mutations against a per-window budget: a limit, the
// number remaining in the current window, and the unix time the window resets. A
// burst of N mutations spends N of that budget. If N were allowed to drive the
// remaining count to zero (or below), every subsequent worker in the fleet — and
// any retry — would be hard-blocked until the window reset, and a batch that had
// already closed some issues but not commented on them would be stranded partway.
//
// The guard's job is to refuse a burst BEFORE it starts when finishing it would
// leave less than a reserve buffer of budget. Holding the whole burst is the safe
// failure: the operator waits for the reset (or reduces the batch) rather than
// discovering the limit was hit on mutation 9 of 12.
//
// The guard is a pure decision function: it takes the budget, the planned mutation
// count, and the reserve as inputs and returns an Allow/HOLD Decision with an
// actionable reason. It calls NO GitHub API and reads NO clock inside the decision
// — the "reset in Nm" window it reports is computed from an explicit now, via
// ResetInSec, so the same inputs always yield the same verdict. Wiring this guard
// into live gh calls is a later ticket; this package is only the guard logic.
//
// It is stdlib-only and imports nothing internal — off the hot path.
package mutationbudget

import (
	"fmt"
	"time"
)

// Budget is a snapshot of the remaining GitHub API allowance in the current
// rate-limit window: how many calls are Remaining, the window's Limit, and the
// unix second at which the window Resets (and Remaining refills to Limit).
type Budget struct {
	Remaining   int   // API calls left in the current window
	ResetAtUnix int64 // unix seconds at which the window resets
	Limit       int   // the window's total API-call allowance
}

// ResetInSec reports how many seconds remain until the window resets, measured
// from nowUnix. It never returns a negative value: a reset time already in the
// past (or a now past the reset) yields 0, meaning "resets now / already reset".
func (b Budget) ResetInSec(nowUnix int64) int64 {
	d := b.ResetAtUnix - nowUnix
	if d < 0 {
		return 0
	}
	return d
}

// Decision is the guard's verdict on one planned burst: whether it may proceed
// (Allow) and, when held, an actionable Reason naming the shortfall and the reset
// window. It also carries the numbers the verdict turned on so an operator or a
// caller can act without re-deriving them.
type Decision struct {
	Allow          bool   // true when the burst may proceed
	Reason         string // actionable message; "" is never used — always populated
	Remaining      int    // budget remaining before the burst
	Planned        int    // mutations the burst would spend
	Reserve        int    // minimum budget that must survive the burst
	AfterRemaining int    // Remaining - Planned, the budget the burst would leave
}

// HourlyPlan is the dry-run rate-limit estimate for one dispatch hour. It names
// the GitHub calls the close/create/comment machinery plans to spend before any
// live mutation runs. Fetches are not mutations, but they still consume the same
// API window and must be counted against the hourly budget.
type HourlyPlan struct {
	Creates  int `json:"creates"`
	Comments int `json:"comments"`
	Closes   int `json:"closes"`
	Labels   int `json:"labels"`
	Fetches  int `json:"fetches"`
}

// TotalCalls returns all GitHub API calls in the planned hour, with negative
// counts clamped to zero so a caller cannot subtract calls from the budget.
func (p HourlyPlan) TotalCalls() int {
	return clampNonNegative(p.Creates) +
		clampNonNegative(p.Comments) +
		clampNonNegative(p.Closes) +
		clampNonNegative(p.Labels) +
		clampNonNegative(p.Fetches)
}

// MutationCalls returns the write-side calls in the planned hour. Fetches are
// reported separately so an operator can tell write pressure from read pressure.
func (p HourlyPlan) MutationCalls() int {
	return clampNonNegative(p.Creates) +
		clampNonNegative(p.Comments) +
		clampNonNegative(p.Closes) +
		clampNonNegative(p.Labels)
}

// HourlyEstimate is the pure dry-run verdict for one planned dispatch hour.
type HourlyEstimate struct {
	Allow          bool       `json:"allow"`
	Warning        string     `json:"warning,omitempty"`
	Reason         string     `json:"reason"`
	Plan           HourlyPlan `json:"plan"`
	TotalCalls     int        `json:"total_calls"`
	MutationCalls  int        `json:"mutation_calls"`
	FetchCalls     int        `json:"fetch_calls"`
	Remaining      int        `json:"remaining"`
	Reserve        int        `json:"reserve"`
	AfterRemaining int        `json:"after_remaining"`
}

// EstimateHour checks a planned dispatch hour against the remaining GitHub API
// window. It is deterministic: the reset clock is injected via nowUnix, and it
// performs no I/O. A non-allowing estimate carries a rate-limit warning before a
// live close/comment/label/create wave can strand itself mid-hour.
func EstimateHour(budget Budget, plan HourlyPlan, minReserve int, nowUnix int64) HourlyEstimate {
	reserve := minReserve
	if reserve < 0 {
		reserve = 0
	}
	total := plan.TotalCalls()
	mutations := plan.MutationCalls()
	fetches := clampNonNegative(plan.Fetches)
	after := budget.Remaining - total
	out := HourlyEstimate{
		Allow:          true,
		Plan:           plan,
		TotalCalls:     total,
		MutationCalls:  mutations,
		FetchCalls:     fetches,
		Remaining:      budget.Remaining,
		Reserve:        reserve,
		AfterRemaining: after,
	}
	if total == 0 {
		out.Reason = fmt.Sprintf("ALLOW: 0 planned GitHub API calls spend nothing (remaining %d, reserve %d)",
			budget.Remaining, reserve)
		return out
	}
	if budget.Remaining < reserve {
		out.Allow = false
		out.Warning = fmt.Sprintf("RATE_LIMIT_WARNING: remaining %d already below reserve %d before planned dispatch hour; resets in %s",
			budget.Remaining, reserve, humanizeSec(budget.ResetInSec(nowUnix)))
		out.Reason = hourlyReason(plan, total, mutations, fetches, out.Warning)
		return out
	}
	if after < reserve {
		out.Allow = false
		out.Warning = fmt.Sprintf("RATE_LIMIT_WARNING: planned dispatch hour needs %d GitHub API calls and would leave %d < reserve %d; resets in %s",
			total, after, reserve, humanizeSec(budget.ResetInSec(nowUnix)))
		out.Reason = hourlyReason(plan, total, mutations, fetches, out.Warning)
		return out
	}
	out.Reason = fmt.Sprintf("ALLOW: planned dispatch hour needs %d GitHub API calls (mutations=%d fetches=%d) and leaves %d >= reserve %d",
		total, mutations, fetches, after, reserve)
	return out
}

// Guard decides whether plannedMutations may run against budget while keeping at
// least minReserve of the window's remaining allowance in hand. It is
// deterministic: nowUnix supplies the clock so the reset window in a HOLD reason is
// computed without reading a live clock, and the same inputs always yield the same
// Decision.
//
// The rule is a single inequality: HOLD when finishing the burst would drop the
// remaining budget below the reserve, i.e. when
//
//	remaining - plannedMutations < minReserve
//
// Equivalently, ALLOW exactly when afterRemaining (remaining - planned) >= reserve.
// So the boundary is inclusive on the allow side: landing exactly on the reserve is
// permitted; one below it is held.
//
// Edge cases:
//   - Zero (or negative) plannedMutations spends nothing, so it always ALLOWs
//     regardless of remaining — there is nothing to throttle.
//   - When remaining is already below the reserve, any positive burst HOLDs (it can
//     only make an already-thin budget thinner); this is reported distinctly so the
//     operator knows the budget was under reserve before the burst, not because of
//     it.
//   - A negative minReserve is clamped to 0: you cannot require a negative buffer.
func Guard(budget Budget, plannedMutations, minReserve int, nowUnix int64) Decision {
	reserve := minReserve
	if reserve < 0 {
		reserve = 0
	}
	after := budget.Remaining - plannedMutations
	d := Decision{
		Remaining:      budget.Remaining,
		Planned:        plannedMutations,
		Reserve:        reserve,
		AfterRemaining: after,
	}

	// Nothing to spend: always allow. Guard against a zero/negative plan up front
	// so a caller that passes an empty batch is never held.
	if plannedMutations <= 0 {
		d.Allow = true
		d.Reason = fmt.Sprintf("ALLOW: 0 planned mutations spend nothing (remaining %d, reserve %d)",
			budget.Remaining, reserve)
		return d
	}

	resetIn := budget.ResetInSec(nowUnix)

	// Already under reserve before the burst: the budget was thin before we planned
	// anything, so name that distinctly.
	if budget.Remaining < reserve {
		d.Allow = false
		d.Reason = fmt.Sprintf(
			"HOLD: remaining %d already below reserve %d before any of the %d planned mutations; resets in %s — wait for the window to reset",
			budget.Remaining, reserve, plannedMutations, humanizeSec(resetIn))
		return d
	}

	// Finishing the burst would breach the reserve: hold the whole batch.
	if after < reserve {
		d.Allow = false
		d.Reason = fmt.Sprintf(
			"HOLD: %d planned mutations would leave %d < reserve %d; resets in %s — wait or reduce batch",
			plannedMutations, after, reserve, humanizeSec(resetIn))
		return d
	}

	// Ample budget: the burst finishes with at least the reserve intact.
	d.Allow = true
	d.Reason = fmt.Sprintf(
		"ALLOW: %d planned mutations leave %d >= reserve %d (remaining %d)",
		plannedMutations, after, reserve, budget.Remaining)
	return d
}

// String renders the decision as one operator-facing line: the reason (which
// already leads with ALLOW/HOLD and names the numbers), so a caller can log or
// print d.String() directly.
func (d Decision) String() string {
	return d.Reason
}

func hourlyReason(plan HourlyPlan, total, mutations, fetches int, warning string) string {
	return fmt.Sprintf("%s (creates=%d comments=%d closes=%d labels=%d fetches=%d; mutations=%d fetches=%d total=%d)",
		warning,
		clampNonNegative(plan.Creates),
		clampNonNegative(plan.Comments),
		clampNonNegative(plan.Closes),
		clampNonNegative(plan.Labels),
		clampNonNegative(plan.Fetches),
		mutations,
		fetches,
		total)
}

func clampNonNegative(n int) int {
	if n < 0 {
		return 0
	}
	return n
}

// humanizeSec renders a non-negative second count as a compact human window for a
// reset message: "now" at zero, whole seconds under a minute, whole minutes under
// an hour, and hours-plus-minutes above. It is presentation only — the guard's
// verdict never turns on it.
func humanizeSec(sec int64) string {
	if sec <= 0 {
		return "now"
	}
	d := time.Duration(sec) * time.Second
	if d < time.Minute {
		return fmt.Sprintf("%ds", sec)
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm", sec/60)
	}
	h := sec / 3600
	m := (sec % 3600) / 60
	if m == 0 {
		return fmt.Sprintf("%dh", h)
	}
	return fmt.Sprintf("%dh%dm", h, m)
}
