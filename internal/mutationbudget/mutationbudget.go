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

// Budget is a snapshot of the remaining GitHub mutation allowance in the current
// rate-limit window: how many calls are Remaining, the window's Limit, and the
// unix second at which the window Resets (and Remaining refills to Limit).
type Budget struct {
	Remaining   int   // mutations left in the current window
	ResetAtUnix int64 // unix seconds at which the window resets
	Limit       int   // the window's total mutation allowance
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
