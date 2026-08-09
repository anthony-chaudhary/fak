package ctxresidency

// pressure.go — issue #4037: carve the RECLAIMABLE-reserved slack out of TRUE
// consumption, so a live context meter can show three bands instead of one
// lumped number and can turn red only on pressure that is actually pinned.
//
// # The gap this closes
//
// Every ingredient for a banded context meter already existed and none of them
// were joined: cmd/fak's ablateBarCells is a three-band stacked bar renderer
// (wired to cache attribution), internal/gateway's ctxvalue carries a 50%/80%
// util → any/bounded/checkpoint pressure scale, and THIS package already
// classifies every live span as resident (pinned — a live cachemeta dependent
// parents its K/V, so evicting it invalidates that dependent) or evictable (no
// dependents — a clean eviction candidate). What was missing is the carve: the
// query summed BOTH classes into one Snapshot.ResidentTokens, so a window at
// 90% that is 85 points of clean eviction candidates read identically to one
// that is 85 points of pinned spans. The first is fine; the second is stuck.
// Only the second is pressure.
//
// # No new thresholds, and no second bar renderer
//
// Two deliberate non-additions, because the whole point is to JOIN the parts
// that exist rather than grow parallel ones:
//
//   - The pressure scale is INJECTED (boundedPct / checkpointPct), never
//     declared here. The single source of truth stays internal/gateway's
//     ctxStepBoundedPercent / ctxStepCheckpointPercent (the latter is itself
//     compactionNudgeNearPercent, so the agent query, the per-turn nudge, and
//     this meter cannot drift apart). A caller that supplies no scale gets
//     PressureUnknown, never a fabricated tier.
//   - There is no bar-cell function here. Bands emits TOKEN counts in exactly
//     the shape cmd/fak's existing ablateBarCells already consumes, so the
//     render side composes with the renderer that shipped rather than a second
//     copy of it:
//
//     committed, reclaimable, free := ablateBarCells(
//     float64(b.CommittedTokens), float64(b.ReclaimableTokens),
//     float64(b.BudgetTokens), barW)
//
//     ablateBarCells fills (committed+reclaimable)/budget of the bar and splits
//     that fill between the two positive contributions, returning the unfilled
//     remainder as the third band — which is precisely committed / reclaimable
//     / free. It needs no change to serve this meter.
//
// Everything below is pure and time-free: no I/O, no kernel state, no mutation.
// Like Query, reading residency can never launder a held span back into
// context.

// PressureClass is the closed verdict this fold emits. The vocabulary MIRRORS
// internal/gateway's StepClass ("any" / "bounded" / "checkpoint" / "unknown")
// so a gateway-side caller maps it across with a plain string conversion. It is
// re-declared rather than imported because gateway is a tier-4 integrator and
// this package is a tier-3 composer: importing it would be an upward edge the
// architest gate refuses. The VALUES are the contract; the thresholds that
// select between them are not owned here at all (see the note above).
type PressureClass string

const (
	// PressureAny — the pinned share of the window is small: a wide step fits.
	PressureAny PressureClass = "any"
	// PressureBounded — the pinned share crossed the bounded rung: keep new
	// residency deliberate.
	PressureBounded PressureClass = "bounded"
	// PressureCheckpoint — the pinned share crossed the checkpoint rung. This is
	// the only red state, and reaching it means eviction CANNOT buy the window
	// back: the tokens are held by live dependents.
	PressureCheckpoint PressureClass = "checkpoint"
	// PressureUnknown — no budget configured, or no threshold scale supplied.
	// Fail closed: the meter says it cannot decide instead of guessing a tier.
	PressureUnknown PressureClass = "unknown"
)

// Bands is the three-band split of a context budget plus the verdict graded on
// the non-reclaimable band alone.
//
// The bands partition the budget: CommittedTokens + ReclaimableTokens +
// FreeTokens == BudgetTokens whenever a budget is configured and residency has
// not overrun it. An overrun (resident past the budget — legal, the budget is a
// compaction trigger and not a hard wall) clamps FreeTokens to 0 rather than
// reporting negative free space, so the bands stay renderable.
type Bands struct {
	// CommittedTokens is TRUE consumption: resident spans with at least one live
	// cachemeta dependent. Evicting one invalidates that dependent, so these
	// tokens are not recoverable by shedding.
	CommittedTokens int
	// ReclaimableTokens is reserved-but-idle slack: resident spans with no live
	// dependents (StateEvictable). They occupy the window now and a clean
	// eviction returns every one of them.
	ReclaimableTokens int
	// FreeTokens is the unspent remainder of the budget.
	FreeTokens int
	// BudgetTokens is the configured resident budget; 0 = none configured.
	BudgetTokens int

	// TruePressurePct is CommittedTokens as a share of the budget — the number
	// Class is graded on, and the one an operator should read as "how stuck am
	// I". 0 when no budget is configured.
	TruePressurePct float64
	// LumpedPct is (Committed+Reclaimable) as a share of the budget: the SINGLE
	// number a one-band meter showed. It is reported beside TruePressurePct
	// precisely so the gap between them is legible — that gap is the slack a
	// lumped meter was hiding.
	LumpedPct float64

	// Class is the verdict, graded on TruePressurePct ONLY.
	Class PressureClass
}

// Reclaimable reports whether any of the resident window can be bought back by
// a clean eviction. A meter can use it to justify why a visibly full bar is
// not red.
func (b Bands) Reclaimable() bool { return b.ReclaimableTokens > 0 }

// Pressure splits this snapshot's residency into committed / reclaimable / free
// against budgetTokens and grades the committed band on the caller's scale. It
// is the Snapshot-shaped front door to PressureBands; see that function for the
// grading rules and the fail-closed cases.
func (s Snapshot) Pressure(budgetTokens, boundedPct, checkpointPct int) Bands {
	return PressureBands(s.CommittedTokens, s.ReclaimableTokens, budgetTokens, boundedPct, checkpointPct)
}

// PressureBands is the pure fold, split out from Snapshot.Pressure so the
// policy is unit-testable without a kvmmu context (the adviseCtxStep pattern).
//
// boundedPct / checkpointPct are the caller's existing pressure scale as whole
// percents (gateway passes ctxStepBoundedPercent / ctxStepCheckpointPercent —
// 50 and 80). Grading walks the rungs highest-first against TruePressurePct, so
// an inverted or equal pair still yields a defined answer.
//
// It fails closed to PressureUnknown — with the bands still filled in, because
// the split is a measurement and stays true even when the verdict cannot be —
// in exactly two cases:
//
//   - budgetTokens <= 0: no budget is configured, so there is no window edge to
//     be near and no denominator to grade against.
//   - checkpointPct <= 0 || boundedPct <= 0: the caller supplied no scale.
//     Grading against a zero rung would read EVERY window as checkpoint, which
//     is worse than admitting the scale is missing.
//
// Negative inputs clamp to zero rather than propagating into the arithmetic.
func PressureBands(committed, reclaimable, budgetTokens, boundedPct, checkpointPct int) Bands {
	committed = clampNonNeg(committed)
	reclaimable = clampNonNeg(reclaimable)
	budgetTokens = clampNonNeg(budgetTokens)

	b := Bands{
		CommittedTokens:   committed,
		ReclaimableTokens: reclaimable,
		BudgetTokens:      budgetTokens,
		Class:             PressureUnknown,
	}
	if budgetTokens == 0 {
		return b
	}
	if free := budgetTokens - committed - reclaimable; free > 0 {
		b.FreeTokens = free
	}
	b.TruePressurePct = float64(committed) * 100 / float64(budgetTokens)
	b.LumpedPct = float64(committed+reclaimable) * 100 / float64(budgetTokens)

	if checkpointPct <= 0 || boundedPct <= 0 {
		return b
	}
	// Integer comparison against the rung (committed*100 >= budget*pct) rather
	// than a float compare on TruePressurePct, so the boundary lands on exactly
	// the same side as gateway's adviseCtxStep, which grades the same way.
	switch {
	case committed*100 >= budgetTokens*checkpointPct:
		b.Class = PressureCheckpoint
	case committed*100 >= budgetTokens*boundedPct:
		b.Class = PressureBounded
	default:
		b.Class = PressureAny
	}
	return b
}

func clampNonNeg(n int) int {
	if n < 0 {
		return 0
	}
	return n
}
