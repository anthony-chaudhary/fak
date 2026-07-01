package ctxplan

import "math"

// pace.go — issue #1585 (managed-context epic #1570): feed a runtime PACE/THROUGHPUT signal
// into the context planner's Budget, as a sibling constraint threaded through the same
// PlanLayout/PlanCells call rather than a parallel planning path (the house pattern
// adaptive.go already set for task-difficulty sizing). RecommendBudget (adaptive.go) sizes
// W to how HARD a turn looks; ScaleBudgetForPace sizes W to how FAST the session is actually
// moving — an observed runtime rate, not a static forecast signal. The two are orthogonal and
// compose: a caller may RecommendBudget first, then ScaleBudgetForPace the result.
//
// THE SIGNAL. Ratio is a (0,1] fraction the caller derives from a real runtime observation —
// e.g. session.Pace's ThroughputRatio (measured tokens/sec against an expected baseline) or
// ThrottleRatio (a configured per-turn output cap against its baseline). ctxplan does not
// compute the ratio itself (that math lives in session, a tier-1 sibling ctxplan must not
// import): this file only consumes the already-computed (0,1] number and applies it to a
// Budget, exactly the seam CostModel/Tokenizer already use to let a higher tier plug in a
// real computation while ctxplan stays stdlib-only.
//
// THE FLOOR. "minimum resident context preserved" (the issue's Done condition) is enforced
// here, not left to the caller: MinResidentTokens is an explicit floor the scaled budget can
// never drop below, mirroring BudgetBounds.Floor's "never 0, never a lost fact" guarantee. A
// pace signal can shrink the window arbitrarily hard, but the structural pins + a minimal
// recency tail always still fit.
//
// PlanLayout/PlanCells are unchanged: a caller composes the scaled Budget BEFORE calling them
// (exactly like RecommendBudget's result is handed to Optimize), so this is additive — a
// caller who never calls ScaleBudgetForPace sees byte-identical planning.

// DefaultMinResidentTokens is the seed floor ScaleBudgetForPace applies when the caller passes
// a non-positive minimum: enough for a small pin set plus a couple of hot spans, the same
// order of magnitude as BudgetBounds' DefaultBudgetBounds().Floor (512). It is a conservative
// seed, not a tuned constant.
const DefaultMinResidentTokens = 512

// PaceBudget is the observed runtime pace/throughput signal the caller composes into a
// resident-context Budget. Ratio is the (0,1] fraction of the baseline the session is
// currently achieving — 1.0 means "at or above baseline, no constraint," a smaller value
// means the session is running slower or more constrained than its baseline and the resident
// window should shrink proportionally. MinResidentTokens is the floor the scaled budget must
// never drop below; <=0 falls back to DefaultMinResidentTokens.
type PaceBudget struct {
	// Ratio is the pace/throughput fraction in (0,1]. Values <=0 or NaN/Inf are treated as no
	// signal (1.0 — the base budget passes through unscaled), the same fail-closed posture
	// Difficulty.Score applies to a garbage input. A value > 1 clamps to 1 (a session running
	// FASTER than baseline is never a reason to WIDEN the window here — widening is
	// RecommendBudget's job, not the pace constraint's).
	Ratio float64
	// MinResidentTokens is the floor the scaled Tokens may never drop below. <=0 uses
	// DefaultMinResidentTokens.
	MinResidentTokens int
}

// normalizedRatio clamps Ratio into (0,1], failing closed to 1.0 (no scaling) on a
// non-positive or non-finite input — a poisoned pace signal must never zero out or invert the
// resident window.
func (p PaceBudget) normalizedRatio() float64 {
	r := p.Ratio
	if math.IsNaN(r) || math.IsInf(r, 0) || r <= 0 {
		return 1.0
	}
	if r > 1 {
		return 1.0
	}
	return r
}

// floor returns the effective minimum-resident-tokens floor: MinResidentTokens if positive,
// else DefaultMinResidentTokens.
func (p PaceBudget) floor() int {
	if p.MinResidentTokens > 0 {
		return p.MinResidentTokens
	}
	return DefaultMinResidentTokens
}

// ScaleBudgetForPace scales base's Tokens by the observed pace ratio, floored at
// MinResidentTokens (or DefaultMinResidentTokens) so a hard throttle never starves the
// resident view below a usable minimum — the "minimum resident context preserved" half of
// the issue's Done condition. An unthrottled signal (Ratio>=1, the default zero value) or a
// non-positive base returns base's Tokens unchanged, so a caller that never observes a pace
// constraint plans byte-for-byte as before ScaleBudgetForPace existed.
//
// Guarantees, mirroring RecommendBudget's:
//   - MONOTONE: a lower ratio never INCREASES Tokens.
//   - FLOORED: the result is never below min(base.Tokens, floor) — a base budget smaller than
//     the floor is left alone (there is nothing to preserve above what was already asked for).
//   - PURE: same (base, PaceBudget) => byte-identical Budget.
func ScaleBudgetForPace(base Budget, p PaceBudget) Budget {
	if base.Tokens <= 0 {
		return base
	}
	r := p.normalizedRatio()
	if r >= 1.0 {
		return base
	}
	floor := p.floor()
	if floor > base.Tokens {
		floor = base.Tokens
	}
	scaled := int(math.Round(float64(base.Tokens) * r))
	if scaled < floor {
		scaled = floor
	}
	return Budget{Tokens: scaled}
}

// PlanLayoutForPace is the one-call convenience: scale budget by the observed pace signal,
// then plan the O(1) layout view exactly as PlanLayout does. It performs no extra I/O and
// adds no planning logic beyond the Budget composition — the pace constraint is threaded
// through the SAME PlanLayout call every other caller uses, never a parallel path.
func (ix *Index) PlanLayoutForPace(f Forecast, budget Budget, cost CostModel, layout Layout, pace PaceBudget) Plan {
	return ix.PlanLayout(f, ScaleBudgetForPace(budget, pace), cost, layout)
}

// PlanCellsForPace is PlanLayoutForPace's non-layout peer: scale budget by the observed pace
// signal, then plan over the given candidate spans exactly as PlanCells does.
func PlanCellsForPace(spans []Span, f Forecast, budget Budget, cost CostModel, pace PaceBudget) Plan {
	return PlanCells(spans, f, ScaleBudgetForPace(budget, pace), cost)
}

// PlanCellsForPace is the Index-bounded peer of the package-level PlanCellsForPace: scale
// budget by the observed pace signal, then plan the bounded-probe view exactly as
// Index.PlanCells does. This is the entry point agent.SessionPlanner.PlanTurn uses on its
// non-Layout path, so a session's pace/throughput signal reaches the same bounded-compute
// planning the persistent index already does per turn.
func (ix *Index) PlanCellsForPace(f Forecast, b Budget, cost CostModel, opts ProbeOptions, pace PaceBudget) Plan {
	return ix.PlanCells(f, ScaleBudgetForPace(b, pace), cost, opts)
}
