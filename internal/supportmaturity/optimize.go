package supportmaturity

import "github.com/anthony-chaudhary/fak/internal/shipgate"

// This file binds plane B's R2 "optimize" regime (epic #1243, child #1253) onto the
// ladder concretely: an optimize-regime cell's work IS a long-running rsiloop whose
// promotion target is the cell's NEXT ladder rung (M4 -> M5 -> M6), kept or reverted by
// internal/shipgate on a witnessed metric gain. So "optimization" carries the same
// propose -> verify-correct -> measure-faster -> keep discipline everywhere, and a rung
// stays an ENVELOPE WITH A WITNESS: it can only move on shipgate's non-forgeable keep-bit,
// never on a self-report.
//
// Layering: this binding lives at the ladder's own tier and depends only on shipgate's
// keep-bit (tier 2), the SAME seam internal/rsiloop folds its measurement through. It does
// NOT import the rsiloop engine (tier 4) — instead it consumes the typed verdict an rsiloop
// run produces (rsiloop.Result.Final is a shipgate.Decision). The end-to-end witness — a
// real rsiloop run advancing a fixture cell's rung only on a measured gain — is exercised
// in optimize_test.go, which drives an actual rsiloop.Run and feeds its verdict here. This
// is the keep/revert rule; wiring a cell's optimization to literally run as an rsiloop
// (a maturity -> rsiloop.Harness adapter) is the router's next-action work (#1252, C9).

// OptimizeCell is a maturity cell in the R2 "optimize" regime: a cell that already holds a
// correctness witness (M4 or above) whose work is to climb toward Target one witnessed rung
// at a time. Current is the rung it honestly holds today (its measured floor); Target is the
// rung the optimization loop is climbing toward (its horizon — the rsiloop's, not a guess).
type OptimizeCell struct {
	// Name labels the (family x backend) cell for the journal / read-out.
	Name string
	// Current is the rung the cell honestly holds today — only a witnessed gain moves it.
	Current Rung
	// Target is the rung the optimize loop climbs toward (e.g. M6Parity). Promotion never
	// overshoots it: at or above Target there is nothing left to climb.
	Target Rung
}

// PromotionTarget is the cell's promotion goal for the next rsiloop run — the rung exactly
// one step above Current, capped at Target. ok is false when the cell already sits at or
// above its Target (the optimize loop's horizon is reached; there is no rung to climb).
func (c OptimizeCell) PromotionTarget() (next Rung, ok bool) {
	if !c.Current.Less(c.Target) {
		return c.Current, false
	}
	return c.Current + 1, true
}

// PromoteOnRun folds a completed rsiloop run's terminal shipgate verdict into the cell's
// rung. The rung ADVANCES by exactly one step toward Target only when the run KEPT — shipgate
// confirmed a strict, non-author measured gain over the baseline; any other verdict
// (REVERT or ESCALATE — a no-gain run) leaves the rung UNCHANGED. The cell cannot fabricate a
// promotion: the only input that moves the rung is shipgate's keep-bit, which Evaluate sets
// solely from a measured witness the loop did not author. A run with no rung left to climb
// (PromotionTarget ok=false) holds at Current regardless of verdict.
func (c OptimizeCell) PromoteOnRun(runVerdict shipgate.Decision) OptimizeCell {
	next, climbable := c.PromotionTarget()
	if climbable && runVerdict == shipgate.KEEP {
		c.Current = next
	}
	return c
}

// Regime is the cell's dev-regime derived from its Current rung — the C7 lowering
// (regime.go, #1250). An OptimizeCell is well-placed only while its rung sits in
// R2Optimize, the regime whose playbook IS rsiloop+shipgate; once a kept run climbs it
// into R3Production the optimize loop has done its job and the self-tax gate takes over.
func (c OptimizeCell) Regime() Regime { return RegimeFor(c.Current) }

// InOptimizeRegime reports whether the cell's Current rung is still in the R2 optimize
// regime — the guard the rsiloop consumer checks before starting a run, so it only ever
// optimizes a cell the router (NextActionFor) actually routes to LoopRSIShipgate.
func (c OptimizeCell) InOptimizeRegime() bool { return c.Regime() == R2Optimize }

// NextAction is the routed next-action the C9 router (router.go, #1252) emits for the
// cell's Current rung. For a cell in R2Optimize it is LoopRSIShipgate — the long-running
// rsiloop run PromoteOnRun then folds back into the rung. This is the seam that makes the
// C10 consumer the named destination of the C9 routing: the router says "send this cell
// to rsiloop", and OptimizeCell is what receives that work and climbs the ladder with it.
func (c OptimizeCell) NextAction() NextAction { return NextActionFor(c.Current) }
