// Rung G6 (issue #1893): the no-progress escape — after K consecutive legs make no
// verified progress, the relay stops with RELAY_NO_PROGRESS and escalates to the
// operator instead of spinning fresh windows forever.
//
// The spine (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md, "Thrash";
// RELAY-REASON-VOCABULARY-2026-07-01.md) puts it sharply: an infinite relay is worse
// than a stop. G4 hysteresis (hysteresis.go) throttles thrash by withholding re-arm;
// this rung is the terminal backstop — if K legs in a row advance the VERIFIED
// progress cursor by nothing, automation halts and hands the goal back to a human.
// RELAY_NO_PROGRESS is an OPERATOR_GATE reason: the escape does not auto-replan.
//
// Verified, never claimed; fail-closed. "Made progress" means the D3 ledger-verified
// step count (progress.go) rose above the high-water mark a prior leg reached. An
// unverifiable read (ProgressUnknown), or a verified read that did not advance, counts
// as NO progress — a relay that cannot prove it moved forward is exactly the spin this
// escape exists to stop. The counter resets ONLY on verified forward movement, so a
// stream of unverifiable legs still trips it.
//
// Pure counter, like its G2/G4 siblings: no clock, no I/O, only the consecutive-empty
// run and the verified high-water mark, folded one completed leg at a time.
package relay

// ReasonNoProgress is the closed relay reason token (OPERATOR_GATE,
// docs/notes/RELAY-REASON-VOCABULARY-2026-07-01.md) emitted when K consecutive legs
// make no verified progress: stop automatic rotation, inspect the blocker and the
// hysteresis settings, and relaunch only behind a new progress witness or a narrowed
// objective. It joins the Reason* discipline (ReasonGoalDone, ReasonInFlight, …) so a
// supervisor reads a checkable cause, never free text.
const ReasonNoProgress = "RELAY_NO_PROGRESS"

// NoProgressEscape counts consecutive relay legs that made no verified progress and
// trips once the run reaches MaxEmptyLegs. The zero value is an unconfigured,
// non-tripping escape (MaxEmptyLegs 0 never halts — an unset policy never refuses);
// set MaxEmptyLegs to K, the number of empty legs the relay tolerates before halting.
type NoProgressEscape struct {
	// MaxEmptyLegs is K: the number of consecutive no-verified-progress legs the relay
	// tolerates before the escape trips. <= 0 disables the escape.
	MaxEmptyLegs int

	// empty is the current run of consecutive legs that showed no verified forward
	// movement.
	empty int
	// highWater is the greatest verified progress-step count any observed leg reached;
	// forward movement must exceed it to count as progress, so a ledger that stalls or
	// shrinks never resets the counter. The zero value (0) is correct at the start: a
	// first leg needs at least one verified step to count as progress.
	highWater int
}

// ObserveLeg folds one completed leg's verified progress into the counter and reports
// whether the relay must now halt. A leg counts as progress only when its reading is
// ProgressVerified AND its step count exceeds the running high-water mark; any other
// reading — verified-but-not-advanced, or ProgressUnknown — counts as an empty leg
// (fail closed). On forward movement the empty run resets to zero and the high-water
// mark rises. When the empty run reaches MaxEmptyLegs the escape trips: halt is true
// and reason is RELAY_NO_PROGRESS. With MaxEmptyLegs <= 0 the escape is disabled and
// halt is always false.
func (e *NoProgressEscape) ObserveLeg(now VerifiedProgress) (halt bool, reason string) {
	if now.Verdict == ProgressVerified && len(now.Steps) > e.highWater {
		e.empty = 0
		e.highWater = len(now.Steps)
		return false, ""
	}
	if e.MaxEmptyLegs <= 0 {
		return false, ""
	}
	e.empty++
	if e.empty >= e.MaxEmptyLegs {
		return true, ReasonNoProgress
	}
	return false, ""
}

// EmptyRun returns the current count of consecutive no-verified-progress legs — the
// operator/debug read of how close the relay is to the escape.
func (e *NoProgressEscape) EmptyRun() int { return e.empty }
