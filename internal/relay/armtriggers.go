// Rung G3 (issue #1890): the trigger-axes math that decides WHEN a relay leg arms.
// It is the missing input to the G2 arm/fire state machine (armfire.go, #1889): that
// rung consumes a single `softMarkCrossed bool` and explicitly never re-derives it —
// this rung is where that bool is computed, by projecting the reused Envelope budget
// axes (context tokens, turns, wall-clock, spend) onto their soft marks.
//
// The spine (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md, "Thresholds and
// triggers") is explicit about the shape:
//
//   - Reuse the Envelope axes — context tokens (PRIMARY), turns, wall-clock, spend —
//     rather than inventing a parallel trigger set. Each axis is a used/cap pair; the
//     soft mark is the fraction of the cap at which the leg arms (SOTA rotates around
//     50–70% of the window, so the mark sits well below the hard ceiling).
//   - The soft mark is policy DATA, not a magic number in code (DOS lesson #3): the
//     caller supplies the fraction; this rung holds no default and reads no config.
//   - An axis the user never bounded (cap <= 0) never arms — a soft mark on an
//     unstated axis is meaningless, so it is silently inert rather than a surprise
//     trigger on an axis nobody set (the same "0 = no opinion" convention Envelope
//     itself uses).
//
// Like every G-track rung this is a pure fold: it reads no clock, does no I/O, and
// does NOT import internal/session. The caller (a later floor) reads the current
// usage off the live session.Envelope/Budget and hands the plain numbers in, exactly
// as armfire.go takes a plain bool and safepoint.go takes plain verdicts. Keeping the
// axis values as caller-supplied scalars is what lets the relay package stay a
// dependency-free signal-fold that the session package can wire without a cycle.
package relay

// Axis is the closed set of Envelope budget axes a rotation can arm on. The order is
// the spine's priority order — Context is the PRIMARY axis (window pressure is the
// dominant reason a leg rotates), the rest are secondary caps that independently arm.
// A stable, ordered set is what makes ArmTriggers.Cross report a single deterministic
// arming axis when several cross on the same boundary.
type Axis string

const (
	// AxisContext is the resident-context-token axis — the primary rotation trigger.
	// A leg arms here when its resident prompt/context fills past the soft mark of the
	// window, which is the dominant real-world rotation cause.
	AxisContext Axis = "context"
	// AxisTurns is the model-round-trip axis: arm when the leg has spent past the soft
	// mark of its turn budget.
	AxisTurns Axis = "turns"
	// AxisWall is the wall-clock axis: arm when elapsed real time passes the soft mark
	// of the wall-clock budget. Its used/cap pair is any consistent time unit (nanos,
	// seconds) — the fold only ever takes their ratio, so the unit cancels.
	AxisWall Axis = "wall"
	// AxisSpend is the priced-spend axis: arm when the run's cost passes the soft mark
	// of its spend cap. Used/cap are any consistent money unit (cents, micro-cents).
	AxisSpend Axis = "spend"
)

// axisOrder is the deterministic evaluation order — the spine's priority order, with
// context primary. Cross walks it so that when several axes cross on one boundary the
// FIRST (highest-priority) crossed axis is the one reported.
var axisOrder = []Axis{AxisContext, AxisTurns, AxisWall, AxisSpend}

// AxisUsage is one axis's current pressure: how much of the cap is used, and the cap
// itself. Both are int64 so the same type carries token counts, turn counts,
// wall-clock nanoseconds, and micro-cent spend without a per-axis struct. A cap of 0
// (or negative) means the axis is unbounded / not stated — it never arms, whatever
// Used reads (see Cross). Used is clamped at the fold, so a caller passing a raw
// counter never has to pre-clamp.
type AxisUsage struct {
	Used int64
	Cap  int64
}

// crossed reports whether this axis has reached its soft mark: used/cap >= softMark.
// An unbounded axis (cap <= 0) never crosses — a soft mark on an unstated cap is
// meaningless. A non-positive Used never crosses (nothing has been consumed yet).
//
// The comparison is the DIVISION form (used/cap) >= softMark, deliberately NOT the
// algebraically-equal multiply form used >= softMark*cap. The multiply form rounds the
// boundary up for any soft mark whose product with the cap is not binary-exact (e.g.
// 0.55*100 == 55.00000000000000710543 in float64), so a run sitting exactly AT a 55%
// mark on a cap of 100 would fail to arm — the boundary would silently drift off the
// decimal fraction the operator stated. Dividing first keeps the comparison against the
// stated fraction itself and matches session.boundedFraction's ratio convention
// (internal/session/observe.go), the sibling this rung's soft mark is the multi-axis
// generalization of. softMark is assumed already validated into (0,1] by the caller
// (ArmTriggers.valid gates it).
func (u AxisUsage) crossed(softMark float64) bool {
	if u.Cap <= 0 || u.Used <= 0 {
		return false
	}
	return float64(u.Used)/float64(u.Cap) >= softMark
}

// ArmTriggers is the pure trigger-axes evaluator for one relay leg. It holds only the
// soft-mark policy fraction — the one piece of rotation policy the spine says is DATA,
// supplied by the caller from an Envelope field or the [relay] dos.toml table, never a
// constant baked here. The zero value (SoftMark 0) is inert: it arms on nothing, so a
// leg with no configured rotation policy simply never rotates on a budget axis, which
// is the safe default.
type ArmTriggers struct {
	// SoftMark is the fraction of each axis's cap at which the leg arms, in (0,1].
	// Outside that range (including the 0 zero value) the evaluator is inert — no axis
	// can cross an unset or nonsensical mark.
	SoftMark float64
}

// valid reports whether the soft mark is a usable fraction. A mark <= 0 (the zero
// value, or a caller clearing the policy) or > 1 (past the hard ceiling, which the
// arm phase must sit below) disables all arming — the fold fails closed to "never
// arm" rather than arming on a garbage threshold.
func (t ArmTriggers) valid() bool { return t.SoftMark > 0 && t.SoftMark <= 1 }

// Cross folds the four axes against the soft mark and returns the FIRST axis (in the
// spine's context-primary priority order) that has crossed, and whether any crossed.
// This is the single call a caller makes to turn live budget usage into the
// `softMarkCrossed bool` that ArmFire.Step consumes — pass the returned bool straight
// through, and use the Axis to name WHICH axis armed the rotation (for the RELAY_ARMED
// reason detail).
//
// Each axis is evaluated INDEPENDENTLY against the same mark: any single bounded axis
// crossing arms the leg, exactly the done-condition the issue names ("each axis
// independently arms rotation at its soft mark"). Unbounded axes (cap <= 0) are inert.
// An invalid/unset soft mark arms on nothing.
func (t ArmTriggers) Cross(context, turns, wall, spend AxisUsage) (Axis, bool) {
	if !t.valid() {
		return "", false
	}
	byAxis := map[Axis]AxisUsage{
		AxisContext: context,
		AxisTurns:   turns,
		AxisWall:    wall,
		AxisSpend:   spend,
	}
	for _, ax := range axisOrder {
		if byAxis[ax].crossed(t.SoftMark) {
			return ax, true
		}
	}
	return "", false
}

// Crossed is the boolean-only convenience wrapper: the exact `softMarkCrossed bool`
// ArmFire.Step wants, dropping the arming-axis detail a caller that only drives the
// state machine does not need.
func (t ArmTriggers) Crossed(context, turns, wall, spend AxisUsage) bool {
	_, crossed := t.Cross(context, turns, wall, spend)
	return crossed
}
