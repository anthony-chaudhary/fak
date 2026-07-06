// Rung G2 (issue #1889): the two-phase arm/fire rotation state machine — the
// control-flow core that keeps a relay from ever rotating mid-action.
//
// The spine (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md, "Rotation is
// two-phase (arm, then fire)") splits a rotation into two observable phases so the
// window is never hit during a tool call or a half-commit:
//
//   - ARM at a soft mark, well below the hard ceiling (SOTA rotates around 50–70% of
//     the window, not 95%). Arming stops nothing; it is the advisory RELAY_ARMED —
//     "rotate at the next safe point."
//   - FIRE at the next safe point after arming — never before. Firing is the clean
//     RELAY_ROTATED drain: the leg writes its baton and hands control to its successor.
//
// This rung is the state machine only. It consumes two caller-supplied signals it
// never re-derives: whether the soft mark has crossed (the trigger-axes math over the
// Envelope axes is G3, #1890) and a SafePoint verdict (rung E1, #1880). It reads no
// clock, does no I/O, and holds no hysteresis (that is G4, #1891). Its one
// load-bearing invariant — FIRE happens only when the SafePoint is safe — is
// structural, not merely documented: RotationFired is assigned in exactly one place,
// guarded by sp.IsSafe(), so mid-action rotation is unrepresentable.
package relay

// RotationState is the closed set of phases the two-phase rotation passes through
// within a single relay leg. A leg starts Disarmed, arms once its soft mark crosses,
// and fires once — and only once — it reaches a safe point. Fired is terminal for the
// leg (see ArmFire.Step): a rotated leg does not re-rotate.
type RotationState string

const (
	// RotationDisarmed is the running state before any soft mark has crossed. The leg
	// works normally; no rotation is pending. It is the zero value's meaning.
	RotationDisarmed RotationState = "disarmed"
	// RotationArmed is the advisory pending-rotation state: a soft mark has crossed
	// (RELAY_ARMED) and the leg will rotate at the next safe point. Arming stops nothing.
	RotationArmed RotationState = "armed"
	// RotationFired is the terminal state: the leg reached a safe point while armed and
	// rotated (RELAY_ROTATED). A fired leg stays fired.
	RotationFired RotationState = "fired"
)

// ArmFire is the two-phase rotation controller for one relay leg. The zero value is a
// ready, Disarmed controller — no construction needed. It is a pure state machine:
// Step folds the two external signals into the next state and never reaches for a
// clock, the filesystem, or git.
type ArmFire struct {
	state RotationState
}

// State returns the controller's current phase. The zero-value controller reports
// RotationDisarmed so a freshly declared ArmFire is usable without initialization.
func (a *ArmFire) State() RotationState {
	if a.state == "" {
		return RotationDisarmed
	}
	return a.state
}

// Armed reports whether a rotation is pending — the soft mark has crossed but the leg
// has not yet reached a safe point to fire at.
func (a *ArmFire) Armed() bool { return a.State() == RotationArmed }

// Fired reports whether the leg has rotated. Once true it stays true.
func (a *ArmFire) Fired() bool { return a.State() == RotationFired }

// Step advances the state machine by one boundary evaluation and returns the new
// state. softMarkCrossed is whether any rotation trigger has crossed its soft mark
// (computed by the G3 trigger axes, #1890); sp is the safe-point verdict for this
// boundary (rung E1, #1880).
//
// The transitions encode the two-phase rule:
//
//   - Disarmed, soft mark not crossed -> Disarmed (keep working).
//   - Disarmed, soft mark crossed     -> Armed (RELAY_ARMED); if the leg is already at
//     a safe point this same boundary, it fires immediately — the "next safe point" is
//     now.
//   - Armed, not safe                 -> Armed (hold; arming is sticky — a crossed soft
//     mark never un-crosses within a leg, so softMarkCrossed is ignored once armed).
//   - Armed, safe                     -> Fired (RELAY_ROTATED).
//   - Fired                           -> Fired (terminal; a rotated leg never re-rotates).
//
// FIRE is reachable only through the sp.IsSafe() guard below — that single guard is
// what makes a mid-action rotation impossible to represent.
func (a *ArmFire) Step(softMarkCrossed bool, sp SafePoint) RotationState {
	switch a.State() {
	case RotationFired:
		return RotationFired
	case RotationDisarmed:
		if !softMarkCrossed {
			return RotationDisarmed
		}
		a.state = RotationArmed
		fallthrough
	case RotationArmed:
		if sp.IsSafe() {
			a.state = RotationFired
		} else {
			a.state = RotationArmed
		}
	}
	return a.state
}

// Reason maps the current phase to its closed relay reason token
// (docs/notes/RELAY-REASON-VOCABULARY-2026-07-01.md): RotationArmed -> RELAY_ARMED
// (advisory), RotationFired -> RELAY_ROTATED (TRUE_DRAIN). RotationDisarmed has no
// token — no rotation condition has fired — so it returns "". This is the seam a
// later floor (which opts into the tokens deliberately) reads; this rung emits the
// token, it does not refuse or drain on it.
func (a *ArmFire) Reason() string {
	switch a.State() {
	case RotationArmed:
		return "RELAY_ARMED"
	case RotationFired:
		return "RELAY_ROTATED"
	default:
		return ""
	}
}
