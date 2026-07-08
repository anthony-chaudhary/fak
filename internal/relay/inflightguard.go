// Rung E2 (issue #1881): the in-flight-tool-call guard — never rotate mid-decode.
// Rung E1 (#1880) composed a SafePoint from three caller-supplied verdicts, but
// nothing yet ASSERTED the first before a rotation fired, and a mid-action rotation
// is a named failure (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md, Track E —
// "Safe-stop predicate"). This rung is that assertion: given the turn-boundary signal
// the session Decide gate already produces, it refuses a rotation while a tool call or
// decode is in flight and permits it only at the boundary. It reuses Draining
// semantics (Draining runs one more boundary then stops, never mid-action); it mints
// no new turn model and re-derives none of the E1 axes — it only derives the one axis
// (NoInFlightTurn) that E1 left to the caller, from the boundary the relay is about to
// rotate at. This rung reads no clock and does no I/O.
package relay

// ReasonInFlight is the closed refusal token the guard stamps when a rotation is
// attempted while a tool call/decode is still mid-turn — mirroring the Reason*
// discipline the session Decide gate uses (ReasonDrained, ReasonBudgetTurns, …) so a
// supervisor reads a checkable cause, never free text.
const ReasonInFlight = "IN_FLIGHT_TOOL_CALL"

// ReasonNotAtSafePoint is the closed refusal token the guard stamps when the turn is
// at its boundary (no tool call in flight) but another SafePoint axis — a dirty tree
// or a mid-thought next action — still fails. The in-flight axis is not the blocker
// here; the guard defers to the E1 conjunction rather than weaken it.
const ReasonNotAtSafePoint = "NOT_AT_SAFE_POINT"

// GuardVerdict is what the in-flight guard returns: Permit gates the rotation and
// Reason names the closed cause when it is refused (empty on a permit).
type GuardVerdict struct {
	Permit bool
	Reason string
}

// GuardRotation is the relay's pre-rotation assertion of the NoInFlightTurn axis.
// turnInFlight reuses the turn-boundary verdict the session Decide gate produces: it
// is true whenever the loop sits between a Decide that proceeded and that turn's
// completion — the exact window a tool call can be mid-decode — so a Draining session
// is still in flight until its last turn crosses the boundary. While in flight the
// rotation is refused with ReasonInFlight (never rotate mid-decode); at the boundary
// the guard asserts the derived axis onto the E1 SafePoint and defers to the full
// conjunction, refusing ReasonNotAtSafePoint if another axis fails and permitting the
// rotation only when every axis holds. Deriving NoInFlightTurn here — instead of
// trusting the caller's field — is the whole of rung E2.
func GuardRotation(sp SafePoint, turnInFlight bool) GuardVerdict {
	if turnInFlight {
		return GuardVerdict{Permit: false, Reason: ReasonInFlight}
	}
	sp.NoInFlightTurn = true // asserted from the boundary, not trusted from the caller
	if !sp.IsSafe() {
		return GuardVerdict{Permit: false, Reason: ReasonNotAtSafePoint}
	}
	return GuardVerdict{Permit: true}
}
