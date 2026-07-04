// Rung E1 (issue #1880): the safe-point predicate. A relay may only rotate at a safe
// point — never mid-action — and a safe point is the conjunction the spine names
// (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md, "Safe-stop-point detection"): no
// in-flight tool call, a green-or-parked working tree, and a next action expressible in
// one line. This rung only composes the three signals into one predicate; it reads no
// clock and does no I/O, and it does not decide WHEN to check (that is the G-track
// arm/fire state machine, #1889) or how each signal is derived (later rungs).
package relay

// SafePoint is the conjunction of the three sub-conditions a rotation must observe
// before firing. Each field is a caller-supplied verdict from the existing adjudication
// it names — Draining/Decide for the turn boundary, safecommit/guard for the tree state,
// and the one-line-next-action extractor for the last — so this type never re-derives
// them, only composes them.
type SafePoint struct {
	// NoInFlightTurn is true when the turn-boundary model guarantees no tool call is
	// mid-decode (Draining runs one more boundary then stops, never mid-action).
	NoInFlightTurn bool
	// TreeGreenOrParked is true when the working tree has no half-written commit — it is
	// either clean/committed (green) or explicitly parked at a committable boundary.
	TreeGreenOrParked bool
	// NextActionExpressible is true when the leg can name its single next step in one
	// line; false means the leg is mid-thought, not at a boundary.
	NextActionExpressible bool
}

// IsSafe reports whether every sub-condition holds — the point at which a rotation is
// permitted to fire. All three are required; any single false sub-condition means the
// leg is not at a safe point.
func (s SafePoint) IsSafe() bool {
	return s.NoInFlightTurn && s.TreeGreenOrParked && s.NextActionExpressible
}
