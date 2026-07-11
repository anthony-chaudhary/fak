// Rung H5 (issue #1898): the hard-ceiling park path. A relay leg arms and fires only at
// a safe point (E1, safepoint.go); but a leg can reach its window hard ceiling BEFORE any
// safe point arrives — an in-flight turn that never settles, a tree that never goes green,
// a next action that never collapses to one line. The spine
// (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md, "Safe-stop-point detection") is
// blunt about the choice this forces: never blow the window to keep going. An overrun leg
// is unrecoverable — the window is gone and its uncommitted state with it; a PARKED leg is
// resumable. This rung is that fail-closed park: when the leg exhausts its window without a
// safe point, it emits a RELAY_PARKED_UNSAFE tombstone anchored at the last committed SHA,
// so a later resume picks up from the last good commit and no committed work is lost. It
// mints no successor — a park is a stop that hands the goal back to an operator, not a
// rotation.
//
// Verified, never claimed; lose no committed work. The resume anchor is the last committed
// SHA a boundary actually OBSERVED (BoundaryObs.AtSHA), never a number a leg asserted — a
// git ref a resume re-verifies (reload.go), exactly like the anchor a rotation pins. Because
// a relay never force-rotates mid-action (that is out of scope by construction), every
// committed byte sits at a commit boundary the park anchors on: parking at the last observed
// commit strands nothing committed. A leg that committed nothing parks with an empty anchor —
// a cold resume — and "lose no committed work" holds vacuously: there was none to lose, and
// the empty anchor fails closed to a re-derive from durable state rather than a false handoff.
//
// Pure fold, like its H/G-track siblings (idle.go, noprogress.go, safepoint.go): no clock,
// no I/O. The driver folds each boundary's observed commit SHA and hold reason in as it goes;
// on exhaustion it asks this rung to render the park record. Deciding WHEN the ceiling is hit
// (the MaxBoundaries loop bound) stays in the driver, exactly as arm/fire timing does — this
// rung only renders the park, it does not decide to invoke it.
package relay

import "fmt"

// ReasonParkedUnsafe is the closed relay reason token (OPERATOR_GATE category,
// docs/notes/RELAY-REASON-VOCABULARY-2026-07-01.md) emitted when a leg reaches the window
// hard ceiling before reaching a safe point: automatic rotation stops, the parked state is
// written for a later resume, and the goal is handed back to an operator. It is the
// fail-closed twin of RELAY_ROTATED — the same window pressure, but with no safe point to
// fire at, so the leg parks instead of rotating. It joins the Reason* discipline
// (ReasonGoalDone, ReasonIdleParked, ReasonNoProgress, …) so a supervisor reads a checkable,
// closed-vocabulary cause rather than free text. Already reserved as a Tombstone reason in
// baton.go; this rung is the producer that mints it.
const ReasonParkedUnsafe = "RELAY_PARKED_UNSAFE"

// CeilingPark folds a leg's per-boundary observations so that, if the leg exhausts its window
// without a safe point, the driver can render a resumable park. It tracks only two things: the
// last committed SHA any boundary observed — the high-water resume anchor the park pins — and
// the most recent hold reason, a display-only note on WHY no safe point was reached. The zero
// value is a fresh, empty tracker: no anchor observed yet, so an immediate park is a cold stop.
type CeilingPark struct {
	// lastSHA is the most recent non-empty committed SHA a boundary observed. It is the resume
	// anchor a park pins; empty means the leg committed nothing and the park is a cold resume.
	lastSHA string
	// lastHold is the most recent closed hold reason a boundary recorded (IN_FLIGHT_TOOL_CALL,
	// TREE_DIRTY_UNPARKED, NO_NEXT_ACTION, …) — a display-only note on why no safe point came.
	lastHold string
}

// Observe folds one boundary into the tracker: a non-empty committed SHA advances the running
// resume anchor, and a non-empty hold reason is remembered as the latest cause. An empty SHA
// (a boundary that observed no commit) leaves the anchor UNCHANGED — the last real commit stays
// the resume point, so a later dirty boundary never rewinds the anchor behind committed work.
func (p *CeilingPark) Observe(atSHA, hold string) {
	if atSHA != "" {
		p.lastSHA = atSHA
	}
	if hold != "" {
		p.lastHold = hold
	}
}

// Anchor returns the resumable SHA the park would pin — the last committed SHA observed, or
// empty if the leg committed nothing. It is the operator/debug read of "where a resume picks
// up", and the value the driver copies into the park baton's ProgressCursor.StartSHA.
func (p *CeilingPark) Anchor() string { return p.lastSHA }

// Park renders the terminal park record for a leg that exhausted its window without ever
// reaching a safe point. It returns a RELAY_PARKED_UNSAFE tombstone anchored at the last
// observed commit (Anchor), so the parked state is resumable and no committed work is lost.
// leg and boundaries are folded into a display-only note, with the last hold reason appended
// when one was recorded. The park mints no successor — the caller writes this tombstone and
// stops; it does not rotate.
func (p *CeilingPark) Park(leg, boundaries int) Tombstone {
	note := fmt.Sprintf("leg %d hit the window ceiling after %d boundaries without a safe point", leg, boundaries)
	if p.lastHold != "" {
		note += "; last hold: " + p.lastHold
	}
	return Tombstone{
		Reason: ReasonParkedUnsafe,
		AtSHA:  p.lastSHA,
		Note:   note,
	}
}
