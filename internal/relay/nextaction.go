// Rung E4 (issue #1883): the one-line-next-action extractor. Rung E1 (safepoint.go)
// composed a SafePoint from three caller-supplied verdicts and left deriving each to a
// later rung; E2 (inflightguard.go) derived NoInFlightTurn. This rung derives the THIRD
// axis — NextActionExpressible — the "one-line-next-action predicate" the spine names
// (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md, "Safe-stop-point detection": the
// leg can name its single next step in one line; if it cannot, it is mid-thought, not at a
// boundary).
//
// It EXTRACTS and VALIDATES the next action from LEG STATE, never from a model summary
// (issue #1883 Out of scope: "No model call; the next_action comes from leg state, not a
// summary"). A boundary is a leg that has collapsed to exactly one well-formed one-line
// next step; a leg that names none, or names more than one competing step, or whose step
// is a multi-line recap, is mid-thought — the extractor fails the axis rather than invent a
// next action. Like its E-track siblings it reads no clock and does no I/O.
package relay

import "strings"

// maxNextActionLen bounds a next_action to one line's worth of text. A candidate longer
// than this is a paragraph/recap, not a single atomic step, so it is not baton-expressible
// even on a single physical line. The bound reflects the schema doc's "one line" intent
// (docs/notes/RELAY-BATON-SCHEMA-2026-07-01.md, next_action); it invents no wire rule the
// codec (C2) enforces, it is only this predicate's expressibility budget.
const maxNextActionLen = 200

// ReasonNoNextAction is the closed cause the extractor stamps when leg state names no
// well-formed next step at all — nothing is baton-expressible, so the leg is mid-thought,
// not at a boundary. It mirrors the Reason* discipline of the E2 guard (inflightguard.go)
// so a supervisor reads a checkable cause, never free text.
const ReasonNoNextAction = "NO_NEXT_ACTION"

// ReasonAmbiguousNextAction is the closed cause stamped when leg state names MORE THAN ONE
// competing next step: the leg has not resolved which single step is next, so it is
// mid-thought, not at a boundary. This is the "ambiguous state" the Done condition fails.
const ReasonAmbiguousNextAction = "AMBIGUOUS_NEXT_ACTION"

// LegState is the boundary-relevant slice of a leg's OWN state the extractor reads: the
// candidate next steps the leg has named for itself from durable leg state — never a model
// summary (issue #1883 Out of scope). Each entry is one proposed atomic action; the
// extractor decides whether they collapse to a single baton-expressible next_action.
type LegState struct {
	// NextSteps are the leg's candidate next actions, in the order the leg named them. A
	// leg at a real boundary has resolved to exactly one; zero (nothing named) and two or
	// more distinct steps (competing, unresolved) are both mid-thought. Blank and
	// multi-line entries are not well-formed one-line next actions and do not count toward
	// the candidate set; the same step named twice counts once, not as two competitors.
	NextSteps []string
}

// NextActionVerdict is the typed result of the extractor: the derived SafePoint axis
// (Expressible IS SafePoint.NextActionExpressible), the extracted one-line NextAction on a
// positive verdict (empty otherwise), and the closed Reason naming the cause when the axis
// fails (empty on a positive verdict).
type NextActionVerdict struct {
	// Expressible is the derived NextActionExpressible axis: true only when leg state
	// collapses to exactly one well-formed one-line next step. Feed it straight into
	// SafePoint.NextActionExpressible.
	Expressible bool
	// NextAction is the extracted, normalized one-line next action — set only when
	// Expressible; it is exactly the string a Baton carries in next_action (baton.go).
	NextAction string
	// Reason names the closed cause of a failed axis (ReasonNoNextAction /
	// ReasonAmbiguousNextAction); empty when Expressible.
	Reason string
}

// ExtractNextAction derives the NextActionExpressible axis from leg state. It normalizes
// each candidate (trims surrounding whitespace) and keeps only the WELL-FORMED, distinct
// ones — a candidate is well-formed when, trimmed, it is non-empty, spans a single physical
// line (no embedded newline — a multi-line blob is a recap/summary, out of scope), and fits
// the one-line budget; a repeat of an already-kept step is not a second competitor. It then
// classifies:
//
//   - exactly one well-formed candidate -> Expressible, NextAction = that candidate (a
//     nameable boundary: the leg named its single next step in one line);
//   - none                              -> not Expressible, ReasonNoNextAction (the leg
//     named no baton-expressible next step — mid-thought);
//   - two or more distinct              -> not Expressible, ReasonAmbiguousNextAction (the
//     leg named competing next steps and has not resolved which is next — mid-thought).
//
// Pure: no clock, no I/O, and no model call — the next_action comes from leg state only.
func ExtractNextAction(ls LegState) NextActionVerdict {
	var wellFormed []string
	seen := make(map[string]bool)
	for _, cand := range ls.NextSteps {
		c := strings.TrimSpace(cand)
		if c == "" || strings.ContainsAny(c, "\r\n") || len(c) > maxNextActionLen {
			continue // blank, multi-line, or paragraph-length: not a one-line next step
		}
		if seen[c] {
			continue // the same next step named twice is not two competing steps
		}
		seen[c] = true
		wellFormed = append(wellFormed, c)
	}
	switch len(wellFormed) {
	case 1:
		return NextActionVerdict{Expressible: true, NextAction: wellFormed[0]}
	case 0:
		return NextActionVerdict{Reason: ReasonNoNextAction}
	default:
		return NextActionVerdict{Reason: ReasonAmbiguousNextAction}
	}
}
