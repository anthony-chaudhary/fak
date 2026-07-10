package dispatchtick

import (
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// WORK SHAPE — the placement axis the DIFFICULTY tier does not carry (#3521).
// ---------------------------------------------------------------------------
//
// A model can be capable and still starve on the wrong task. Witnessed on this
// fleet: a claude-fable-5 worker ran ~13 clean turns on a hard, CHURNING issue and
// committed nothing — its context ballooned past the compaction target, every
// overflow triggered a budget-restart that reseeded only a sliver of context, and it
// looped into restart-amnesia until it exhausted the restart limit. The same model
// does surgical diffs cleanly. That was not a dumb model; it was a placement failure
// wearing a model's face.
//
// The tier launch table (launchprofile.go) maps DIFFICULTY (T2/T1/T0 + tier/ultra)
// onto a {model, effort, ultracode} profile. Difficulty says how HARD the work is; it
// says nothing about the work's SHAPE — bounded/surgical (a known file, a small diff)
// vs churning/exploratory (wide reading, long context, many turns) — nor about whether
// the target model's restart/reliability profile can hold that shape. In fact the
// default table routes the hardest BucketUltra to ProfileFableUltracode, placing the
// cheap model on the MOST churning bucket by design: precisely the witnessed mismatch.
//
// This file adds the missing axis as a pure, closed signal plus the placement verdict a
// resolver consults BEFORE launch — a preventive gate, not the reactive
// CLAIM_NO_COMMIT downgrade chain (which never fires on a restart-amnesia starvation,
// since that is not one of the model-switchable walls).
//
// CONSERVATIVE DEGRADE is load-bearing. An untagged issue, or one carrying
// contradictory shape labels, yields ShapeUnknown — and an unknown shape NEVER gates a
// placement. The gate is opt-in by label, exactly as the ultra uplift is.

// WorkShape is the CLOSED work-shape vocabulary. It is an additive set — a new value is
// an added constant plus a knownWorkShape arm, never a free-text field.
type WorkShape string

const (
	// ShapeSurgical is bounded work: a known file, a small diff, few turns. Every
	// model's restart profile can hold it.
	ShapeSurgical WorkShape = "surgical"
	// ShapeChurning is exploratory work: wide reading, long resident context, many
	// turns — the shape whose compaction/restart pressure a fast/cheap model's profile
	// may be unable to survive.
	ShapeChurning WorkShape = "churning"
	// ShapeUnknown is the conservative default: untagged, or contradictory labels. It
	// NEVER gates a placement, so today's behavior is byte-identical for unlabeled work.
	ShapeUnknown WorkShape = "unknown"
)

// knownWorkShape reports whether s is one of the closed WorkShape set.
func knownWorkShape(s WorkShape) bool {
	switch s {
	case ShapeSurgical, ShapeChurning, ShapeUnknown:
		return true
	}
	return false
}

// Valid reports whether s is a member of the closed WorkShape vocabulary.
func (s WorkShape) Valid() bool { return knownWorkShape(s) }

// The shape labels an issue carries. Two synonyms per family so the operator vocabulary
// reads naturally either way; both map to the same closed WorkShape.
const (
	ShapeSurgicalLabel    = "shape/surgical"
	ShapeBoundedLabel     = "shape/bounded"
	ShapeChurningLabel    = "shape/churning"
	ShapeExploratoryLabel = "shape/exploratory"
)

var shapeLabelTable = map[string]WorkShape{
	ShapeSurgicalLabel:    ShapeSurgical,
	ShapeBoundedLabel:     ShapeSurgical,
	ShapeChurningLabel:    ShapeChurning,
	ShapeExploratoryLabel: ShapeChurning,
}

// WorkShapeForIssue derives the work shape from an issue's labels (case-insensitive,
// trimmed — the same lower-cased label grammar tiertag uses).
//
// It returns ShapeUnknown for an untagged issue AND for one carrying labels from BOTH
// families: a contradictory signal is not a reason to guess, and guessing wrong is the
// exact failure this gate exists to prevent.
func WorkShapeForIssue(labels []string) WorkShape {
	var surgical, churning bool
	for _, l := range labels {
		switch shapeLabelTable[strings.ToLower(strings.TrimSpace(l))] {
		case ShapeSurgical:
			surgical = true
		case ShapeChurning:
			churning = true
		}
	}
	switch {
	case surgical && churning:
		return ShapeUnknown // contradictory — refuse to guess
	case churning:
		return ShapeChurning
	case surgical:
		return ShapeSurgical
	default:
		return ShapeUnknown
	}
}

// PlacementShapeMismatch is the closed reason token a refused (work-shape ×
// model-reliability) pairing emits, so the block is legible in the tick payload and
// routes to a re-placement instead of a silent starve.
const PlacementShapeMismatch = "PLACEMENT_SHAPE_MISMATCH"

// ChurningSafeModel is the re-route target for a churning slot: the frontier model
// whose restart profile holds a long, context-heavy session.
const ChurningSafeModel = WorkerModelOpus

// churningCapableModels is the per-model RELIABILITY/RESTART profile — can this model
// hold a churning slot under the fleet's current restart/compaction budget? Sourced
// from the witnessed profile, not from difficulty: claude-fable-5 does surgical diffs
// cleanly and starves on churning work (restart-amnesia), while opus holds both.
var churningCapableModels = map[string]bool{
	WorkerModelOpus:  true,
	WorkerModelFable: false,
}

// ModelHoldsChurningSlot reports whether model's restart/reliability profile can hold a
// churning slot. An UNKNOWN model id is assumed capable — the gate fails OPEN and never
// blocks a placement it has no witnessed profile for.
func ModelHoldsChurningSlot(model string) bool {
	if v, ok := churningCapableModels[strings.TrimSpace(model)]; ok {
		return v
	}
	return true
}

// PlacementVerdict is the placement gate's decision: OK to launch as placed, or a
// typed refusal naming the SafeModel the placement should be re-routed onto.
type PlacementVerdict struct {
	OK        bool
	Reason    string
	SafeModel string
	Detail    string
}

// AssessPlacement is the pure preventive placement gate: given the work's shape and the
// model a resolver placed on it, decide whether that pairing is one the model can hold.
//
// It is inert (OK) for every shape but ShapeChurning, for an unpinned (blank) model,
// and for any model whose witnessed profile holds a churning slot — so an untagged
// issue, a surgical issue, or an unprofiled model all leave the placement untouched.
// Pure: same inputs, same verdict, no I/O.
func AssessPlacement(shape WorkShape, model string) PlacementVerdict {
	model = strings.TrimSpace(model)
	if shape != ShapeChurning || model == "" || ModelHoldsChurningSlot(model) {
		return PlacementVerdict{OK: true}
	}
	return PlacementVerdict{
		OK:        false,
		Reason:    PlacementShapeMismatch,
		SafeModel: ChurningSafeModel,
		Detail: fmt.Sprintf("%s cannot hold a churning slot under the fleet restart budget; re-routing to %s",
			model, ChurningSafeModel),
	}
}
