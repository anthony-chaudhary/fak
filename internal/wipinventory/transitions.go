package wipinventory

import (
	"fmt"
	"strings"
	"time"
)

// TransitionKind is the closed WIP-unit lifecycle vocabulary.
type TransitionKind string

const (
	TransitionCreate    TransitionKind = "create"
	TransitionBind      TransitionKind = "bind"
	TransitionHandoff   TransitionKind = "handoff"
	TransitionRecover   TransitionKind = "recover"
	TransitionPark      TransitionKind = "park"
	TransitionLand      TransitionKind = "land"
	TransitionAbandon   TransitionKind = "abandon"
	TransitionSplit     TransitionKind = "split"
	TransitionMerge     TransitionKind = "merge"
	TransitionSupersede TransitionKind = "supersede"
)

// Transition records one asserted lifecycle change. IDs are repeated in both
// predecessor and successor positions for unary, identity-preserving changes.
type Transition struct {
	Kind         TransitionKind     `json:"kind"`
	Timestamp    time.Time          `json:"timestamp"`
	Source       string             `json:"source"`
	Provenance   Provenance         `json:"provenance"`
	Predecessors []WIPUnitID        `json:"predecessors,omitempty"`
	Successors   []WIPUnitID        `json:"successors,omitempty"`
	References   []SurfaceReference `json:"references,omitempty"`
	Witness      string             `json:"witness"`
}

type unitState uint8

const (
	stateActive unitState = iota + 1
	stateParked
	stateLanded
	stateAbandoned
	stateSuperseded
)

func (state unitState) terminal() bool {
	return state == stateLanded || state == stateAbandoned || state == stateSuperseded
}

// ValidateHistory validates an entire ordered history, including existence,
// lifecycle state, cardinality, and acyclicity constraints.
func ValidateHistory(history History) error {
	if history.Schema != WIPUnitSchema {
		return fmt.Errorf("incompatible WIP unit schema %q: want %q", history.Schema, WIPUnitSchema)
	}
	states := make(map[WIPUnitID]unitState)
	edges := make(map[WIPUnitID][]WIPUnitID)
	for index, transition := range history.Transitions {
		if err := validateTransition(index, transition, states, edges); err != nil {
			return err
		}
	}
	return nil
}

func validateTransition(index int, transition Transition, states map[WIPUnitID]unitState, edges map[WIPUnitID][]WIPUnitID) error {
	fail := func(format string, args ...any) error {
		return fmt.Errorf("transition %d (%s): %s", index, transition.Kind, fmt.Sprintf(format, args...))
	}
	if transition.Timestamp.IsZero() || transition.Timestamp.Location() != time.UTC {
		return fail("timestamp must be a non-zero UTC time")
	}
	if strings.TrimSpace(transition.Source) == "" || strings.TrimSpace(transition.Provenance.Actor) == "" || strings.TrimSpace(transition.Provenance.Mechanism) == "" || strings.TrimSpace(transition.Witness) == "" {
		return fail("source, provenance actor/mechanism, and witness are required")
	}
	for _, ref := range transition.References {
		if err := ref.Validate(); err != nil {
			return fail("%v", err)
		}
	}
	if duplicateID(transition.Predecessors) || duplicateID(transition.Successors) {
		return fail("predecessor and successor lists must not contain duplicates")
	}
	for _, id := range append(append([]WIPUnitID(nil), transition.Predecessors...), transition.Successors...) {
		if err := id.Validate(); err != nil {
			return fail("%v", err)
		}
	}

	requireKnown := func(ids []WIPUnitID) error {
		for _, id := range ids {
			if _, ok := states[id]; !ok {
				return fail("unknown WIP unit ID %q", id)
			}
		}
		return nil
	}
	requireNonterminal := func(ids []WIPUnitID) error {
		for _, id := range ids {
			if states[id].terminal() {
				return fail("WIP unit %q is terminal", id)
			}
		}
		return nil
	}

	switch transition.Kind {
	case TransitionCreate:
		if len(transition.Predecessors) != 0 || len(transition.Successors) != 1 {
			return fail("create requires zero predecessors and one successor")
		}
		id := transition.Successors[0]
		if _, exists := states[id]; exists {
			return fail("duplicate creation of %q", id)
		}
		states[id] = stateActive
		return nil
	case TransitionBind, TransitionHandoff:
		if !sameUnary(transition) {
			return fail("%s requires one identical predecessor and successor", transition.Kind)
		}
		if err := requireKnown(transition.Predecessors); err != nil {
			return err
		}
		if err := requireNonterminal(transition.Predecessors); err != nil {
			return err
		}
		if transition.Kind == TransitionBind && len(transition.References) == 0 {
			return fail("bind requires at least one surface reference")
		}
		return nil
	case TransitionRecover:
		if !sameUnary(transition) {
			return fail("recover requires one identical predecessor and successor")
		}
		if err := requireKnown(transition.Predecessors); err != nil {
			return err
		}
		if err := requireNonterminal(transition.Predecessors); err != nil {
			return err
		}
		states[transition.Predecessors[0]] = stateActive
		return nil
	case TransitionPark:
		if !sameUnary(transition) {
			return fail("park requires one identical predecessor and successor")
		}
		if err := requireKnown(transition.Predecessors); err != nil {
			return err
		}
		if err := requireNonterminal(transition.Predecessors); err != nil {
			return err
		}
		states[transition.Predecessors[0]] = stateParked
		return nil
	case TransitionLand, TransitionAbandon:
		if !sameUnary(transition) {
			return fail("%s requires one identical predecessor and successor", transition.Kind)
		}
		if err := requireKnown(transition.Predecessors); err != nil {
			return err
		}
		if err := requireNonterminal(transition.Predecessors); err != nil {
			return err
		}
		if transition.Kind == TransitionLand {
			states[transition.Predecessors[0]] = stateLanded
		} else {
			states[transition.Predecessors[0]] = stateAbandoned
		}
		return nil
	case TransitionSplit:
		if len(transition.Predecessors) != 1 || len(transition.Successors) < 2 {
			return fail("split requires one predecessor and at least two successors")
		}
	case TransitionMerge:
		if len(transition.Predecessors) < 2 || len(transition.Successors) != 1 {
			return fail("merge requires at least two predecessors and one successor")
		}
	case TransitionSupersede:
		if len(transition.Predecessors) != 1 || len(transition.Successors) != 1 {
			return fail("supersede requires one predecessor and one successor")
		}
		if transition.Predecessors[0] == transition.Successors[0] {
			return fail("self-supersession is forbidden")
		}
	default:
		return fail("unknown transition kind")
	}

	if err := requireKnown(transition.Predecessors); err != nil {
		return err
	}
	if err := requireKnown(transition.Successors); err != nil {
		return err
	}
	// Detect cycles before lifecycle-state errors so a back-edge is reported as
	// the structural identity error it is, even though its target was retired.
	for _, from := range transition.Predecessors {
		for _, to := range transition.Successors {
			if from == to {
				return fail("relationship cannot connect a WIP unit to itself")
			}
			if reachable(edges, to, from, make(map[WIPUnitID]bool)) {
				return fail("relationship would create a cycle from %q to %q", from, to)
			}
		}
	}
	if err := requireNonterminal(transition.Predecessors); err != nil {
		return err
	}
	if err := requireNonterminal(transition.Successors); err != nil {
		return err
	}
	for _, from := range transition.Predecessors {
		edges[from] = append(edges[from], transition.Successors...)
		states[from] = stateSuperseded
	}
	return nil
}

func sameUnary(transition Transition) bool {
	return len(transition.Predecessors) == 1 && len(transition.Successors) == 1 && transition.Predecessors[0] == transition.Successors[0]
}

func duplicateID(ids []WIPUnitID) bool {
	seen := make(map[WIPUnitID]struct{}, len(ids))
	for _, id := range ids {
		if _, ok := seen[id]; ok {
			return true
		}
		seen[id] = struct{}{}
	}
	return false
}

func reachable(edges map[WIPUnitID][]WIPUnitID, from, target WIPUnitID, seen map[WIPUnitID]bool) bool {
	if from == target {
		return true
	}
	if seen[from] {
		return false
	}
	seen[from] = true
	for _, next := range edges[from] {
		if reachable(edges, next, target, seen) {
			return true
		}
	}
	return false
}
