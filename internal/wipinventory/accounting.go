package wipinventory

import (
	"fmt"
	"sort"
	"strings"
)

// AccountingDebtKind classifies a receipt that cannot be safely applied.
type AccountingDebtKind string

const (
	AccountingDebtIncompleteTransition      AccountingDebtKind = "incomplete_transition"
	AccountingDebtDuplicateSuccessor        AccountingDebtKind = "duplicate_successor"
	AccountingDebtCycle                     AccountingDebtKind = "cycle"
	AccountingDebtPartialMergeRetirement    AccountingDebtKind = "partial_merge_retirement"
	AccountingDebtAmbiguousMultiParentOwner AccountingDebtKind = "ambiguous_multi_parent_ownership"
)

// AccountingDebt preserves an unapplied transition as typed debt.
type AccountingDebt struct {
	Kind            AccountingDebtKind `json:"kind"`
	TransitionIndex int                `json:"transition_index"`
	TransitionKind  TransitionKind     `json:"transition_kind,omitempty"`
	Detail          string             `json:"detail"`
}

// AccountedUnitState is the retained logical lifecycle state of a WIP unit.
type AccountedUnitState string

const (
	AccountedUnitActive     AccountedUnitState = "active"
	AccountedUnitParked     AccountedUnitState = "parked"
	AccountedUnitLanded     AccountedUnitState = "landed"
	AccountedUnitAbandoned  AccountedUnitState = "abandoned"
	AccountedUnitSuperseded AccountedUnitState = "superseded"
)

// AccountedUnit retains history rather than deleting retired units.
type AccountedUnit struct {
	ID       WIPUnitID          `json:"id"`
	State    AccountedUnitState `json:"state"`
	Receipts []int              `json:"receipts"`
}

// AccountingReceipt attributes an accepted transition's active-count delta.
type AccountingReceipt struct {
	TransitionIndex int            `json:"transition_index"`
	Kind            TransitionKind `json:"kind"`
	Witness         string         `json:"witness"`
	Before          int            `json:"before"`
	After           int            `json:"after"`
	Delta           int            `json:"delta"`
}

// Accounting is a deterministic, read-only projection of a WIP history.
type Accounting struct {
	ActiveCount int                 `json:"active_count"`
	Units       []AccountedUnit     `json:"units"`
	Receipts    []AccountingReceipt `json:"receipts"`
	Debt        []AccountingDebt    `json:"debt"`
}

// AccountHistory counts only transitions it can apply completely. Invalid
// receipts remain debt; no ownership, predecessor, or successor is inferred.
func AccountHistory(history History) Accounting {
	result := Accounting{}
	states := make(map[WIPUnitID]unitState)
	edges := make(map[WIPUnitID][]WIPUnitID)
	unitReceipts := make(map[WIPUnitID][]int)
	owners := make(map[WIPUnitID][]WIPUnitID)
	if history.Schema != WIPUnitSchema {
		result.Debt = append(result.Debt, AccountingDebt{Kind: AccountingDebtIncompleteTransition, TransitionIndex: -1, Detail: fmt.Sprintf("incompatible WIP unit schema %q: want %q", history.Schema, WIPUnitSchema)})
		return result
	}
	for index, transition := range history.Transitions {
		before := activeStateCount(states)
		if kind, detail, invalid := accountingPrecheck(transition, states, edges, owners); invalid {
			result.Debt = append(result.Debt, AccountingDebt{Kind: kind, TransitionIndex: index, TransitionKind: transition.Kind, Detail: detail})
			continue
		}
		if err := validateTransition(index, transition, states, edges); err != nil {
			result.Debt = append(result.Debt, AccountingDebt{Kind: classifyAccountingError(transition, states, err), TransitionIndex: index, TransitionKind: transition.Kind, Detail: err.Error()})
			continue
		}
		after := activeStateCount(states)
		result.Receipts = append(result.Receipts, AccountingReceipt{TransitionIndex: index, Kind: transition.Kind, Witness: transition.Witness, Before: before, After: after, Delta: after - before})
		for _, unitID := range touchedIDs(transition) {
			unitReceipts[unitID] = append(unitReceipts[unitID], index)
		}
		if relationshipKind(transition.Kind) {
			for _, successor := range transition.Successors {
				owners[successor] = append([]WIPUnitID(nil), transition.Predecessors...)
			}
		}
	}
	ids := make([]WIPUnitID, 0, len(states))
	for unitID := range states {
		ids = append(ids, unitID)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for _, unitID := range ids {
		result.Units = append(result.Units, AccountedUnit{ID: unitID, State: publicUnitState(states[unitID]), Receipts: unitReceipts[unitID]})
	}
	result.ActiveCount = activeStateCount(states)
	return result
}

func accountingPrecheck(transition Transition, states map[WIPUnitID]unitState, edges map[WIPUnitID][]WIPUnitID, owners map[WIPUnitID][]WIPUnitID) (AccountingDebtKind, string, bool) {
	if duplicateID(transition.Successors) {
		return AccountingDebtDuplicateSuccessor, "successor list contains a duplicate WIP unit ID", true
	}
	if !relationshipKind(transition.Kind) {
		return "", "", false
	}
	for _, from := range transition.Predecessors {
		for _, to := range transition.Successors {
			if from == to || reachable(edges, to, from, make(map[WIPUnitID]bool)) {
				return AccountingDebtCycle, fmt.Sprintf("relationship would create a cycle from %q to %q", from, to), true
			}
		}
	}
	if transition.Kind == TransitionMerge {
		active, retired := 0, 0
		for _, predecessor := range transition.Predecessors {
			state, ok := states[predecessor]
			if !ok {
				continue
			}
			if state.terminal() {
				retired++
			} else {
				active++
			}
		}
		if active > 0 && retired > 0 {
			return AccountingDebtPartialMergeRetirement, "merge predecessors are only partially retired", true
		}
	}
	for _, successor := range transition.Successors {
		if previous, exists := owners[successor]; exists && !sameIDSet(previous, transition.Predecessors) {
			return AccountingDebtAmbiguousMultiParentOwner, fmt.Sprintf("successor %q already has different explicit predecessors", successor), true
		}
	}
	return "", "", false
}

func classifyAccountingError(transition Transition, states map[WIPUnitID]unitState, err error) AccountingDebtKind {
	message := err.Error()
	if strings.Contains(message, "cycle") || strings.Contains(message, "itself") {
		return AccountingDebtCycle
	}
	if transition.Kind == TransitionMerge {
		active, retired := false, false
		for _, predecessor := range transition.Predecessors {
			state, ok := states[predecessor]
			if !ok {
				continue
			}
			if state.terminal() {
				retired = true
			} else {
				active = true
			}
		}
		if active && retired {
			return AccountingDebtPartialMergeRetirement
		}
	}
	return AccountingDebtIncompleteTransition
}

func relationshipKind(kind TransitionKind) bool {
	return kind == TransitionSplit || kind == TransitionMerge || kind == TransitionSupersede
}
func activeStateCount(states map[WIPUnitID]unitState) int {
	count := 0
	for _, state := range states {
		if state == stateActive {
			count++
		}
	}
	return count
}
func touchedIDs(transition Transition) []WIPUnitID {
	seen := make(map[WIPUnitID]bool)
	ids := make([]WIPUnitID, 0, len(transition.Predecessors)+len(transition.Successors))
	for _, unitID := range append(append([]WIPUnitID(nil), transition.Predecessors...), transition.Successors...) {
		if !seen[unitID] {
			seen[unitID] = true
			ids = append(ids, unitID)
		}
	}
	return ids
}
func sameIDSet(left, right []WIPUnitID) bool {
	if len(left) != len(right) {
		return false
	}
	seen := make(map[WIPUnitID]int, len(left))
	for _, id := range left {
		seen[id]++
	}
	for _, id := range right {
		seen[id]--
	}
	for _, count := range seen {
		if count != 0 {
			return false
		}
	}
	return true
}
func publicUnitState(state unitState) AccountedUnitState {
	switch state {
	case stateActive:
		return AccountedUnitActive
	case stateParked:
		return AccountedUnitParked
	case stateLanded:
		return AccountedUnitLanded
	case stateAbandoned:
		return AccountedUnitAbandoned
	case stateSuperseded:
		return AccountedUnitSuperseded
	default:
		panic("unknown WIP unit state")
	}
}
