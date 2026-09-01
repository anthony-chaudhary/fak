package wipinventory

import (
	"fmt"
	"sort"
	"strings"
)

// IssueSessionStatus is the inventory-facing vocabulary for an issue/session
// join. Debt remains visible instead of being guessed away.

// ExecutionBindingStatus is the inventory-facing status contract for a session binding.
type ExecutionBindingStatus string

const (
	ExecutionBindingJoined      ExecutionBindingStatus = "joined"
	ExecutionBindingMissing     ExecutionBindingStatus = "missing"
	ExecutionBindingAmbiguous   ExecutionBindingStatus = "ambiguous"
	ExecutionBindingConflicting ExecutionBindingStatus = "conflicting"
	ExecutionBindingStale       ExecutionBindingStatus = "stale"
)

type ExecutionIssueIdentity struct {
	Repository string `json:"repository"`
	Number     int    `json:"number"`
}

type ExecutionBinding struct {
	RootRegistrationID string                  `json:"root_registration_id"`
	Issue              *ExecutionIssueIdentity `json:"issue,omitempty"`
	RegistrationIDs    []string                `json:"registration_ids"`
	AttemptIDs         []string                `json:"attempt_ids"`
	SessionIDs         []string                `json:"session_ids,omitempty"`
	Status             ExecutionBindingStatus  `json:"status"`
	Details            []string                `json:"details,omitempty"`
}

type ExecutionBindingReport struct {
	Bindings []ExecutionBinding `json:"bindings"`
}

type IssueSessionStatus string

const (
	IssueSessionJoined      IssueSessionStatus = "joined"
	IssueSessionMissing     IssueSessionStatus = "missing"
	IssueSessionAmbiguous   IssueSessionStatus = "ambiguous"
	IssueSessionConflicting IssueSessionStatus = "conflicting"
	IssueSessionStale       IssueSessionStatus = "stale"
)

// IssueSessionUnit is one conserved logical WIP unit. Root, retry, resume, and
// child execution identities are aliases of UnitID, not additional WIP units.
type IssueSessionUnit struct {
	UnitID             WIPUnitID          `json:"unit_id,omitempty"`
	Issue              IssueReference     `json:"issue"`
	RootRegistrationID string             `json:"root_registration_id,omitempty"`
	RegistrationIDs    []string           `json:"registration_ids,omitempty"`
	AttemptIDs         []string           `json:"attempt_ids,omitempty"`
	SessionIDs         []string           `json:"session_ids,omitempty"`
	Status             IssueSessionStatus `json:"status"`
	Details            []string           `json:"details,omitempty"`
}

// IssueSessionDebt retains unresolved execution identity without inventing a
// WIP unit or assigning it to an issue.
type IssueSessionDebt struct {
	RootRegistrationID string             `json:"root_registration_id"`
	RegistrationIDs    []string           `json:"registration_ids,omitempty"`
	AttemptIDs         []string           `json:"attempt_ids,omitempty"`
	SessionIDs         []string           `json:"session_ids,omitempty"`
	Status             IssueSessionStatus `json:"status"`
	Details            []string           `json:"details,omitempty"`
}

type IssueSessionJoin struct {
	Units []IssueSessionUnit `json:"units"`
	Debt  []IssueSessionDebt `json:"debt,omitempty"`
}

// JoinIssueSessions binds authoritative WIP histories to the read-only session
// adapter. Issue identity is exact repository+number identity; mutable titles
// and task strings are never compared. Histories are not modified.
func JoinIssueSessions(histories []History, bindings ExecutionBindingReport) (IssueSessionJoin, error) {
	unitByIssue := make(map[string]IssueSessionUnit)
	for _, history := range histories {
		if err := ValidateHistory(history); err != nil {
			return IssueSessionJoin{}, err
		}
		units, err := issueUnitsFromHistory(history)
		if err != nil {
			return IssueSessionJoin{}, err
		}
		for key, unit := range units {
			if prior, exists := unitByIssue[key]; exists && prior.UnitID != unit.UnitID {
				prior.Status = IssueSessionAmbiguous
				prior.Details = sortedUnique(append(prior.Details, "issue is bound to multiple active WIP units: "+string(prior.UnitID)+","+string(unit.UnitID)))
				unitByIssue[key] = prior
				continue
			}
			unitByIssue[key] = unit
		}
	}

	result := IssueSessionJoin{}
	for _, binding := range bindings.Bindings {
		status := issueSessionStatus(binding.Status)
		if binding.Issue == nil || status == IssueSessionMissing || status == IssueSessionConflicting {
			result.Debt = append(result.Debt, debtFromBinding(binding, status))
			continue
		}
		key := issueKey(IssueReference{Repository: binding.Issue.Repository, Number: binding.Issue.Number})
		unit, ok := unitByIssue[key]
		if !ok {
			result.Debt = append(result.Debt, debtFromBinding(binding, IssueSessionMissing, "issue has no WIP-unit binding"))
			continue
		}
		if unit.Status == IssueSessionAmbiguous || status == IssueSessionAmbiguous {
			unit.Status = IssueSessionAmbiguous
		} else if status == IssueSessionStale {
			unit.Status = IssueSessionStale
		} else {
			unit.Status = IssueSessionJoined
		}
		unit.RootRegistrationID = binding.RootRegistrationID
		unit.RegistrationIDs = append([]string(nil), binding.RegistrationIDs...)
		unit.AttemptIDs = append([]string(nil), binding.AttemptIDs...)
		unit.SessionIDs = append([]string(nil), binding.SessionIDs...)
		unit.Details = sortedUnique(append(unit.Details, binding.Details...))
		unitByIssue[key] = unit
	}

	keys := make([]string, 0, len(unitByIssue))
	for key := range unitByIssue {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		result.Units = append(result.Units, unitByIssue[key])
	}
	sort.Slice(result.Debt, func(i, j int) bool { return result.Debt[i].RootRegistrationID < result.Debt[j].RootRegistrationID })
	return result, nil
}

func issueUnitsFromHistory(history History) (map[string]IssueSessionUnit, error) {
	result := make(map[string]IssueSessionUnit)
	for _, transition := range history.Transitions {
		if transition.Kind != TransitionCreate && transition.Kind != TransitionBind {
			continue
		}
		ids := append(append([]WIPUnitID(nil), transition.Successors...), transition.Predecessors...)
		for _, ref := range transition.References {
			if ref.Kind != SurfaceIssue || ref.Issue == nil {
				continue
			}
			if len(ids) == 0 {
				return nil, fmt.Errorf("issue reference %s has no WIP unit", issueKey(*ref.Issue))
			}
			unit := IssueSessionUnit{UnitID: ids[0], Issue: *ref.Issue, Status: IssueSessionMissing}
			key := issueKey(*ref.Issue)
			if prior, exists := result[key]; exists && prior.UnitID != unit.UnitID {
				prior.Status = IssueSessionAmbiguous
				prior.Details = sortedUnique(append(prior.Details, "issue is bound to multiple WIP units"))
				result[key] = prior
			} else {
				result[key] = unit
			}
		}
	}
	return result, nil
}

func debtFromBinding(binding ExecutionBinding, status IssueSessionStatus, details ...string) IssueSessionDebt {
	return IssueSessionDebt{
		RootRegistrationID: binding.RootRegistrationID,
		RegistrationIDs:    append([]string(nil), binding.RegistrationIDs...),
		AttemptIDs:         append([]string(nil), binding.AttemptIDs...),
		SessionIDs:         append([]string(nil), binding.SessionIDs...),
		Status:             status,
		Details:            sortedUnique(append(append([]string(nil), binding.Details...), details...)),
	}
}

func issueSessionStatus(status ExecutionBindingStatus) IssueSessionStatus {
	switch status {
	case ExecutionBindingJoined:
		return IssueSessionJoined
	case ExecutionBindingAmbiguous:
		return IssueSessionAmbiguous
	case ExecutionBindingConflicting:
		return IssueSessionConflicting
	case ExecutionBindingStale:
		return IssueSessionStale
	default:
		return IssueSessionMissing
	}
}

func issueKey(issue IssueReference) string {
	return fmt.Sprintf("%s#%d", strings.TrimSpace(issue.Repository), issue.Number)
}

func sortedUnique(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}
