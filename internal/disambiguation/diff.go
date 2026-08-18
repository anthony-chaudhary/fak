package disambiguation

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
)

const DiffSchemaVersion = "fak-disambiguation-diff/1"

type ChangeKind string

const (
	ChangeAdditive        ChangeKind = "additive"
	ChangeAliasMove       ChangeKind = "alias-move"
	ChangeSemantic        ChangeKind = "semantic-change"
	ChangeContrast        ChangeKind = "contrast-change"
	ChangeOwnerMove       ChangeKind = "owner-move"
	ChangeStaleTransition ChangeKind = "stale-transition"
	ChangeRemoval         ChangeKind = "removal"
)

type CompatibilityImpact string

const (
	ImpactCompatible CompatibilityImpact = "compatible"
	ImpactReview     CompatibilityImpact = "review"
	ImpactBreaking   CompatibilityImpact = "breaking"
)

type IndexChange struct {
	Kind          ChangeKind          `json:"kind"`
	CanonicalTerm string              `json:"canonical_term"`
	Detail        string              `json:"detail"`
	QueryImpact   CompatibilityImpact `json:"query_impact"`
}
type DiffReport struct {
	Schema  string        `json:"schema"`
	Changes []IndexChange `json:"changes"`
}

func DiffIndexes(before, after []Entry) DiffReport {
	report := DiffReport{Schema: DiffSchemaVersion, Changes: []IndexChange{}}
	oldByTerm, newByTerm := entriesByTerm(before), entriesByTerm(after)
	for term, oldEntry := range oldByTerm {
		newEntry, ok := newByTerm[term]
		if !ok {
			report.Changes = append(report.Changes, IndexChange{Kind: ChangeRemoval, CanonicalTerm: term, Detail: "canonical identity removed", QueryImpact: ImpactBreaking})
			continue
		}
		oldAliases, newAliases := diffStringSet(oldEntry.Identity.Aliases), diffStringSet(newEntry.Identity.Aliases)
		if !reflect.DeepEqual(oldAliases, newAliases) {
			report.Changes = append(report.Changes, IndexChange{Kind: ChangeAliasMove, CanonicalTerm: term, Detail: "declared aliases changed", QueryImpact: removedAny(oldAliases, newAliases, ImpactBreaking, ImpactCompatible)})
		}
		if oldEntry.Definition != newEntry.Definition || oldEntry.Scope != newEntry.Scope {
			report.Changes = append(report.Changes, IndexChange{Kind: ChangeSemantic, CanonicalTerm: term, Detail: "definition or scope changed", QueryImpact: ImpactReview})
		}
		if !reflect.DeepEqual(oldEntry.Contrasts, newEntry.Contrasts) {
			report.Changes = append(report.Changes, IndexChange{Kind: ChangeContrast, CanonicalTerm: term, Detail: "contrast set changed", QueryImpact: ImpactReview})
		}
		if oldEntry.Owner != newEntry.Owner {
			report.Changes = append(report.Changes, IndexChange{Kind: ChangeOwnerMove, CanonicalTerm: term, Detail: "owner leaf or lane changed", QueryImpact: ImpactCompatible})
		}
		if oldEntry.Freshness.Verdict != newEntry.Freshness.Verdict {
			report.Changes = append(report.Changes, IndexChange{Kind: ChangeStaleTransition, CanonicalTerm: term, Detail: string(oldEntry.Freshness.Verdict) + " -> " + string(newEntry.Freshness.Verdict), QueryImpact: ImpactReview})
		}
	}
	for term := range newByTerm {
		if _, ok := oldByTerm[term]; !ok {
			report.Changes = append(report.Changes, IndexChange{Kind: ChangeAdditive, CanonicalTerm: term, Detail: "canonical identity added", QueryImpact: ImpactCompatible})
		}
	}
	sort.Slice(report.Changes, func(i, j int) bool {
		if report.Changes[i].CanonicalTerm != report.Changes[j].CanonicalTerm {
			return report.Changes[i].CanonicalTerm < report.Changes[j].CanonicalTerm
		}
		return report.Changes[i].Kind < report.Changes[j].Kind
	})
	return report
}
func entriesByTerm(entries []Entry) map[string]Entry {
	out := map[string]Entry{}
	for _, e := range entries {
		out[e.Identity.CanonicalTerm] = e
	}
	return out
}
func diffStringSet(values []string) map[string]bool {
	out := map[string]bool{}
	for _, v := range values {
		out[v] = true
	}
	return out
}
func removedAny(old, new map[string]bool, removed, unchanged CompatibilityImpact) CompatibilityImpact {
	for key := range old {
		if !new[key] {
			return removed
		}
	}
	return unchanged
}

func DecodeGeneratedIndex(data []byte) ([]Entry, error) {
	var document GeneratedIndex
	if err := json.Unmarshal(data, &document); err != nil {
		return nil, fmt.Errorf("decode generated index: %w", err)
	}
	if document.Schema != GeneratedIndexSchemaVersion {
		return nil, fmt.Errorf("generated index schema %q", document.Schema)
	}
	return document.Entries, nil
}
