package modelsetresolve

import (
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
)

// Resolve validates both source contracts, then selects one compatible
// candidate per role without consulting ambient time, policy, storage, or the
// network. Alternative order is semantic; candidate order is not.
func Resolve(intent harnessmodelset.Intent, inventory modelinventory.Inventory, asOf time.Time) (Resolution, error) {
	intentErr := intent.Validate()
	inventoryDiagnostics := inventory.ValidateAt(asOf)
	if intentErr != nil || len(inventoryDiagnostics) != 0 {
		return Resolution{}, &InputError{
			IntentDiagnostics:    harnessmodelset.Diagnostics(intentErr),
			InventoryDiagnostics: append(modelinventory.Diagnostics(nil), inventoryDiagnostics...),
		}
	}

	asOf = asOf.UTC().Truncate(time.Second)
	roles := append([]harnessmodelset.Role(nil), intent.Roles...)
	sort.Slice(roles, func(i, j int) bool { return roles[i].ID < roles[j].ID })
	candidates := append([]modelinventory.Candidate(nil), inventory.Candidates...)
	sort.Slice(candidates, func(i, j int) bool { return inventoryEntryKey(candidates[i]) < inventoryEntryKey(candidates[j]) })

	resolution := Resolution{
		Schema:      Schema,
		EvaluatedAt: asOf.Format(time.RFC3339),
		Roles:       make([]RoleResolution, 0, len(roles)),
	}
	var requiredFailures []string
	for _, role := range roles {
		outcome := resolveRole(role, candidates, asOf)
		resolution.Roles = append(resolution.Roles, outcome)
		if outcome.Status == StatusRequiredUnresolved {
			requiredFailures = append(requiredFailures, role.ID)
		}
	}
	if len(requiredFailures) != 0 {
		return resolution, &RequiredRolesError{RoleIDs: requiredFailures}
	}
	return resolution, nil
}

func resolveRole(role harnessmodelset.Role, candidates []modelinventory.Candidate, asOf time.Time) RoleResolution {
	type compatible struct {
		alternativeIndex int
		candidate        modelinventory.Candidate
	}
	var matches []compatible
	var rejections []Rejection
	for alternativeIndex, alternative := range role.Alternatives {
		for _, candidate := range candidates {
			reasons := evaluateInventoryEntry(role, alternative, alternativeIndex, candidate, asOf)
			if len(reasons) == 0 {
				matches = append(matches, compatible{alternativeIndex: alternativeIndex, candidate: candidate})
				continue
			}
			rejections = append(rejections, reasons...)
		}
	}
	sortRejections(rejections)

	outcome := RoleResolution{
		RoleID:     role.ID,
		Required:   role.Required,
		Rejections: rejections,
	}
	if len(matches) == 0 {
		if role.Required {
			outcome.Status = StatusRequiredUnresolved
		} else {
			outcome.Status = StatusOptionalUnresolved
		}
		return outcome
	}

	firstAlternative := matches[0].alternativeIndex
	eligible := make([]modelinventory.Candidate, 0, len(matches))
	for _, match := range matches {
		if match.alternativeIndex == firstAlternative {
			eligible = append(eligible, match.candidate)
		}
	}
	preference := harnessmodelset.PreferenceDeclaredOrder
	if role.Preference != nil {
		preference = role.Preference.Mode
	}
	sortCandidates(eligible, preference)
	winner := eligible[0]
	outcome.Status = StatusSelected
	outcome.Selection = &Selection{
		AlternativeID: role.Alternatives[firstAlternative].ID,
		CandidateID:   winner.ID,
	}
	return outcome
}

func sortCandidates(candidates []modelinventory.Candidate, preference harnessmodelset.PreferenceMode) {
	sort.SliceStable(candidates, func(i, j int) bool {
		a, b := candidates[i], candidates[j]
		switch preference {
		case harnessmodelset.PreferenceLocalFirst:
			aLocal := a.Identity.Source == modelinventory.SourceLocal
			bLocal := b.Identity.Source == modelinventory.SourceLocal
			if aLocal != bLocal {
				return aLocal
			}
		case harnessmodelset.PreferenceLowestMemory:
			aMemory, aKnown := integerFact(a.Evidence.Platform, "accelerator_memory_bytes")
			bMemory, bKnown := integerFact(b.Evidence.Platform, "accelerator_memory_bytes")
			if aKnown != bKnown {
				return aKnown
			}
			if aKnown && aMemory != bMemory {
				return aMemory < bMemory
			}
		}
		return inventoryEntryKey(a) < inventoryEntryKey(b)
	})
}

func inventoryEntryKey(candidate modelinventory.Candidate) string {
	identity := candidate.Identity
	return strings.Join([]string{
		candidate.ID,
		string(identity.Source),
		identity.Provider,
		identity.Artifact,
		identity.Revision,
		identity.Digest,
		identity.Format,
	}, "\x00")
}
