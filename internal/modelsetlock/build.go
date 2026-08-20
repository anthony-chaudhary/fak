package modelsetlock

import (
	"crypto/sha256"
	"encoding/hex"
	"reflect"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

type preparedInputs struct {
	intent          harnessmodelset.Intent
	inventory       modelinventory.Inventory
	digests         InputDigests
	target          Target
	resolverVersion string
}

// New verifies the supplied resolution by rerunning the pure resolver, then
// snapshots its selected immutable identities into a sealed canonical lock.
func New(inputs Inputs, resolution modelsetresolve.Resolution) (Lock, error) {
	evaluatedAt, err := time.Parse(time.RFC3339, resolution.EvaluatedAt)
	if err != nil || resolution.Schema != modelsetresolve.Schema {
		return Lock{}, lockError(failure(
			CodeResolutionMismatch, "resolution", "resolution schema or evaluated_at is invalid",
			"use the unchanged result returned by modelsetresolve.Resolve",
		))
	}
	prepared, err := prepare(inputs)
	if err != nil {
		return Lock{}, err
	}
	expected, resolveErr := modelsetresolve.Resolve(prepared.intent, prepared.inventory, evaluatedAt)
	if resolveErr != nil {
		return Lock{}, lockError(failure(
			CodeInputInvalid, "resolution", resolveErr.Error(),
			"resolve compatible required roles before creating a lock",
		))
	}
	if !reflect.DeepEqual(expected, resolution) {
		return Lock{}, lockError(failure(
			CodeResolutionMismatch, "resolution", "resolution does not match the canonical resolver result for the bound inputs",
			"pass the unchanged modelsetresolve.Resolve result for these exact inputs",
		))
	}

	rolesByID := make(map[string]harnessmodelset.Role, len(prepared.intent.Roles))
	for _, role := range prepared.intent.Roles {
		rolesByID[role.ID] = role
	}
	inventoryByID := make(map[string]modelinventory.Candidate, len(prepared.inventory.Candidates))
	for _, candidate := range prepared.inventory.Candidates {
		inventoryByID[candidate.ID] = candidate
	}

	lock := Lock{
		Schema:          Schema,
		Inputs:          prepared.digests,
		Target:          prepared.target,
		ResolverVersion: prepared.resolverVersion,
		EvaluatedAt:     evaluatedAt.UTC().Format(time.RFC3339),
		Roles:           make([]Role, 0, len(resolution.Roles)),
	}
	for _, outcome := range resolution.Roles {
		intentRole, ok := rolesByID[outcome.RoleID]
		if !ok {
			return Lock{}, lockError(failure(
				CodeResolutionMismatch, "roles."+outcome.RoleID, "resolution names a role absent from intent",
				"rerun modelsetresolve.Resolve with the bound intent",
			))
		}
		role := Role{
			ID:             outcome.RoleID,
			Required:       outcome.Required,
			Status:         outcome.Status,
			AlternativeIDs: alternativeIDs(intentRole.Alternatives),
			Rejections:     append([]modelsetresolve.Rejection(nil), outcome.Rejections...),
		}
		if outcome.Selection != nil {
			candidate, exists := inventoryByID[outcome.Selection.CandidateID]
			if !exists {
				return Lock{}, lockError(failure(
					CodeSelectedIdentityMissing, "roles."+outcome.RoleID+".selected.candidate_id",
					"selected candidate has no immutable identity in the bound inventory",
					"rerun resolution against an inventory containing the selected candidate",
				))
			}
			role.Selected = &Selected{
				AlternativeID: outcome.Selection.AlternativeID,
				CandidateID:   outcome.Selection.CandidateID,
				Identity:      candidate.Identity,
				Evidence:      identityEvidence(candidate),
			}
		}
		lock.Roles = append(lock.Roles, role)
	}
	lock = canonicalizeLock(lock)
	lock.ContentDigest = digestBytes(payloadBytes(lock))
	if err := validateLock(lock); err != nil {
		return Lock{}, err
	}
	return lock, nil
}

func prepare(inputs Inputs) (preparedInputs, error) {
	intentJSON, err := harnessmodelset.CanonicalJSON(inputs.Intent)
	if err != nil {
		return preparedInputs{}, lockError(failure(CodeInputInvalid, "intent", err.Error(), "repair and validate the model-set intent"))
	}
	intent, err := harnessmodelset.ParseJSON(intentJSON)
	if err != nil {
		return preparedInputs{}, lockError(failure(CodeInputInvalid, "intent", err.Error(), "repair and validate the model-set intent"))
	}
	inventoryJSON, diagnostics := inputs.Inventory.CanonicalJSON()
	if len(diagnostics) != 0 {
		return preparedInputs{}, lockError(failure(CodeInputInvalid, "inventory", diagnostics.Error(), "repair and normalize the model inventory"))
	}
	inventoryAsOf, parseErr := time.Parse(time.RFC3339, inputs.Inventory.AsOf)
	if parseErr != nil {
		return preparedInputs{}, lockError(failure(CodeInputInvalid, "inventory.as_of", "inventory as_of is not RFC3339", "normalize the inventory with an explicit UTC time"))
	}
	inventory, diagnostics := modelinventory.ParseJSON(inventoryJSON, inventoryAsOf)
	if len(diagnostics) != 0 {
		return preparedInputs{}, lockError(failure(CodeInputInvalid, "inventory", diagnostics.Error(), "repair and normalize the model inventory"))
	}
	if len(inputs.RuleBytes) == 0 {
		return preparedInputs{}, lockError(failure(CodeInputInvalid, "policy", "canonical policy bytes are empty", "supply the exact credential-free policy bytes used for resolution"))
	}
	target := canonicalTarget(inputs.Target)
	if failures := targetFailures(target); len(failures) != 0 {
		return preparedInputs{}, lockError(failures...)
	}
	resolverVersion := strings.TrimSpace(inputs.ResolverVersion)
	if resolverVersion == "" || resolverVersion != inputs.ResolverVersion {
		return preparedInputs{}, lockError(failure(CodeInputInvalid, "resolver_version", "resolver version is empty or has surrounding whitespace", "supply the exact stable resolver version"))
	}
	return preparedInputs{
		intent:          intent,
		inventory:       inventory,
		target:          target,
		resolverVersion: resolverVersion,
		digests: InputDigests{
			Intent:    digestBytes(intentJSON),
			Inventory: digestBytes(inventoryJSON),
			Policy:    digestBytes(inputs.RuleBytes),
			Target:    digestBytes(targetBytes(target)),
		},
	}, nil
}

func alternativeIDs(alternatives []harnessmodelset.Alternative) []string {
	ids := make([]string, len(alternatives))
	for i, alternative := range alternatives {
		ids[i] = alternative.ID
	}
	return ids
}

func identityEvidence(candidate modelinventory.Candidate) []modelinventory.Witness {
	witnesses := append([]modelinventory.Witness(nil), candidate.Identity.Witnesses...)
	witnesses = append(witnesses, candidate.Evidence.Availability.Witnesses...)
	groups := [][]modelinventory.Fact{
		candidate.Evidence.Serving,
		candidate.Evidence.Platform,
		candidate.Evidence.Policy,
		candidate.Evidence.Capabilities,
	}
	for _, facts := range groups {
		for _, fact := range facts {
			witnesses = append(witnesses, fact.Witnesses...)
		}
	}
	return normalizeEvidence(witnesses)
}

func digestBytes(raw []byte) string {
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:])
}
