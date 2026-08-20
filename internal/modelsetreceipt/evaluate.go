package modelsetreceipt

import (
	"bytes"
	"encoding/json"
	"errors"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

// Bind validates and content-binds the exact requirements, resolution, and
// inventory that a model-set lock intends startup to enforce.
func Bind(requirements harnessmodelset.Intent, resolution modelsetresolve.Resolution, inventory modelinventory.Inventory) (Expectation, error) {
	requirementsRaw, err := harnessmodelset.CanonicalJSON(requirements)
	if err != nil {
		return Expectation{}, validationError("requirements are invalid: " + err.Error())
	}
	inventoryRaw, canonicalInventory, err := canonicalInventory(inventory)
	if err != nil {
		return Expectation{}, err
	}
	resolutionRaw, err := canonicalResolutionJSON(resolution)
	if err != nil {
		return Expectation{}, err
	}
	evaluatedAt, err := time.Parse(time.RFC3339, resolution.EvaluatedAt)
	if err != nil {
		return Expectation{}, validationError("resolution.evaluated_at must be RFC3339")
	}
	recomputed, resolveErr := modelsetresolve.Resolve(requirements, canonicalInventory, evaluatedAt)
	if resolveErr != nil {
		return Expectation{}, validationError("resolution cannot be bound: " + resolveErr.Error())
	}
	recomputedRaw, err := canonicalResolutionJSON(recomputed)
	if err != nil || !bytes.Equal(resolutionRaw, recomputedRaw) {
		return Expectation{}, validationError("resolution does not match requirements and inventory")
	}
	bindings, _, err := bindingsFor(resolution, canonicalInventory)
	if err != nil {
		return Expectation{}, err
	}
	expectation := Expectation{
		Schema: ExpectationSchema,
		Digests: Digests{
			Requirements: digest(requirementsRaw),
			Resolution:   digest(resolutionRaw),
			Inventory:    digest(inventoryRaw),
		},
		Roles: bindings,
	}
	return canonicalExpectation(expectation), expectation.Validate()
}

// Evaluate independently revalidates current inventory, re-runs resolution,
// and emits a receipt. Any unknown, stale, malformed, or mismatched required
// fact returns the receipt plus *IncompatibleError.
func Evaluate(expectation Expectation, requirements harnessmodelset.Intent, locked modelsetresolve.Resolution, inventory modelinventory.Inventory, asOf time.Time) (Receipt, error) {
	if err := expectation.Validate(); err != nil {
		return Receipt{}, err
	}
	if asOf.IsZero() {
		return incompatibleReceipt(expectation, time.Unix(0, 0).UTC(), expectation.Digests, expectedRoleReceipts(expectation), []Failure{{
			Code: CodeAsOfRequired, Field: "as_of", Actual: "zero",
			Remediation: "pass the explicit UTC startup evaluation time",
		}})
	}
	asOf = asOf.UTC().Truncate(time.Second)
	var failures []Failure

	requirementsRaw, requirementsErr := harnessmodelset.CanonicalJSON(requirements)
	if requirementsErr != nil {
		failures = append(failures, Failure{Code: CodeRequirementsInvalid, Field: "requirements", Actual: requirementsErr.Error(), Remediation: "repair and re-resolve the model-set requirements"})
		requirementsRaw = fallbackJSON(requirements)
	}
	resolutionRaw, resolutionErr := canonicalResolutionJSON(locked)
	if resolutionErr != nil {
		failures = append(failures, Failure{Code: CodeResolutionInvalid, Field: "resolution", Actual: resolutionErr.Error(), Remediation: "regenerate the model-set resolution from validated inputs"})
		resolutionRaw = fallbackJSON(locked)
	}
	inventoryRaw, canonicalInventory, inventoryErr := canonicalInventory(inventory)
	if inventoryErr != nil {
		failures = append(failures, Failure{Code: CodeInventoryInvalid, Field: "inventory", Actual: inventoryErr.Error(), Remediation: "regenerate the normalized inventory from current observations"})
		inventoryRaw = fallbackInventoryJSON(inventory)
	}
	observedDigests := Digests{Requirements: digest(requirementsRaw), Resolution: digest(resolutionRaw), Inventory: digest(inventoryRaw)}
	compareDigest("requirements", expectation.Digests.Requirements, observedDigests.Requirements, CodeRequirementsDigest, &failures)
	compareDigest("resolution", expectation.Digests.Resolution, observedDigests.Resolution, CodeResolutionDigest, &failures)
	compareDigest("inventory", expectation.Digests.Inventory, observedDigests.Inventory, CodeInventoryDigest, &failures)

	var observedBindings []RoleBinding
	capabilities := map[string][]FactBinding{}
	if resolutionErr == nil && inventoryErr == nil {
		var bindingErr error
		observedBindings, capabilities, bindingErr = bindingsFor(locked, canonicalInventory)
		if bindingErr != nil {
			failures = append(failures, Failure{Code: CodeInventoryEntryMissing, Field: "resolution.selection", Actual: bindingErr.Error(), Remediation: "restore the locked candidate inventory entry or re-resolve"})
		}
	}

	var reevaluated modelsetresolve.Resolution
	canResolve := requirementsErr == nil && inventoryErr == nil
	if canResolve {
		inventoryDiagnostics := canonicalInventory.ValidateAt(asOf)
		for _, diagnostic := range inventoryDiagnostics {
			failures = append(failures, failuresFromInventory(diagnostic, expectation.Roles)...)
		}
		if len(inventoryDiagnostics) == 0 {
			var resolveErr error
			reevaluated, resolveErr = modelsetresolve.Resolve(requirements, canonicalInventory, asOf)
			var requiredErr *modelsetresolve.RequiredRolesError
			if resolveErr != nil && !errors.As(resolveErr, &requiredErr) {
				failures = append(failures, Failure{Code: CodeResolutionInvalid, Field: "reevaluation", Actual: resolveErr.Error(), Remediation: "repair the current model-set inputs and retry startup"})
			}
		}
	}

	roles := make([]RoleReceipt, 0, len(expectation.Roles))
	for _, expected := range expectation.Roles {
		role := RoleReceipt{RoleID: expected.RoleID, Required: expected.Required, Status: RoleCompatible}
		expectedCopy := expected
		role.Expected = &expectedCopy
		observed, ok := findBinding(observedBindings, expected.RoleID)
		if !ok {
			failures = append(failures, Failure{Code: CodeRoleMissing, RoleID: expected.RoleID, Field: "resolution.roles", Expected: expected.RoleID, Actual: "missing", Remediation: "re-resolve every locked role before startup"})
		} else {
			observedCopy := observed
			role.Observed = &observedCopy
			role.FactBindings = append([]FactBinding(nil), capabilities[expected.RoleID]...)
			compareBinding(expected, observed, &failures)
		}

		if current, ok := findRoleResolution(reevaluated, expected.RoleID); ok {
			if current.Selection != nil {
				role.Reevaluated = &Selection{AlternativeID: current.Selection.AlternativeID, CandidateID: current.Selection.CandidateID}
			}
			if current.Status != modelsetresolve.StatusSelected || current.Selection == nil {
				failures = append(failures, Failure{Code: CodeRequiredRoleUnresolved, RoleID: expected.RoleID, Field: "reevaluated_selection", Expected: expected.CandidateID, Actual: string(current.Status), Remediation: "refresh evidence or re-resolve the lock before startup"})
				for _, rejection := range current.Rejections {
					failures = append(failures, failureFromRejection(rejection))
				}
			} else if current.Selection.AlternativeID != expected.AlternativeID || current.Selection.CandidateID != expected.CandidateID {
				failures = append(failures, Failure{Code: CodeSelectionMismatch, RoleID: expected.RoleID, CandidateID: current.Selection.CandidateID, Field: "reevaluated_selection", Expected: expected.AlternativeID + "/" + expected.CandidateID, Actual: current.Selection.AlternativeID + "/" + current.Selection.CandidateID, Remediation: "re-resolve and review the changed role-local selection"})
			}
		}
		roles = append(roles, role)
	}

	failures = canonicalFailures(failures)
	for i := range roles {
		if hasRoleFailure(failures, roles[i].RoleID) || hasGlobalFailure(failures) {
			roles[i].Status = RoleIncompatible
		}
	}
	return incompatibleReceipt(expectation, asOf, observedDigests, roles, failures)
}

func incompatibleReceipt(expectation Expectation, asOf time.Time, observed Digests, roles []RoleReceipt, failures []Failure) (Receipt, error) {
	status := StatusCompatible
	if len(failures) != 0 {
		status = StatusIncompatible
	}
	receipt := canonicalReceipt(Receipt{
		Schema: ReceiptSchema, EvaluatedAt: asOf.Format(time.RFC3339), Status: status,
		Expected: expectation.Digests, Observed: observed, Roles: roles, Failures: failures,
	})
	if err := receipt.Validate(); err != nil {
		return Receipt{}, err
	}
	if status == StatusIncompatible {
		return receipt, &IncompatibleError{Failures: append([]Failure(nil), receipt.Failures...)}
	}
	return receipt, nil
}

func canonicalInventory(inventory modelinventory.Inventory) ([]byte, modelinventory.Inventory, error) {
	raw, diagnostics := inventory.CanonicalJSON()
	if len(diagnostics) != 0 {
		return nil, modelinventory.Inventory{}, validationError("inventory is invalid: " + diagnostics.Error())
	}
	asOf, err := time.Parse(time.RFC3339, inventory.AsOf)
	if err != nil {
		return nil, modelinventory.Inventory{}, validationError("inventory.as_of must be RFC3339")
	}
	canonical, diagnostics := modelinventory.ParseJSON(raw, asOf)
	if len(diagnostics) != 0 {
		return nil, modelinventory.Inventory{}, validationError("inventory read-back failed: " + diagnostics.Error())
	}
	return raw, canonical, nil
}

func bindingsFor(resolution modelsetresolve.Resolution, inventory modelinventory.Inventory) ([]RoleBinding, map[string][]FactBinding, error) {
	inventoryByID := make(map[string]modelinventory.Candidate, len(inventory.Candidates))
	for _, candidate := range inventory.Candidates {
		inventoryByID[candidate.ID] = candidate
	}
	var bindings []RoleBinding
	capabilities := map[string][]FactBinding{}
	for _, role := range resolution.Roles {
		if role.Status != modelsetresolve.StatusSelected || role.Selection == nil {
			continue
		}
		candidate, ok := inventoryByID[role.Selection.CandidateID]
		if !ok {
			return nil, nil, validationError("resolution candidate " + role.Selection.CandidateID + " is absent from inventory")
		}
		identityRaw, _ := json.Marshal(candidate.Identity)
		evidenceRaw, _ := json.Marshal(candidate.Evidence)
		factSetRaw, _ := json.Marshal(candidate.Evidence.Capabilities)
		bindings = append(bindings, RoleBinding{
			RoleID: role.RoleID, Required: role.Required,
			AlternativeID: role.Selection.AlternativeID, CandidateID: role.Selection.CandidateID,
			IdentityDigest: digest(identityRaw), EvidenceDigest: digest(evidenceRaw), FactSetDigest: digest(factSetRaw),
		})
		capabilities[role.RoleID] = bindCapabilities(candidate.Evidence.Capabilities)
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].RoleID < bindings[j].RoleID })
	return bindings, capabilities, nil
}

func bindCapabilities(facts []modelinventory.Fact) []FactBinding {
	out := make([]FactBinding, 0, len(facts))
	for _, fact := range facts {
		valueRaw, _ := json.Marshal(fact.Value)
		binding := FactBinding{Name: fact.Name, ValueDigest: digest(valueRaw)}
		for _, witness := range fact.Witnesses {
			binding.Witnesses = append(binding.Witnesses, EvidenceRef{
				Kind: string(witness.Kind), Source: witness.Source, ObservedAt: witness.ObservedAt, ExpiresAt: witness.ExpiresAt,
			})
		}
		out = append(out, binding)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func compareDigest(field, expected, actual string, code Code, failures *[]Failure) {
	if expected == actual {
		return
	}
	*failures = append(*failures, Failure{Code: code, Field: field + "_digest", Expected: expected, Actual: actual, Remediation: "re-resolve and review the changed " + field + " before startup"})
}

func compareBinding(expected, observed RoleBinding, failures *[]Failure) {
	if expected.AlternativeID != observed.AlternativeID || expected.CandidateID != observed.CandidateID {
		*failures = append(*failures, Failure{Code: CodeSelectionMismatch, RoleID: expected.RoleID, CandidateID: observed.CandidateID, Field: "selection", Expected: expected.AlternativeID + "/" + expected.CandidateID, Actual: observed.AlternativeID + "/" + observed.CandidateID, Remediation: "re-resolve and review the changed role-local selection"})
	}
	checks := []struct {
		field            string
		expected, actual string
		code             Code
		remediation      string
	}{
		{"identity_digest", expected.IdentityDigest, observed.IdentityDigest, CodeIdentityMismatch, "restore the locked immutable identity or re-resolve"},
		{"evidence_digest", expected.EvidenceDigest, observed.EvidenceDigest, CodeEvidenceMismatch, "refresh evidence and re-resolve before startup"},
		{"fact_set_digest", expected.FactSetDigest, observed.FactSetDigest, CodeFactSetMismatch, "re-probe capabilities and re-resolve before startup"},
	}
	for _, check := range checks {
		if check.expected != check.actual {
			*failures = append(*failures, Failure{Code: check.code, RoleID: expected.RoleID, CandidateID: observed.CandidateID, Field: check.field, Expected: check.expected, Actual: check.actual, Remediation: check.remediation})
		}
	}
}

func failuresFromInventory(d modelinventory.Diagnostic, roles []RoleBinding) []Failure {
	code := CodeInventoryInvalid
	switch d.Code {
	case modelinventory.CodeEvidenceStale:
		code = CodeEvidenceStale
	case modelinventory.CodeEvidenceUnknown, modelinventory.CodeMissingFact:
		code = CodeRequiredFactUnknown
	case modelinventory.CodeCredentialMaterial:
		code = CodeCredentialMaterial
	}
	base := Failure{Code: code, SourceCode: string(d.Code), CandidateID: d.Candidate, Field: d.Field, Actual: d.Detail, EvidenceSource: d.EvidenceSource, Remediation: d.Remediation}
	var failures []Failure
	for _, role := range roles {
		if role.CandidateID == d.Candidate {
			failure := base
			failure.RoleID = role.RoleID
			failures = append(failures, failure)
		}
	}
	if len(failures) == 0 {
		failures = append(failures, base)
	}
	return failures
}

func failureFromRejection(r modelsetresolve.Rejection) Failure {
	code := CodeRequiredFactMismatch
	switch r.Code {
	case modelsetresolve.CodeRuntime, modelsetresolve.CodeServingProtocol:
		code = CodeRuntimeMismatch
	case modelsetresolve.CodeFactUnknown, modelsetresolve.CodeFactType, modelsetresolve.CodeEvidenceKindMissing:
		code = CodeRequiredFactUnknown
	case modelsetresolve.CodeEvidenceStale:
		code = CodeEvidenceStale
	}
	return Failure{Code: code, SourceCode: string(r.Code), RoleID: r.RoleID, CandidateID: r.CandidateID, Field: r.Constraint, Expected: r.Expected, Actual: r.Actual, EvidenceSource: r.EvidenceSource, Remediation: r.Remediation}
}

func canonicalFailures(in []Failure) []Failure {
	out := append([]Failure{}, in...)
	sortFailures(out)
	unique := out[:0]
	for _, failure := range out {
		if len(unique) == 0 || unique[len(unique)-1] != failure {
			unique = append(unique, failure)
		}
	}
	return unique
}

func findBinding(bindings []RoleBinding, roleID string) (RoleBinding, bool) {
	for _, binding := range bindings {
		if binding.RoleID == roleID {
			return binding, true
		}
	}
	return RoleBinding{}, false
}

func expectedRoleReceipts(expectation Expectation) []RoleReceipt {
	roles := make([]RoleReceipt, 0, len(expectation.Roles))
	for _, binding := range expectation.Roles {
		expected := binding
		roles = append(roles, RoleReceipt{
			RoleID: binding.RoleID, Required: binding.Required, Status: RoleIncompatible, Expected: &expected,
			FactBindings: []FactBinding{},
		})
	}
	return roles
}

func findRoleResolution(resolution modelsetresolve.Resolution, roleID string) (modelsetresolve.RoleResolution, bool) {
	for _, role := range resolution.Roles {
		if role.RoleID == roleID {
			return role, true
		}
	}
	return modelsetresolve.RoleResolution{}, false
}

func hasRoleFailure(failures []Failure, roleID string) bool {
	for _, failure := range failures {
		if failure.RoleID == roleID {
			return true
		}
	}
	return false
}

func hasGlobalFailure(failures []Failure) bool {
	for _, failure := range failures {
		if failure.RoleID == "" {
			return true
		}
	}
	return false
}

func fallbackJSON(value any) []byte {
	raw, err := json.Marshal(value)
	if err != nil {
		return []byte("null")
	}
	return raw
}

func fallbackInventoryJSON(value modelinventory.Inventory) []byte {
	type rawInventory modelinventory.Inventory
	return fallbackJSON(rawInventory(value))
}
