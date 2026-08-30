package harnessmodelset

import (
	"encoding/json"
	"regexp"
	"sort"
	"strings"
)

var stableID = regexp.MustCompile(`^[a-z][a-z0-9._-]*$`)

// Validate enforces the complete v1 contract without consulting an inventory,
// network, clock, or ambient policy.
func (intent Intent) Validate() error {
	var diagnostics []Diagnostic
	if intent.Schema != SchemaV1 {
		diagnostics = append(diagnostics, diagnostic(
			CodeIntentVersionUnknown, "$.schema", "intent version is not recognized", "set schema to "+SchemaV1,
		))
	}
	if len(intent.Roles) == 0 {
		diagnostics = append(diagnostics, diagnostic(
			CodeRolesEmpty, "$.roles", "at least one role is required", "declare each model-backed harness responsibility as a role",
		))
	}

	seenRoles := map[string]struct{}{}
	for roleIndex, role := range intent.Roles {
		rolePath := indexPath("$.roles", roleIndex)
		validateID(role.ID, rolePath+".id", "role", &diagnostics)
		if _, exists := seenRoles[role.ID]; exists && role.ID != "" {
			diagnostics = append(diagnostics, diagnostic(
				CodeRoleDuplicate, rolePath+".id", "role id is declared more than once", "merge the requirements or give each role a unique stable id",
			))
		}
		seenRoles[role.ID] = struct{}{}
		if len(role.Alternatives) == 0 {
			diagnostics = append(diagnostics, diagnostic(
				CodeAlternativesEmpty, rolePath+".alternatives", "role must declare at least one compatible alternative", "add an alternative with explicit hard constraints",
			))
		}
		if role.Preference != nil {
			validatePreference(*role.Preference, rolePath+".preference", &diagnostics)
		}
		validateEvidence(role.Evidence, rolePath+".evidence", &diagnostics)

		seenAlternatives := map[string]struct{}{}
		for alternativeIndex, alternative := range role.Alternatives {
			alternativePath := indexPath(rolePath+".alternatives", alternativeIndex)
			validateID(alternative.ID, alternativePath+".id", "alternative", &diagnostics)
			if _, exists := seenAlternatives[alternative.ID]; exists && alternative.ID != "" {
				diagnostics = append(diagnostics, diagnostic(
					CodeAlternativeDuplicate, alternativePath+".id", "alternative id is duplicated within the role", "give each role-local alternative a unique id",
				))
			}
			seenAlternatives[alternative.ID] = struct{}{}
			validateAlternative(alternative, alternativePath, &diagnostics)
		}
	}
	return validationError(diagnostics...)
}

func validateAlternative(alternative Alternative, path string, diagnostics *[]Diagnostic) {
	capabilities := alternative.Capabilities
	operational := alternative.Operational
	if !hasModelRequirement(capabilities) && !hasOperationalConstraint(operational) {
		*diagnostics = append(*diagnostics, diagnostic(
			CodeConstraintsEmpty, path, "alternative has no hard constraints", "declare at least one capability or operational constraint",
		))
	}
	if capabilities.Family != "" {
		validateToken(capabilities.Family, path+".capabilities.family", "model family", diagnostics)
	}
	if capabilities.Quantization != "" {
		validateToken(capabilities.Quantization, path+".capabilities.quantization", "quantization scheme", diagnostics)
	}
	validatePositiveRequirement(capabilities.ToolCalling, path+".capabilities.tool_calling", diagnostics)
	validatePositiveRequirement(capabilities.StructuredOutput, path+".capabilities.structured_output", diagnostics)
	if capabilities.ToolProtocol != "" && !oneOf(string(capabilities.ToolProtocol), string(ToolProtocolOpenAI), string(ToolProtocolAnthropic), string(ToolProtocolMCP)) {
		*diagnostics = append(*diagnostics, unknownValue(path+".capabilities.tool_protocol", string(capabilities.ToolProtocol), "tool protocol"))
	}
	if capabilities.MinimumInputTokens != nil && *capabilities.MinimumInputTokens <= 0 {
		*diagnostics = append(*diagnostics, positiveValue(path+".capabilities.minimum_input_tokens", "minimum input tokens"))
	}
	validateModalities(capabilities.Modalities, path+".capabilities.modalities", diagnostics)

	if operational.Runtime != "" {
		validateToken(operational.Runtime, path+".operational.runtime", "runtime inventory key", diagnostics)
	}
	if operational.ServingProtocol != "" && !oneOf(string(operational.ServingProtocol), string(ServingProtocolOpenAI), string(ServingProtocolAnthropic), string(ServingProtocolGRPC), string(ServingProtocolInProcess)) {
		*diagnostics = append(*diagnostics, unknownValue(path+".operational.serving_protocol", string(operational.ServingProtocol), "serving protocol"))
	}
	validateStringSet(operational.Platforms, path+".operational.platforms", "platform", diagnostics)
	validateStringSet(operational.Accelerators, path+".operational.accelerators", "accelerator", diagnostics)
	if operational.MaxMemoryBytes != nil && *operational.MaxMemoryBytes <= 0 {
		*diagnostics = append(*diagnostics, positiveValue(path+".operational.max_memory_bytes", "maximum memory bytes"))
	}
	if operational.Locality != "" && !oneOf(string(operational.Locality), string(LocalityLocalOnly), string(LocalityRemoteAllowed)) {
		*diagnostics = append(*diagnostics, unknownValue(path+".operational.locality", string(operational.Locality), "locality"))
	}
	if operational.Privacy != "" && !oneOf(string(operational.Privacy), string(PrivacyNoEgress), string(PrivacyPrivateEndpoint), string(PrivacyPublicEndpoint)) {
		*diagnostics = append(*diagnostics, unknownValue(path+".operational.privacy", string(operational.Privacy), "privacy mode"))
	}
	validateStringSet(operational.LicenseAllowlist, path+".operational.license_allowlist", "license", diagnostics)
	if operational.Locality == LocalityLocalOnly && operational.Privacy == PrivacyPublicEndpoint {
		*diagnostics = append(*diagnostics, diagnostic(
			CodeConstraintConflict, path+".operational", "local-only execution conflicts with a public endpoint requirement", "use no-egress/private-endpoint privacy or allow remote execution",
		))
	}
}

func validatePreference(preference SelectionPreference, path string, diagnostics *[]Diagnostic) {
	if !oneOf(string(preference.Mode), string(PreferenceDeclaredOrder), string(PreferenceLocalFirst), string(PreferenceLowestMemory)) {
		*diagnostics = append(*diagnostics, unknownValue(path+".mode", string(preference.Mode), "preference mode"))
	}
}

func validateEvidence(evidence EvidencePolicy, path string, diagnostics *[]Diagnostic) {
	if evidence.MaxAgeHours <= 0 {
		*diagnostics = append(*diagnostics, diagnostic(
			CodeFreshnessInvalid, path+".max_age_hours", "evidence freshness must be a positive number of hours", "set max_age_hours to a positive integer",
		))
	}
	seen := map[EvidenceKind]struct{}{}
	for index, kind := range evidence.RequiredKinds {
		itemPath := indexPath(path+".required_kinds", index)
		if !oneOf(string(kind), string(EvidenceModelBehaviorProbe), string(EvidenceRuntimeProbe), string(EvidenceOperatorAttestation)) {
			*diagnostics = append(*diagnostics, unknownValue(itemPath, string(kind), "evidence kind"))
		}
		if _, exists := seen[kind]; exists {
			*diagnostics = append(*diagnostics, duplicateValue(itemPath, "evidence kind"))
		}
		seen[kind] = struct{}{}
	}
}

func validateModalities(values []Modality, path string, diagnostics *[]Diagnostic) {
	seen := map[Modality]struct{}{}
	for index, value := range values {
		itemPath := indexPath(path, index)
		if !oneOf(string(value), string(ModalityText), string(ModalityImage), string(ModalityAudio)) {
			*diagnostics = append(*diagnostics, unknownValue(itemPath, string(value), "modality"))
		}
		if _, exists := seen[value]; exists {
			*diagnostics = append(*diagnostics, duplicateValue(itemPath, "modality"))
		}
		seen[value] = struct{}{}
	}
}

func validateStringSet(values []string, path, name string, diagnostics *[]Diagnostic) {
	seen := map[string]struct{}{}
	for index, value := range values {
		itemPath := indexPath(path, index)
		validateToken(value, itemPath, name, diagnostics)
		if _, exists := seen[value]; exists {
			*diagnostics = append(*diagnostics, duplicateValue(itemPath, name))
		}
		seen[value] = struct{}{}
	}
}

func validateID(value, path, name string, diagnostics *[]Diagnostic) {
	if !stableID.MatchString(value) {
		*diagnostics = append(*diagnostics, diagnostic(
			CodeValueInvalid, path, name+" id must match ^[a-z][a-z0-9._-]*$", "use a non-empty lowercase stable id",
		))
	}
}

func validateToken(value, path, name string, diagnostics *[]Diagnostic) {
	if strings.TrimSpace(value) == "" || strings.TrimSpace(value) != value {
		*diagnostics = append(*diagnostics, diagnostic(
			CodeValueInvalid, path, name+" must be non-empty without surrounding whitespace", "declare the exact normalized "+name,
		))
	}
}

func validatePositiveRequirement(value *bool, path string, diagnostics *[]Diagnostic) {
	if value != nil && !*value {
		*diagnostics = append(*diagnostics, diagnostic(
			CodeConstraintAmbiguous, path, "false is ambiguous in a positive hard-requirement field", "omit the field when the capability is not required, or set it to true",
		))
	}
}

func hasModelRequirement(capabilities ModelRequirements) bool {
	return capabilities.Family != "" || capabilities.Quantization != "" || capabilities.ToolCalling != nil ||
		capabilities.StructuredOutput != nil || capabilities.ToolProtocol != "" ||
		capabilities.MinimumInputTokens != nil || len(capabilities.Modalities) > 0
}

func hasOperationalConstraint(operational OperationalConstraints) bool {
	return operational.Runtime != "" || operational.ServingProtocol != "" || len(operational.Platforms) > 0 ||
		len(operational.Accelerators) > 0 || operational.MaxMemoryBytes != nil || operational.Locality != "" ||
		operational.Privacy != "" || len(operational.LicenseAllowlist) > 0
}

func unknownValue(path, value, name string) Diagnostic {
	return diagnostic(CodeValueInvalid, path, name+" value "+quoted(value)+" is not defined by "+SchemaV1, "use a value defined by the current schema")
}

func positiveValue(path, name string) Diagnostic {
	return diagnostic(CodeValueInvalid, path, name+" must be positive", "set the value to a positive integer or omit the constraint")
}

func duplicateValue(path, name string) Diagnostic {
	return diagnostic(CodeValueDuplicate, path, name+" is declared more than once", "keep each set member exactly once")
}

func oneOf(value string, allowed ...string) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}

func indexPath(path string, index int) string {
	return path + "[" + itoa(index) + "]"
}

func quoted(value string) string {
	raw, _ := json.Marshal(value)
	return string(raw)
}

func itoa(value int) string {
	if value == 0 {
		return "0"
	}
	var digits [20]byte
	position := len(digits)
	for value > 0 {
		position--
		digits[position] = byte('0' + value%10)
		value /= 10
	}
	return string(digits[position:])
}

// CanonicalJSON validates intent and emits deterministic, newline-terminated
// JSON. Roles and set-valued fields are sorted; alternative order is retained.
func CanonicalJSON(intent Intent) ([]byte, error) {
	if err := intent.Validate(); err != nil {
		return nil, err
	}
	canonical := canonicalIntent(intent)
	raw, err := json.MarshalIndent(canonical, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(raw, '\n'), nil
}

func canonicalIntent(intent Intent) Intent {
	out := intent
	out.Roles = append([]Role(nil), intent.Roles...)
	for roleIndex := range out.Roles {
		role := &out.Roles[roleIndex]
		role.Alternatives = append([]Alternative(nil), role.Alternatives...)
		role.Evidence.RequiredKinds = append([]EvidenceKind(nil), role.Evidence.RequiredKinds...)
		sort.Slice(role.Evidence.RequiredKinds, func(i, j int) bool { return role.Evidence.RequiredKinds[i] < role.Evidence.RequiredKinds[j] })
		for alternativeIndex := range role.Alternatives {
			alternative := &role.Alternatives[alternativeIndex]
			alternative.Capabilities.Modalities = append([]Modality(nil), alternative.Capabilities.Modalities...)
			sort.Slice(alternative.Capabilities.Modalities, func(i, j int) bool {
				return alternative.Capabilities.Modalities[i] < alternative.Capabilities.Modalities[j]
			})
			alternative.Operational.Platforms = sortedStrings(alternative.Operational.Platforms)
			alternative.Operational.Accelerators = sortedStrings(alternative.Operational.Accelerators)
			alternative.Operational.LicenseAllowlist = sortedStrings(alternative.Operational.LicenseAllowlist)
		}
	}
	sort.Slice(out.Roles, func(i, j int) bool { return out.Roles[i].ID < out.Roles[j].ID })
	return out
}

func sortedStrings(values []string) []string {
	out := append([]string(nil), values...)
	sort.Strings(out)
	return out
}
