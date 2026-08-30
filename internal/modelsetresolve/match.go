package modelsetresolve

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
)

type evidenceDomain uint8

const (
	domainRuntime evidenceDomain = iota
	domainModelBehavior
	domainAttestation
)

type matcher struct {
	role             harnessmodelset.Role
	alternative      harnessmodelset.Alternative
	alternativeIndex int
	candidate        modelinventory.Candidate
	asOf             time.Time
	observedKinds    map[harnessmodelset.EvidenceKind]struct{}
	rejections       []Rejection
}

func evaluateInventoryEntry(role harnessmodelset.Role, alternative harnessmodelset.Alternative, alternativeIndex int, candidate modelinventory.Candidate, asOf time.Time) []Rejection {
	m := matcher{
		role:             role,
		alternative:      alternative,
		alternativeIndex: alternativeIndex,
		candidate:        candidate,
		asOf:             asOf,
		observedKinds:    map[harnessmodelset.EvidenceKind]struct{}{},
	}
	m.requireBool(candidate.Evidence.Availability, domainRuntime, "availability", true, CodeUnavailable,
		"make the candidate available and refresh its availability witness")

	capabilities := alternative.Capabilities
	if capabilities.Family != "" {
		m.requireNamedText(candidate.Evidence.Capabilities, "model.family", domainModelBehavior,
			"capabilities.family", capabilities.Family, CodeFamily, "use a candidate from the declared model family")
	}
	if capabilities.Quantization != "" {
		m.requireNamedText(candidate.Evidence.Capabilities, "weights.quantization", domainModelBehavior,
			"capabilities.quantization", capabilities.Quantization, CodeQuantization, "use a candidate with the declared quantization scheme")
	}
	if capabilities.ToolCalling != nil {
		m.requireNamedBool(candidate.Evidence.Capabilities, "tool_calling", domainModelBehavior,
			"capabilities.tool_calling", true, CodeToolCalling, "supply a candidate with witnessed tool calling")
	}
	if capabilities.StructuredOutput != nil {
		m.requireNamedBool(candidate.Evidence.Capabilities, "structured_json", domainModelBehavior,
			"capabilities.structured_output", true, CodeStructuredOutput, "supply a candidate with witnessed structured JSON output")
	}
	if capabilities.ToolProtocol != "" {
		m.requireNamedText(candidate.Evidence.Capabilities, "tool_protocol", domainModelBehavior,
			"capabilities.tool_protocol", string(capabilities.ToolProtocol), CodeToolProtocol, "use a candidate that implements the declared tool protocol")
	}
	if capabilities.MinimumInputTokens != nil {
		m.requireNamedMinimum(candidate.Evidence.Capabilities, "context_tokens", domainModelBehavior,
			"capabilities.minimum_input_tokens", *capabilities.MinimumInputTokens, CodeInputTokens, "use a candidate with a large enough witnessed input context")
	}
	for _, modality := range capabilities.Modalities {
		m.requireModality(modality)
	}

	operational := alternative.Operational
	if operational.Runtime != "" {
		m.requireNamedText(candidate.Evidence.Serving, "runtime", domainRuntime,
			"operational.runtime", operational.Runtime, CodeRuntime, "use a candidate witnessed on the declared runtime")
	}
	if operational.ServingProtocol != "" {
		m.requireNamedText(candidate.Evidence.Serving, "protocol", domainRuntime,
			"operational.serving_protocol", string(operational.ServingProtocol), CodeServingProtocol, "use a candidate serving the declared protocol")
	}
	if len(operational.Platforms) != 0 {
		m.requirePlatform(operational.Platforms)
	}
	if len(operational.Accelerators) != 0 {
		m.requireNamedAllowed(candidate.Evidence.Platform, "accelerator", domainRuntime,
			"operational.accelerators", operational.Accelerators, CodeAccelerator, "use a candidate witnessed on an allowed accelerator")
	}
	if operational.MaxMemoryBytes != nil {
		m.requireNamedMaximum(candidate.Evidence.Platform, "accelerator_memory_bytes", domainRuntime,
			"operational.max_memory_bytes", *operational.MaxMemoryBytes, CodeMemory, "use a candidate whose witnessed accelerator memory fits the ceiling")
	}
	if operational.Locality != "" {
		m.requireLocality(operational.Locality)
	}
	if operational.Privacy != "" {
		m.requireNamedText(candidate.Evidence.Policy, "privacy", domainAttestation,
			"operational.privacy", string(operational.Privacy), CodePrivacy, "use a candidate with the declared witnessed privacy posture")
	}
	if len(operational.LicenseAllowlist) != 0 {
		m.requireNamedAllowed(candidate.Evidence.Policy, "license", domainAttestation,
			"operational.license_allowlist", operational.LicenseAllowlist, CodeLicense, "approve the candidate license or choose an allowlisted candidate")
	}
	m.requireEvidenceKinds()
	sortRejections(m.rejections)
	return m.rejections
}

func (m *matcher) requireNamedBool(facts []modelinventory.Fact, name string, domain evidenceDomain, constraint string, expected bool, code RejectionCode, remediation string) {
	fact, ok := findFact(facts, name)
	if !ok {
		m.unknown(constraint, strconv.FormatBool(expected), remediation)
		return
	}
	m.requireBool(fact, domain, constraint, expected, code, remediation)
}

func (m *matcher) requireBool(fact modelinventory.Fact, domain evidenceDomain, constraint string, expected bool, code RejectionCode, remediation string) {
	m.observeEvidence(fact, domain, constraint)
	if fact.Value.Bool == nil {
		m.wrongType(constraint, "boolean", valueType(fact.Value), remediation)
		return
	}
	if *fact.Value.Bool != expected {
		m.reject(code, constraint, strconv.FormatBool(expected), strconv.FormatBool(*fact.Value.Bool), "", remediation)
	}
}

func (m *matcher) requireNamedText(facts []modelinventory.Fact, name string, domain evidenceDomain, constraint, expected string, code RejectionCode, remediation string) {
	fact, ok := findFact(facts, name)
	if !ok {
		m.unknown(constraint, expected, remediation)
		return
	}
	m.observeEvidence(fact, domain, constraint)
	if fact.Value.Text == nil {
		m.wrongType(constraint, "text", valueType(fact.Value), remediation)
		return
	}
	if !strings.EqualFold(*fact.Value.Text, expected) {
		m.reject(code, constraint, expected, *fact.Value.Text, "", remediation)
	}
}

func (m *matcher) requireNamedMinimum(facts []modelinventory.Fact, name string, domain evidenceDomain, constraint string, minimum int64, code RejectionCode, remediation string) {
	fact, ok := findFact(facts, name)
	if !ok {
		m.unknown(constraint, fmt.Sprintf(">=%d", minimum), remediation)
		return
	}
	m.observeEvidence(fact, domain, constraint)
	if fact.Value.Integer == nil {
		m.wrongType(constraint, "integer", valueType(fact.Value), remediation)
		return
	}
	if *fact.Value.Integer < minimum {
		m.reject(code, constraint, fmt.Sprintf(">=%d", minimum), strconv.FormatInt(*fact.Value.Integer, 10), "", remediation)
	}
}

func (m *matcher) requireNamedMaximum(facts []modelinventory.Fact, name string, domain evidenceDomain, constraint string, maximum int64, code RejectionCode, remediation string) {
	fact, ok := findFact(facts, name)
	if !ok {
		m.unknown(constraint, fmt.Sprintf("<=%d", maximum), remediation)
		return
	}
	m.observeEvidence(fact, domain, constraint)
	if fact.Value.Integer == nil {
		m.wrongType(constraint, "integer", valueType(fact.Value), remediation)
		return
	}
	if *fact.Value.Integer > maximum {
		m.reject(code, constraint, fmt.Sprintf("<=%d", maximum), strconv.FormatInt(*fact.Value.Integer, 10), "", remediation)
	}
}

func (m *matcher) requireNamedAllowed(facts []modelinventory.Fact, name string, domain evidenceDomain, constraint string, allowed []string, code RejectionCode, remediation string) {
	fact, ok := findFact(facts, name)
	expected := stableSet(allowed)
	if !ok {
		m.unknown(constraint, expected, remediation)
		return
	}
	m.observeEvidence(fact, domain, constraint)
	if fact.Value.Text == nil {
		m.wrongType(constraint, "text", valueType(fact.Value), remediation)
		return
	}
	if !containsFold(allowed, *fact.Value.Text) {
		m.reject(code, constraint, expected, *fact.Value.Text, "", remediation)
	}
}

func (m *matcher) requireModality(modality harnessmodelset.Modality) {
	constraint := "capabilities.modalities." + string(modality)
	remediation := "use a candidate with the declared witnessed modality"
	if fact, ok := findFact(m.candidate.Evidence.Capabilities, "modality."+string(modality)); ok {
		m.requireBool(fact, domainModelBehavior, constraint, true, CodeModality, remediation)
		return
	}
	if fact, ok := findFact(m.candidate.Evidence.Capabilities, "modality"); ok {
		m.observeEvidence(fact, domainModelBehavior, constraint)
		if fact.Value.Text == nil {
			m.wrongType(constraint, "text", valueType(fact.Value), remediation)
			return
		}
		if !strings.EqualFold(*fact.Value.Text, string(modality)) {
			m.reject(CodeModality, constraint, string(modality), *fact.Value.Text, "", remediation)
		}
		return
	}
	m.unknown(constraint, string(modality), remediation)
}

func (m *matcher) requirePlatform(allowed []string) {
	constraint := "operational.platforms"
	remediation := "use a candidate witnessed on an allowed operating-system and architecture pair"
	if fact, ok := findFact(m.candidate.Evidence.Platform, "platform"); ok {
		m.observeEvidence(fact, domainRuntime, constraint)
		if fact.Value.Text == nil {
			m.wrongType(constraint, "text", valueType(fact.Value), remediation)
			return
		}
		if !containsFold(allowed, *fact.Value.Text) {
			m.reject(CodePlatform, constraint, stableSet(allowed), *fact.Value.Text, "", remediation)
		}
		return
	}
	osFact, osOK := findFact(m.candidate.Evidence.Platform, "os")
	archFact, archOK := findFact(m.candidate.Evidence.Platform, "architecture")
	if !osOK || !archOK {
		m.unknown(constraint, stableSet(allowed), remediation)
		return
	}
	m.observeEvidence(osFact, domainRuntime, constraint+".os")
	m.observeEvidence(archFact, domainRuntime, constraint+".architecture")
	if osFact.Value.Text == nil || archFact.Value.Text == nil {
		m.wrongType(constraint, "text os and architecture", valueType(osFact.Value)+"/"+valueType(archFact.Value), remediation)
		return
	}
	actual := *osFact.Value.Text + "/" + *archFact.Value.Text
	if !containsFold(allowed, actual) {
		m.reject(CodePlatform, constraint, stableSet(allowed), actual, "", remediation)
	}
}

func (m *matcher) requireLocality(expected harnessmodelset.Locality) {
	const constraint = "operational.locality"
	const remediation = "use a candidate whose witnessed locality satisfies the declared boundary"
	fact, ok := findFact(m.candidate.Evidence.Policy, "locality")
	if !ok {
		m.unknown(constraint, string(expected), remediation)
		return
	}
	m.observeEvidence(fact, domainAttestation, constraint)
	if fact.Value.Text == nil {
		m.wrongType(constraint, "text", valueType(fact.Value), remediation)
		return
	}
	actual := strings.ToLower(*fact.Value.Text)
	compatible := false
	switch expected {
	case harnessmodelset.LocalityLocalOnly:
		compatible = actual == "local" || actual == string(harnessmodelset.LocalityLocalOnly)
	case harnessmodelset.LocalityRemoteAllowed:
		compatible = actual == "local" || actual == string(harnessmodelset.LocalityLocalOnly) || actual == "remote" || actual == string(harnessmodelset.LocalityRemoteAllowed)
	}
	if !compatible {
		m.reject(CodeLocality, constraint, string(expected), *fact.Value.Text, "", remediation)
	}
}

func (m *matcher) observeEvidence(fact modelinventory.Fact, domain evidenceDomain, constraint string) {
	maxAge := maxAgeDuration(m.role.Evidence.MaxAgeHours)
	fresh := false
	var newest time.Time
	newestSource := ""
	for _, witness := range fact.Witnesses {
		observed, err := time.Parse(time.RFC3339, witness.ObservedAt)
		if err != nil {
			continue
		}
		age := m.asOf.Sub(observed)
		if age >= 0 && age <= maxAge {
			fresh = true
			if kind, ok := semanticEvidenceKind(domain, witness.Kind); ok {
				m.observedKinds[kind] = struct{}{}
			}
		}
		if observed.After(newest) || (observed.Equal(newest) && (newestSource == "" || witness.Source < newestSource)) {
			newest = observed
			newestSource = witness.Source
		}
	}
	if !fresh {
		cutoff := m.asOf.Add(-maxAge).Format(time.RFC3339)
		actual := "unknown"
		if !newest.IsZero() {
			actual = newest.UTC().Format(time.RFC3339)
		}
		m.reject(CodeEvidenceStale, constraint+".evidence", ">="+cutoff, actual, newestSource,
			"refresh the named evidence within the role's maximum age")
	}
}

func (m *matcher) requireEvidenceKinds() {
	actual := make([]string, 0, len(m.observedKinds))
	for kind := range m.observedKinds {
		actual = append(actual, string(kind))
	}
	sort.Strings(actual)
	actualText := strings.Join(actual, ",")
	if actualText == "" {
		actualText = "none"
	}
	for _, required := range m.role.Evidence.RequiredKinds {
		if _, ok := m.observedKinds[required]; ok {
			continue
		}
		m.reject(CodeEvidenceKindMissing, "evidence.required_kinds", string(required), actualText, "",
			"attach fresh evidence of every kind required by the role")
	}
}

func (m *matcher) unknown(constraint, expected, remediation string) {
	m.reject(CodeFactUnknown, constraint, expected, "unknown", "", remediation)
}

func (m *matcher) wrongType(constraint, expected, actual, remediation string) {
	m.reject(CodeFactType, constraint, expected, actual, "", remediation)
}

func (m *matcher) reject(code RejectionCode, constraint, expected, actual, source, remediation string) {
	m.rejections = append(m.rejections, Rejection{
		RoleID:           m.role.ID,
		AlternativeID:    m.alternative.ID,
		AlternativeIndex: m.alternativeIndex,
		CandidateID:      m.candidate.ID,
		Code:             code,
		Constraint:       constraint,
		Expected:         expected,
		Actual:           actual,
		EvidenceSource:   source,
		Remediation:      remediation,
	})
}

func findFact(facts []modelinventory.Fact, name string) (modelinventory.Fact, bool) {
	for _, fact := range facts {
		if fact.Name == name {
			return fact, true
		}
	}
	return modelinventory.Fact{}, false
}

func integerFact(facts []modelinventory.Fact, name string) (int64, bool) {
	fact, ok := findFact(facts, name)
	if !ok || fact.Value.Integer == nil {
		return 0, false
	}
	return *fact.Value.Integer, true
}

func semanticEvidenceKind(domain evidenceDomain, kind modelinventory.WitnessKind) (harnessmodelset.EvidenceKind, bool) {
	if kind == modelinventory.EvidenceOperatorAttestation {
		return harnessmodelset.EvidenceOperatorAttestation, true
	}
	if kind != modelinventory.EvidenceProbe {
		return "", false
	}
	switch domain {
	case domainModelBehavior:
		return harnessmodelset.EvidenceModelBehaviorProbe, true
	case domainRuntime:
		return harnessmodelset.EvidenceRuntimeProbe, true
	default:
		return "", false
	}
}

func maxAgeDuration(hours int64) time.Duration {
	if hours >= int64(math.MaxInt64)/int64(time.Hour) {
		return time.Duration(math.MaxInt64)
	}
	return time.Duration(hours) * time.Hour
}

func containsFold(values []string, candidate string) bool {
	for _, value := range values {
		if strings.EqualFold(value, candidate) {
			return true
		}
	}
	return false
}

func stableSet(values []string) string {
	out := append([]string(nil), values...)
	sort.Slice(out, func(i, j int) bool {
		left, right := strings.ToLower(out[i]), strings.ToLower(out[j])
		if left != right {
			return left < right
		}
		return out[i] < out[j]
	})
	return strings.Join(out, ",")
}

func valueType(value modelinventory.Value) string {
	switch {
	case value.Bool != nil:
		return "boolean"
	case value.Integer != nil:
		return "integer"
	case value.Text != nil:
		return "text"
	default:
		return "unknown"
	}
}
