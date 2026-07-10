package modelroute

import (
	"fmt"
	"sort"
	"strings"
)

const (
	AuditIndependencePolicyVersion = "audit-independence/v1"
	AuditIdentityRosterSchema      = "fak-audit-identity-roster/v1"
)

type AuditIdentityAxis string

const (
	AuditAxisProvider   AuditIdentityAxis = "provider"
	AuditAxisFamily     AuditIdentityAxis = "family"
	AuditAxisModel      AuditIdentityAxis = "model"
	AuditAxisWeights    AuditIdentityAxis = "weights_revision"
	AuditAxisHarness    AuditIdentityAxis = "harness"
	AuditAxisEndpoint   AuditIdentityAxis = "endpoint_class"
	AuditAxisAccount    AuditIdentityAxis = "account_class"
	AuditAxisEffort     AuditIdentityAxis = "effort"
	AuditAxisProvenance AuditIdentityAxis = "provenance_source"
)

type AuditRisk string

const (
	AuditRiskDefault AuditRisk = "default"
	AuditRiskHigh    AuditRisk = "high"
)

type AuditIndependenceVerdict string

const (
	AuditIndependenceAdmit   AuditIndependenceVerdict = "ADMIT"
	AuditIndependenceRefuse  AuditIndependenceVerdict = "REFUSE"
	AuditIndependenceUnknown AuditIndependenceVerdict = "UNKNOWN"
)

type AuditIndependenceReason string

const (
	AuditReasonAdmitIndependent           AuditIndependenceReason = "ADMIT_INDEPENDENT"
	AuditReasonRefuseSameFamily           AuditIndependenceReason = "REFUSE_SAME_FAMILY"
	AuditReasonRefuseSameWeights          AuditIndependenceReason = "REFUSE_SAME_WEIGHTS"
	AuditReasonRefuseSameProviderHighRisk AuditIndependenceReason = "REFUSE_SAME_PROVIDER_HIGH_RISK"
	AuditReasonRefuseSameHarness          AuditIndependenceReason = "REFUSE_SAME_HARNESS"
	AuditReasonRefuseSameEndpoint         AuditIndependenceReason = "REFUSE_SAME_ENDPOINT"
	AuditReasonRefuseSameAccount          AuditIndependenceReason = "REFUSE_SAME_ACCOUNT"
	AuditReasonRefuseAliasConflict        AuditIndependenceReason = "REFUSE_ALIAS_IDENTITY_CONFLICT"
	AuditReasonRefuseObservedMismatch     AuditIndependenceReason = "REFUSE_OBSERVED_IDENTITY_MISMATCH"
	AuditReasonUnknownRequiredAxis        AuditIndependenceReason = "UNKNOWN_REQUIRED_AXIS"
	AuditReasonUnknownAliasAmbiguous      AuditIndependenceReason = "UNKNOWN_ALIAS_AMBIGUOUS"
	AuditReasonUnknownAliasUnresolved     AuditIndependenceReason = "UNKNOWN_ALIAS_UNRESOLVED"
	AuditReasonUnknownPolicy              AuditIndependenceReason = "UNKNOWN_POLICY_VERSION"
	AuditReasonUnknownPolicyRisk          AuditIndependenceReason = "UNKNOWN_POLICY_RISK"
	AuditReasonUnknownPolicyAxis          AuditIndependenceReason = "UNKNOWN_POLICY_AXIS"
	AuditReasonUnknownObservedIdentity    AuditIndependenceReason = "UNKNOWN_OBSERVED_IDENTITY"
)

// AuditIdentity is the canonical provenance row shared by issue authors and
// auditors. Empty fields remain empty: normalization never guesses lineage from
// a display name. An AuditIdentityAlias roster entry is the only mechanism that
// may fill provider/family/weights for an alias.
type AuditIdentity struct {
	Harness          string `json:"harness,omitempty"`
	Provider         string `json:"provider"`
	Family           string `json:"family"`
	Model            string `json:"model"`
	WeightsRevision  string `json:"weights_revision,omitempty"`
	AccountClass     string `json:"account_class,omitempty"`
	EndpointClass    string `json:"endpoint_class,omitempty"`
	ReasoningPosture string `json:"reasoning_posture,omitempty"`
	ProvenanceSource string `json:"provenance_source,omitempty"`
	Driver           string `json:"driver,omitempty"`
}

// ModelIdentity is the v1 spine's compatibility name. It is an alias rather
// than a second struct, so old manifests parse unchanged while every admission
// now uses the canonical AuditIdentity policy.
type ModelIdentity = AuditIdentity

// AuditIdentityAlias is one independently configured roster fact. Alias is the
// request/display name; CanonicalModel, provider, family, and weights describe
// what it really resolves to. ProvenanceSource identifies the roster evidence.
type AuditIdentityAlias struct {
	Alias            string `json:"alias"`
	CanonicalModel   string `json:"canonical_model"`
	Provider         string `json:"provider"`
	Family           string `json:"family"`
	WeightsRevision  string `json:"weights_revision,omitempty"`
	ProvenanceSource string `json:"provenance_source"`
}

type AuditIdentityRoster struct {
	Schema  string               `json:"schema"`
	Aliases []AuditIdentityAlias `json:"aliases"`
}

func (roster AuditIdentityRoster) Validate() error {
	if roster.Schema != AuditIdentityRosterSchema {
		return fmt.Errorf("modelroute: audit identity roster schema %q, want %q", roster.Schema, AuditIdentityRosterSchema)
	}
	if len(roster.Aliases) == 0 {
		return fmt.Errorf("modelroute: audit identity roster has no aliases")
	}
	seen := map[string]AuditIdentityAlias{}
	for i, raw := range roster.Aliases {
		alias := normalizeAuditAlias(raw)
		if alias.Alias == "" || alias.CanonicalModel == "" || alias.Provider == "" || alias.Family == "" || alias.WeightsRevision == "" || alias.ProvenanceSource == "" {
			return fmt.Errorf("modelroute: audit identity roster alias %d requires alias, canonical_model, provider, family, weights_revision, and provenance_source", i)
		}
		if prior, ok := seen[alias.Alias]; ok && prior != alias {
			return fmt.Errorf("modelroute: audit identity roster alias %q maps to conflicting canonical facts", alias.Alias)
		}
		seen[alias.Alias] = alias
	}
	return nil
}

type AuditIndependencePolicy struct {
	Version                  string               `json:"version"`
	Risk                     AuditRisk            `json:"risk"`
	RequiredAxes             []AuditIdentityAxis  `json:"required_axes"`
	RequireFamilyDiversity   bool                 `json:"require_family_diversity"`
	RequireProviderDiversity bool                 `json:"require_provider_diversity,omitempty"`
	RequireHarnessDiversity  bool                 `json:"require_harness_diversity,omitempty"`
	RequireEndpointDiversity bool                 `json:"require_endpoint_diversity,omitempty"`
	RequireAccountDiversity  bool                 `json:"require_account_diversity,omitempty"`
	RequireRosterResolution  bool                 `json:"require_roster_resolution"`
	Aliases                  []AuditIdentityAlias `json:"aliases,omitempty"`
}

type AuditIndependenceDecision struct {
	Verdict       AuditIndependenceVerdict `json:"verdict"`
	Reason        AuditIndependenceReason  `json:"reason"`
	PolicyVersion string                   `json:"policy_version"`
	PolicyDigest  string                   `json:"policy_digest"`
	Risk          AuditRisk                `json:"risk"`
	Author        AuditIdentity            `json:"author"`
	Auditor       AuditIdentity            `json:"auditor"`
	MissingAxes   []string                 `json:"missing_axes,omitempty"`
}

func DefaultAuditIndependencePolicy() AuditIndependencePolicy {
	return AuditIndependencePolicy{
		Version: AuditIndependencePolicyVersion,
		Risk:    AuditRiskDefault,
		RequiredAxes: []AuditIdentityAxis{
			AuditAxisProvider, AuditAxisFamily, AuditAxisModel, AuditAxisWeights,
			AuditAxisHarness, AuditAxisEndpoint, AuditAxisAccount, AuditAxisEffort, AuditAxisProvenance,
		},
		RequireFamilyDiversity:  true,
		RequireRosterResolution: true,
	}
}

func HighRiskAuditIndependencePolicy() AuditIndependencePolicy {
	p := DefaultAuditIndependencePolicy()
	p.Risk = AuditRiskHigh
	p.RequireProviderDiversity = true
	return p
}

func (v AuditIndependenceVerdict) Valid() bool {
	return v == AuditIndependenceAdmit || v == AuditIndependenceRefuse || v == AuditIndependenceUnknown
}

func (r AuditIndependenceReason) Valid() bool {
	switch r {
	case AuditReasonAdmitIndependent,
		AuditReasonRefuseSameFamily,
		AuditReasonRefuseSameWeights,
		AuditReasonRefuseSameProviderHighRisk,
		AuditReasonRefuseSameHarness,
		AuditReasonRefuseSameEndpoint,
		AuditReasonRefuseSameAccount,
		AuditReasonRefuseAliasConflict,
		AuditReasonRefuseObservedMismatch,
		AuditReasonUnknownRequiredAxis,
		AuditReasonUnknownAliasAmbiguous,
		AuditReasonUnknownAliasUnresolved,
		AuditReasonUnknownPolicy,
		AuditReasonUnknownPolicyRisk,
		AuditReasonUnknownPolicyAxis,
		AuditReasonUnknownObservedIdentity:
		return true
	default:
		return false
	}
}

// EvaluateAuditIndependence is the pure admission PDP. It canonicalizes both
// rows through the policy roster, returns only the closed verdict/reason
// vocabulary, refuses same weights even when aliases claim different families,
// and never guesses a required missing axis.
func EvaluateAuditIndependence(author, auditor AuditIdentity, policy AuditIndependencePolicy) AuditIndependenceDecision {
	policy = normalizeAuditPolicy(policy)
	decision := AuditIndependenceDecision{
		PolicyVersion: policy.Version,
		PolicyDigest:  policy.Digest(),
		Risk:          policy.Risk,
	}
	if policy.Version != AuditIndependencePolicyVersion {
		decision.Verdict = AuditIndependenceUnknown
		decision.Reason = AuditReasonUnknownPolicy
		decision.Author = normalizeAuditIdentityFields(author)
		decision.Auditor = normalizeAuditIdentityFields(auditor)
		return decision
	}
	if policy.Risk != AuditRiskDefault && policy.Risk != AuditRiskHigh {
		decision.Verdict = AuditIndependenceUnknown
		decision.Reason = AuditReasonUnknownPolicyRisk
		decision.Author = normalizeAuditIdentityFields(author)
		decision.Auditor = normalizeAuditIdentityFields(auditor)
		return decision
	}
	for _, axis := range policy.RequiredAxes {
		if !validAuditIdentityAxis(axis) {
			decision.Verdict = AuditIndependenceUnknown
			decision.Reason = AuditReasonUnknownPolicyAxis
			decision.Author = normalizeAuditIdentityFields(author)
			decision.Auditor = normalizeAuditIdentityFields(auditor)
			decision.MissingAxes = []string{string(axis)}
			return decision
		}
	}

	var authorStatus, auditorStatus identityNormalizationStatus
	decision.Author, authorStatus = normalizeAuditIdentity(author, policy.Aliases)
	decision.Auditor, auditorStatus = normalizeAuditIdentity(auditor, policy.Aliases)
	if authorStatus == identityAliasConflict || auditorStatus == identityAliasConflict {
		decision.Verdict = AuditIndependenceRefuse
		decision.Reason = AuditReasonRefuseAliasConflict
		return decision
	}
	if sameKnown(decision.Author.WeightsRevision, decision.Auditor.WeightsRevision) {
		decision.Verdict = AuditIndependenceRefuse
		decision.Reason = AuditReasonRefuseSameWeights
		return decision
	}
	if policy.RequireFamilyDiversity && sameKnown(decision.Author.Family, decision.Auditor.Family) {
		decision.Verdict = AuditIndependenceRefuse
		decision.Reason = AuditReasonRefuseSameFamily
		return decision
	}
	if (policy.RequireProviderDiversity || policy.Risk == AuditRiskHigh) && sameKnown(decision.Author.Provider, decision.Auditor.Provider) {
		decision.Verdict = AuditIndependenceRefuse
		decision.Reason = AuditReasonRefuseSameProviderHighRisk
		return decision
	}
	if policy.RequireHarnessDiversity && sameKnown(decision.Author.Harness, decision.Auditor.Harness) {
		decision.Verdict = AuditIndependenceRefuse
		decision.Reason = AuditReasonRefuseSameHarness
		return decision
	}
	if policy.RequireEndpointDiversity && sameKnown(decision.Author.EndpointClass, decision.Auditor.EndpointClass) {
		decision.Verdict = AuditIndependenceRefuse
		decision.Reason = AuditReasonRefuseSameEndpoint
		return decision
	}
	if policy.RequireAccountDiversity && sameKnown(decision.Author.AccountClass, decision.Auditor.AccountClass) {
		decision.Verdict = AuditIndependenceRefuse
		decision.Reason = AuditReasonRefuseSameAccount
		return decision
	}
	if authorStatus == identityAliasAmbiguous || auditorStatus == identityAliasAmbiguous {
		decision.Verdict = AuditIndependenceUnknown
		decision.Reason = AuditReasonUnknownAliasAmbiguous
		return decision
	}
	if policy.RequireRosterResolution && (authorStatus != identityRosterResolved || auditorStatus != identityRosterResolved) {
		decision.Verdict = AuditIndependenceUnknown
		decision.Reason = AuditReasonUnknownAliasUnresolved
		return decision
	}
	decision.MissingAxes = missingAuditAxes(decision.Author, decision.Auditor, policy.RequiredAxes)
	if len(decision.MissingAxes) > 0 {
		decision.Verdict = AuditIndependenceUnknown
		decision.Reason = AuditReasonUnknownRequiredAxis
		return decision
	}
	decision.Verdict = AuditIndependenceAdmit
	decision.Reason = AuditReasonAdmitIndependent
	return decision
}

// VerifyObservedAuditIdentity compares a declared auditor with independently
// read response identity. Provider/family/weights/model are required: an HTTP
// response that only echoes a model display name is UNKNOWN unless roster data
// canonicalizes that name onto all four axes.
func VerifyObservedAuditIdentity(expected, observed AuditIdentity, aliases []AuditIdentityAlias) AuditIndependenceDecision {
	policy := DefaultAuditIndependencePolicy()
	policy.RequiredAxes = []AuditIdentityAxis{AuditAxisProvider, AuditAxisFamily, AuditAxisModel, AuditAxisWeights}
	policy.Aliases = aliases
	decision := AuditIndependenceDecision{
		PolicyVersion: policy.Version,
		PolicyDigest:  policy.Digest(),
		Risk:          policy.Risk,
	}
	var expectedStatus, observedStatus identityNormalizationStatus
	decision.Author, expectedStatus = normalizeAuditIdentity(expected, aliases)
	decision.Auditor, observedStatus = normalizeAuditIdentity(observed, aliases)
	if expectedStatus == identityAliasConflict || observedStatus == identityAliasConflict {
		decision.Verdict = AuditIndependenceRefuse
		decision.Reason = AuditReasonRefuseAliasConflict
		return decision
	}
	if expectedStatus == identityAliasAmbiguous || observedStatus == identityAliasAmbiguous {
		decision.Verdict = AuditIndependenceUnknown
		decision.Reason = AuditReasonUnknownAliasAmbiguous
		return decision
	}
	if expectedStatus != identityRosterResolved || observedStatus != identityRosterResolved {
		decision.Verdict = AuditIndependenceUnknown
		decision.Reason = AuditReasonUnknownObservedIdentity
		return decision
	}
	decision.MissingAxes = missingAuditAxes(decision.Author, decision.Auditor, policy.RequiredAxes)
	if len(decision.MissingAxes) > 0 {
		decision.Verdict = AuditIndependenceUnknown
		decision.Reason = AuditReasonUnknownObservedIdentity
		return decision
	}
	for _, axis := range policy.RequiredAxes {
		if !strings.EqualFold(auditAxisValue(decision.Author, axis), auditAxisValue(decision.Auditor, axis)) {
			decision.Verdict = AuditIndependenceRefuse
			decision.Reason = AuditReasonRefuseObservedMismatch
			return decision
		}
	}
	decision.Verdict = AuditIndependenceAdmit
	decision.Reason = AuditReasonAdmitIndependent
	return decision
}

// Digest content-addresses the normalized policy plus its authoritative alias
// roster. Slice order is canonicalized so the same policy facts produce the
// same digest regardless of configuration ordering.
func (policy AuditIndependencePolicy) Digest() string {
	policy = normalizeAuditPolicy(policy)
	policy.RequiredAxes = append([]AuditIdentityAxis(nil), policy.RequiredAxes...)
	sort.Slice(policy.RequiredAxes, func(i, j int) bool { return policy.RequiredAxes[i] < policy.RequiredAxes[j] })
	policy.Aliases = append([]AuditIdentityAlias(nil), policy.Aliases...)
	for i := range policy.Aliases {
		policy.Aliases[i] = normalizeAuditAlias(policy.Aliases[i])
	}
	sort.Slice(policy.Aliases, func(i, j int) bool {
		a, b := policy.Aliases[i], policy.Aliases[j]
		return a.Alias+"\x00"+a.CanonicalModel+"\x00"+a.Provider+"\x00"+a.Family+"\x00"+a.WeightsRevision+"\x00"+a.ProvenanceSource <
			b.Alias+"\x00"+b.CanonicalModel+"\x00"+b.Provider+"\x00"+b.Family+"\x00"+b.WeightsRevision+"\x00"+b.ProvenanceSource
	})
	return hashJSON(policy)
}

// ValidateAuditDriverIdentity binds the executable driver to a concrete model
// grammar. Display aliases such as claude-alias and gpt-alt are refused; a
// roster must canonicalize them before this boundary if they are legitimate.
func ValidateAuditDriverIdentity(driver string, id AuditIdentity, aliases []AuditIdentityAlias) (AuditIdentity, error) {
	normalized, status := normalizeAuditIdentity(id, aliases)
	if status != identityRosterResolved {
		return normalized, fmt.Errorf("modelroute: audit driver identity normalization failed: %s", status)
	}
	driver = strings.ToLower(strings.TrimSpace(driver))
	switch driver {
	case "http":
	case "claude":
		if normalized.Provider != "anthropic" || normalized.Family != "claude" {
			return normalized, fmt.Errorf("modelroute: claude driver requires roster-canonical anthropic/claude identity")
		}
	case "codex":
		if normalized.Provider != "openai" || (normalized.Family != "gpt" && normalized.Family != "openai-o") {
			return normalized, fmt.Errorf("modelroute: codex driver requires roster-canonical openai/gpt-or-openai-o identity")
		}
	default:
		return normalized, fmt.Errorf("modelroute: unsupported audit driver %q", driver)
	}
	normalized.Driver = driver
	return normalized, nil
}

// AuditDriverRequiresObservedIdentity reports whether a driver's trust
// contract requires the response to carry an independently readable model
// identity. Keep this closed and capability-based: callers may opt into a
// stronger check, but cannot opt a driver out of its required readback.
func AuditDriverRequiresObservedIdentity(driver string) bool {
	switch strings.ToLower(strings.TrimSpace(driver)) {
	case "http":
		return true
	default:
		return false
	}
}

type identityNormalizationStatus string

const (
	identityExplicit       identityNormalizationStatus = "explicit_unresolved"
	identityRosterResolved identityNormalizationStatus = "roster_resolved"
	identityAliasConflict  identityNormalizationStatus = "alias_conflict"
	identityAliasAmbiguous identityNormalizationStatus = "alias_ambiguous"
)

func normalizeAuditPolicy(policy AuditIndependencePolicy) AuditIndependencePolicy {
	if strings.TrimSpace(policy.Version) == "" {
		base := DefaultAuditIndependencePolicy()
		base.Risk = policy.Risk
		base.Aliases = append([]AuditIdentityAlias(nil), policy.Aliases...)
		base.RequireProviderDiversity = policy.RequireProviderDiversity
		base.RequireHarnessDiversity = policy.RequireHarnessDiversity
		base.RequireEndpointDiversity = policy.RequireEndpointDiversity
		base.RequireAccountDiversity = policy.RequireAccountDiversity
		if len(policy.RequiredAxes) > 0 {
			base.RequiredAxes = append([]AuditIdentityAxis(nil), policy.RequiredAxes...)
		}
		policy = base
	}
	if policy.Risk == "" {
		policy.Risk = AuditRiskDefault
	}
	if policy.Version == AuditIndependencePolicyVersion {
		// V1's floor is not caller-disableable: every canonical provenance axis
		// is required, family diversity is mandatory, and aliases must resolve
		// through authoritative roster facts.
		seen := make(map[AuditIdentityAxis]bool, len(policy.RequiredAxes))
		for _, axis := range policy.RequiredAxes {
			seen[axis] = true
		}
		for _, axis := range DefaultAuditIndependencePolicy().RequiredAxes {
			if !seen[axis] {
				policy.RequiredAxes = append(policy.RequiredAxes, axis)
			}
		}
		policy.RequireFamilyDiversity = true
		policy.RequireRosterResolution = true
	}
	return policy
}

func normalizeAuditIdentity(id AuditIdentity, aliases []AuditIdentityAlias) (AuditIdentity, identityNormalizationStatus) {
	id = normalizeAuditIdentityFields(id)
	var matches []AuditIdentityAlias
	for _, alias := range aliases {
		alias = normalizeAuditAlias(alias)
		if id.Model != "" && (strings.EqualFold(id.Model, alias.Alias) || strings.EqualFold(id.Model, alias.CanonicalModel)) {
			matches = append(matches, alias)
		}
	}
	if len(matches) == 0 {
		return id, identityExplicit
	}
	first := matches[0]
	for _, match := range matches[1:] {
		if match.CanonicalModel != first.CanonicalModel || match.Provider != first.Provider || match.Family != first.Family || match.WeightsRevision != first.WeightsRevision || match.ProvenanceSource != first.ProvenanceSource {
			return id, identityAliasAmbiguous
		}
	}
	if conflictsKnown(id.Provider, first.Provider) || conflictsKnown(id.Family, first.Family) || conflictsKnown(id.WeightsRevision, first.WeightsRevision) {
		return id, identityAliasConflict
	}
	if first.CanonicalModel != "" {
		id.Model = first.CanonicalModel
	}
	if id.Provider == "" {
		id.Provider = first.Provider
	}
	if id.Family == "" {
		id.Family = first.Family
	}
	if id.WeightsRevision == "" {
		id.WeightsRevision = first.WeightsRevision
	}
	if id.ProvenanceSource == "" {
		id.ProvenanceSource = first.ProvenanceSource
	}
	return id, identityRosterResolved
}

func validAuditIdentityAxis(axis AuditIdentityAxis) bool {
	switch axis {
	case AuditAxisProvider, AuditAxisFamily, AuditAxisModel, AuditAxisWeights, AuditAxisHarness, AuditAxisEndpoint, AuditAxisAccount, AuditAxisEffort, AuditAxisProvenance:
		return true
	default:
		return false
	}
}

func normalizeAuditIdentityFields(id AuditIdentity) AuditIdentity {
	id.Harness = strings.ToLower(strings.TrimSpace(id.Harness))
	id.Provider = strings.ToLower(strings.TrimSpace(id.Provider))
	id.Family = strings.ToLower(strings.TrimSpace(id.Family))
	id.Model = strings.ToLower(strings.TrimSpace(id.Model))
	id.WeightsRevision = strings.ToLower(strings.TrimSpace(id.WeightsRevision))
	id.AccountClass = strings.ToLower(strings.TrimSpace(id.AccountClass))
	id.EndpointClass = strings.ToLower(strings.TrimSpace(id.EndpointClass))
	id.ReasoningPosture = strings.ToLower(strings.TrimSpace(id.ReasoningPosture))
	id.ProvenanceSource = strings.TrimSpace(id.ProvenanceSource)
	id.Driver = strings.ToLower(strings.TrimSpace(id.Driver))
	return id
}

func normalizeAuditAlias(alias AuditIdentityAlias) AuditIdentityAlias {
	alias.Alias = strings.ToLower(strings.TrimSpace(alias.Alias))
	alias.CanonicalModel = strings.ToLower(strings.TrimSpace(alias.CanonicalModel))
	alias.Provider = strings.ToLower(strings.TrimSpace(alias.Provider))
	alias.Family = strings.ToLower(strings.TrimSpace(alias.Family))
	alias.WeightsRevision = strings.ToLower(strings.TrimSpace(alias.WeightsRevision))
	alias.ProvenanceSource = strings.TrimSpace(alias.ProvenanceSource)
	return alias
}

func missingAuditAxes(author, auditor AuditIdentity, axes []AuditIdentityAxis) []string {
	var missing []string
	for _, axis := range axes {
		if auditAxisValue(author, axis) == "" {
			missing = append(missing, "author."+string(axis))
		}
		if auditAxisValue(auditor, axis) == "" {
			missing = append(missing, "auditor."+string(axis))
		}
	}
	sort.Strings(missing)
	return missing
}

func auditAxisValue(id AuditIdentity, axis AuditIdentityAxis) string {
	switch axis {
	case AuditAxisProvider:
		return id.Provider
	case AuditAxisFamily:
		return id.Family
	case AuditAxisModel:
		return id.Model
	case AuditAxisWeights:
		return id.WeightsRevision
	case AuditAxisHarness:
		return id.Harness
	case AuditAxisEndpoint:
		return id.EndpointClass
	case AuditAxisAccount:
		return id.AccountClass
	case AuditAxisEffort:
		return id.ReasoningPosture
	case AuditAxisProvenance:
		return id.ProvenanceSource
	default:
		return ""
	}
}

func sameKnown(a, b string) bool {
	return a != "" && b != "" && strings.EqualFold(a, b)
}

func conflictsKnown(got, canonical string) bool {
	return got != "" && canonical != "" && !strings.EqualFold(got, canonical)
}
