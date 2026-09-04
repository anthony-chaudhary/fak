// Package quantpolicy evaluates explicit policy constraints over quantization
// capability metadata. It is format-neutral: callers name formats and evidence
// they approve, and this package performs structural checks without selecting a
// quantizer, converting an artifact, or inferring runtime or hardware support.
//
// Invariant: quant policy evaluation is fail-closed and structural across all predicates.
// Guard: unknown schemas, unknown metadata, unsupported operations, and unapproved
// formats or provenance immediately refuse or abstain without runtime side effects.
package quantpolicy

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/strictjson"
	"math"
	"strings"
)

const (
	// SchemaV1 is the public policy document ID supported by this package.
	SchemaV1 = "quantpolicy.policy/v1"
	// MetadataSchemaV1 is the public capability-metadata document ID supported by this package.
	MetadataSchemaV1 = "quantpolicy.metadata/v1"
)

// Outcome is the stable disposition returned by Evaluate.
type Outcome string

const (
	OutcomeAllow    Outcome = "allow"
	OutcomeRefuse   Outcome = "refuse"
	OutcomeAbstain  Outcome = "abstain"
	OutcomeDelegate Outcome = "delegate"
)

// ReasonCode is a stable, machine-readable explanation for a decision.
type ReasonCode string

const (
	ReasonSatisfied             ReasonCode = "QUANT_ALLOW"
	ReasonInvalidContract       ReasonCode = "QUANT_INVALID_CONTRACT"
	ReasonUnknownSchema         ReasonCode = "QUANT_UNKNOWN_SCHEMA"
	ReasonInvalidJSON           ReasonCode = "QUANT_INVALID_JSON"
	ReasonUnknownMetadata       ReasonCode = "QUANT_UNKNOWN_METADATA"
	ReasonUnknownMetadataSchema ReasonCode = "QUANT_UNKNOWN_METADATA_SCHEMA"
	ReasonOperationNotHandled   ReasonCode = "QUANT_OPERATION_NOT_HANDLED"
	ReasonConversionRefused     ReasonCode = "QUANT_CONVERSION_REFUSED"
	ReasonUnknownPrecision      ReasonCode = "QUANT_UNKNOWN_PRECISION"
	ReasonBelowMinimumPrecision ReasonCode = "QUANT_BELOW_MINIMUM_PRECISION"
	ReasonAboveMaximumPrecision ReasonCode = "QUANT_ABOVE_MAXIMUM_PRECISION"
	ReasonUnknownFormat         ReasonCode = "QUANT_UNKNOWN_FORMAT"
	ReasonFormatNotApproved     ReasonCode = "QUANT_FORMAT_NOT_APPROVED"
	ReasonProvenanceRequired    ReasonCode = "QUANT_PROVENANCE_REQUIRED"
	ReasonUnknownProvenance     ReasonCode = "QUANT_UNKNOWN_PROVENANCE"
	ReasonProvenanceNotApproved ReasonCode = "QUANT_PROVENANCE_NOT_APPROVED"
)

// PredicateID identifies the exact structural predicate that decided a request.
type PredicateID string

const (
	PredicateContract           PredicateID = "policy"
	PredicateMetadata           PredicateID = "metadata"
	PredicateOperation          PredicateID = "operation"
	PredicateConversion         PredicateID = "conversion"
	PredicateMinimumPrecision   PredicateID = "precision.minimum"
	PredicateMaximumPrecision   PredicateID = "precision.maximum"
	PredicateApprovedFormat     PredicateID = "format.approved"
	PredicateArtifactProvenance PredicateID = "provenance.artifact"
	PredicateRecipeProvenance   PredicateID = "provenance.recipe"
	PredicateRuntimeProvenance  PredicateID = "provenance.runtime_delegation"
	PredicateHardwareProvenance PredicateID = "provenance.hardware_envelope"
	PredicateAll                PredicateID = "all"
)

// Operation is the requested handling of an artifact.
type Operation string

const (
	OperationUseExisting Operation = "use_existing"
	OperationConvert     Operation = "convert"
)

// ConversionRule makes conversion permission explicit rather than using an
// absent boolean whose meaning could be guessed.
type ConversionRule string

const (
	ConversionAllow  ConversionRule = "allow"
	ConversionRefuse ConversionRule = "refuse"
)

// EvidenceKind labels what a provenance reference actually establishes.
// These labels do not imply that fak produced or independently verified it.
type EvidenceKind string

const (
	EvidenceReported  EvidenceKind = "reported"
	EvidenceModeled   EvidenceKind = "modeled"
	EvidenceObserved  EvidenceKind = "observed"
	EvidenceMeasured  EvidenceKind = "measured"
	EvidenceWitnessed EvidenceKind = "witnessed"
)

// FormatRef pins a public artifact-format ID to a version. The values are
// caller-defined and compared exactly; fak does not invent a wrapper format.
type FormatRef struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

// Precision describes the effective scalar bit width claimed by the artifact.
// Name remains visible so a numeric match cannot erase whether the artifact is,
// for example, integer, floating-point, binary, or a codebook format.
type Precision struct {
	Name string  `json:"name"`
	Bits float64 `json:"bits"`
}

// Evidence is one independently readable provenance claim.
type Evidence struct {
	Kind      EvidenceKind `json:"kind"`
	Reference string       `json:"reference"`
}

// ClaimEnvelope keeps artifact identity, conversion recipe, delegated runtime,
// and measured hardware-envelope claims separate. A policy allow for one field
// never silently promotes a claim in another field.
type ClaimEnvelope struct {
	Artifact          Evidence `json:"artifact"`
	Recipe            Evidence `json:"recipe"`
	RuntimeDelegation Evidence `json:"runtime_delegation"`
	HardwareEnvelope  Evidence `json:"hardware_envelope"`
}

// ProvenanceRequirement states whether a claim is mandatory and which explicit
// evidence labels the policy accepts for it.
type ProvenanceRequirement struct {
	Required      bool           `json:"required"`
	ApprovedKinds []EvidenceKind `json:"approved_kinds"`
}

// ProvenanceRequirements applies field-specific requirements to the claim envelope.
type ProvenanceRequirements struct {
	Artifact          ProvenanceRequirement `json:"artifact"`
	Recipe            ProvenanceRequirement `json:"recipe"`
	RuntimeDelegation ProvenanceRequirement `json:"runtime_delegation"`
	HardwareEnvelope  ProvenanceRequirement `json:"hardware_envelope"`
}

// Policy is the public quantpolicy.policy/v1 capability constraint contract.
// Precision bounds are inclusive.
type Policy struct {
	Schema           string                 `json:"schema"`
	MinPrecisionBits float64                `json:"min_precision_bits"`
	MaxPrecisionBits float64                `json:"max_precision_bits"`
	ApprovedFormats  []FormatRef            `json:"approved_formats"`
	Provenance       ProvenanceRequirements `json:"provenance"`
	Conversion       ConversionRule         `json:"conversion"`
}

// Metadata is the policy-facing subset of a neutral quantization descriptor.
// It is deliberately additive and local so callers can adapt public ecosystem
// descriptors without converting their artifact to a fak-owned format.
type Metadata struct {
	Schema     string        `json:"schema"`
	ArtifactID string        `json:"artifact_id"`
	Precision  Precision     `json:"precision"`
	Format     FormatRef     `json:"format"`
	Provenance ClaimEnvelope `json:"provenance"`
}

// Request states the operation and capability metadata being gated.
type Request struct {
	Operation Operation `json:"operation"`
	Metadata  Metadata  `json:"metadata"`
}

// Result is always typed. Claims are copied from the request without upgrading
// their evidence labels, including on refuse, abstain, and delegate outcomes.
type Result struct {
	Outcome   Outcome       `json:"outcome"`
	Reason    ReasonCode    `json:"reason"`
	Predicate PredicateID   `json:"predicate"`
	Detail    string        `json:"detail,omitempty"`
	Action    string        `json:"action"`
	Claims    ClaimEnvelope `json:"claims"`
}

// ParseAndEvaluate strictly decodes a policy and request. Unknown request fields
// return a typed abstention rather than being ignored; malformed/extended policy
// input is refused because its intended constraints cannot be established.
func ParseAndEvaluate(contractJSON, requestJSON []byte) (Result, error) {
	var policy Policy
	if err := decodeStrict(contractJSON, &policy); err != nil {
		return decision(OutcomeRefuse, ReasonInvalidContract, PredicateContract, err.Error(), "Repair or replace the policy before using the artifact.", ClaimEnvelope{}), fmt.Errorf("parse quantpolicy policy: %w", err)
	}
	var request Request
	if err := decodeStrict(requestJSON, &request); err != nil {
		reason := ReasonInvalidJSON
		if strings.Contains(err.Error(), "unknown field") {
			reason = ReasonUnknownMetadata
		}
		return decision(OutcomeAbstain, reason, PredicateMetadata, err.Error(), "Provide metadata that conforms exactly to quantpolicy.metadata/v1.", ClaimEnvelope{}), fmt.Errorf("parse quantpolicy request: %w", err)
	}
	return Evaluate(policy, request), nil
}

// Evaluate applies the complete structural policy. It never converts an
// artifact, guesses absent metadata, substitutes a format, or promotes evidence.
func Evaluate(policy Policy, request Request) Result {
	claims := request.Metadata.Provenance
	if policy.Schema != SchemaV1 {
		return decision(OutcomeAbstain, ReasonUnknownSchema, PredicateContract, policy.Schema, "Use quantpolicy.policy/v1 or delegate to an evaluator that supports this policy schema.", claims)
	}
	if detail := validateContract(policy); detail != "" {
		return decision(OutcomeRefuse, ReasonInvalidContract, PredicateContract, detail, "Repair the policy before using or converting the artifact.", claims)
	}
	if request.Metadata.Schema != MetadataSchemaV1 {
		return decision(OutcomeAbstain, ReasonUnknownMetadataSchema, PredicateMetadata, request.Metadata.Schema, "Provide quantpolicy.metadata/v1 metadata or delegate to a parser that supports this schema.", claims)
	}
	switch request.Operation {
	case OperationUseExisting:
	case OperationConvert:
		if policy.Conversion == ConversionRefuse {
			return decision(OutcomeRefuse, ReasonConversionRefused, PredicateConversion, "policy conversion rule is refuse", "Use an existing approved artifact; do not convert it under this policy.", claims)
		}
	default:
		return decision(OutcomeDelegate, ReasonOperationNotHandled, PredicateOperation, string(request.Operation), "Delegate to a policy evaluator that explicitly supports this operation.", claims)
	}

	if strings.TrimSpace(request.Metadata.ArtifactID) == "" {
		return decision(OutcomeAbstain, ReasonUnknownMetadata, PredicateMetadata, "artifact_id is empty", "Provide a stable artifact identity before evaluation.", claims)
	}
	precision := request.Metadata.Precision
	if strings.TrimSpace(precision.Name) == "" || !finitePositive(precision.Bits) {
		return decision(OutcomeAbstain, ReasonUnknownPrecision, PredicateMetadata, "precision name and positive finite bits are required", "Provide explicit precision name and bit width; no precision is inferred.", claims)
	}
	if precision.Bits < policy.MinPrecisionBits {
		return decision(OutcomeRefuse, ReasonBelowMinimumPrecision, PredicateMinimumPrecision, fmt.Sprintf("%g < %g", precision.Bits, policy.MinPrecisionBits), "Use an artifact at or above the policy minimum precision.", claims)
	}
	if precision.Bits > policy.MaxPrecisionBits {
		return decision(OutcomeRefuse, ReasonAboveMaximumPrecision, PredicateMaximumPrecision, fmt.Sprintf("%g > %g", precision.Bits, policy.MaxPrecisionBits), "Use an artifact at or below the policy maximum precision.", claims)
	}
	if strings.TrimSpace(request.Metadata.Format.ID) == "" || strings.TrimSpace(request.Metadata.Format.Version) == "" {
		return decision(OutcomeAbstain, ReasonUnknownFormat, PredicateApprovedFormat, "format id and version are required", "Provide an explicit public format ID and version; no format is inferred.", claims)
	}
	if !approvedFormat(request.Metadata.Format, policy.ApprovedFormats) {
		return decision(OutcomeRefuse, ReasonFormatNotApproved, PredicateApprovedFormat, request.Metadata.Format.ID+"@"+request.Metadata.Format.Version, "Use one of the policy's exact approved format/version pairs.", claims)
	}

	checks := []struct {
		predicate   PredicateID
		evidence    Evidence
		requirement ProvenanceRequirement
	}{
		{PredicateArtifactProvenance, claims.Artifact, policy.Provenance.Artifact},
		{PredicateRecipeProvenance, claims.Recipe, policy.Provenance.Recipe},
		{PredicateRuntimeProvenance, claims.RuntimeDelegation, policy.Provenance.RuntimeDelegation},
		{PredicateHardwareProvenance, claims.HardwareEnvelope, policy.Provenance.HardwareEnvelope},
	}
	for _, check := range checks {
		if got, stop := evaluateEvidence(check.predicate, check.evidence, check.requirement, claims); stop {
			return got
		}
	}

	return decision(OutcomeAllow, ReasonSatisfied, PredicateAll, "all declared predicates matched", "Use the existing artifact only within the separately stated artifact, recipe, runtime-delegation, and measured-hardware claims.", claims)
}

func evaluateEvidence(predicate PredicateID, evidence Evidence, requirement ProvenanceRequirement, claims ClaimEnvelope) (Result, bool) {
	kindBlank := strings.TrimSpace(string(evidence.Kind)) == ""
	referenceBlank := strings.TrimSpace(evidence.Reference) == ""
	if kindBlank && referenceBlank {
		if requirement.Required {
			return decision(OutcomeRefuse, ReasonProvenanceRequired, predicate, "required provenance is absent", "Attach an independently readable provenance reference with an approved evidence kind.", claims), true
		}
		return Result{}, false
	}
	if kindBlank || referenceBlank || !knownEvidenceKind(evidence.Kind) {
		return decision(OutcomeAbstain, ReasonUnknownProvenance, predicate, string(evidence.Kind), "Provide both a reference and a recognized provenance evidence kind; no provenance is inferred.", claims), true
	}
	if len(requirement.ApprovedKinds) > 0 && !containsEvidenceKind(requirement.ApprovedKinds, evidence.Kind) {
		return decision(OutcomeRefuse, ReasonProvenanceNotApproved, predicate, string(evidence.Kind), "Supply provenance with an evidence kind approved for this claim.", claims), true
	}
	return Result{}, false
}

func validateContract(policy Policy) string {
	if !finitePositive(policy.MinPrecisionBits) || !finitePositive(policy.MaxPrecisionBits) || policy.MinPrecisionBits > policy.MaxPrecisionBits {
		return "precision bounds must be positive finite numbers with minimum <= maximum"
	}
	if len(policy.ApprovedFormats) == 0 {
		return "approved_formats is empty"
	}
	seenFormats := make(map[string]bool, len(policy.ApprovedFormats))
	for i, format := range policy.ApprovedFormats {
		if strings.TrimSpace(format.ID) == "" || strings.TrimSpace(format.Version) == "" {
			return fmt.Sprintf("approved_formats[%d] requires id and version", i)
		}
		key := format.ID + "\x00" + format.Version
		if seenFormats[key] {
			return fmt.Sprintf("approved_formats[%d] duplicates %s@%s", i, format.ID, format.Version)
		}
		seenFormats[key] = true
	}
	if policy.Conversion != ConversionAllow && policy.Conversion != ConversionRefuse {
		return "conversion must be allow or refuse"
	}
	requirements := []struct {
		name string
		req  ProvenanceRequirement
	}{
		{"artifact", policy.Provenance.Artifact},
		{"recipe", policy.Provenance.Recipe},
		{"runtime_delegation", policy.Provenance.RuntimeDelegation},
		{"hardware_envelope", policy.Provenance.HardwareEnvelope},
	}
	for _, item := range requirements {
		if item.req.Required && len(item.req.ApprovedKinds) == 0 {
			return "provenance." + item.name + " is required but has no approved kinds"
		}
		seen := map[EvidenceKind]bool{}
		for _, kind := range item.req.ApprovedKinds {
			if !knownEvidenceKind(kind) {
				return "provenance." + item.name + " contains unknown evidence kind " + string(kind)
			}
			if seen[kind] {
				return "provenance." + item.name + " contains duplicate evidence kind " + string(kind)
			}
			seen[kind] = true
		}
	}
	return ""
}

func decodeStrict(raw []byte, dst any) error {
	return strictjson.Decode(raw, dst, "multiple JSON values")
}

func decision(outcome Outcome, reason ReasonCode, predicate PredicateID, detail, action string, claims ClaimEnvelope) Result {
	return Result{Outcome: outcome, Reason: reason, Predicate: predicate, Detail: detail, Action: action, Claims: claims}
}

func approvedFormat(candidate FormatRef, approved []FormatRef) bool {
	for _, format := range approved {
		if candidate == format {
			return true
		}
	}
	return false
}

func finitePositive(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func knownEvidenceKind(kind EvidenceKind) bool {
	switch kind {
	case EvidenceReported, EvidenceModeled, EvidenceObserved, EvidenceMeasured, EvidenceWitnessed:
		return true
	default:
		return false
	}
}

func containsEvidenceKind(all []EvidenceKind, want EvidenceKind) bool {
	for _, kind := range all {
		if kind == want {
			return true
		}
	}
	return false
}
