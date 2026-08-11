// Package quantlicense evaluates evidence supplied for a quantized-model license chain.
//
// It does not interpret licenses or provide legal advice. Callers provide explicit,
// independently sourced permission and compatibility attestations; the package only
// checks that the evidence chain covers the requested operation without a denied or
// unknown term.
package quantlicense

import (
	"encoding/json"
	"fmt"
	"strings"
)

const SchemaV1 = "quantlicense/v1"

// Outcome is the stable disposition returned by Evaluate.
type Outcome string

const (
	OutcomeAllow    Outcome = "allow"
	OutcomeRefuse   Outcome = "refuse"
	OutcomeAbstain  Outcome = "abstain"
	OutcomeDelegate Outcome = "delegate"
)

// ReasonCode is a stable, machine-readable explanation for an outcome.
type ReasonCode string

const (
	ReasonCompatible                   ReasonCode = "compatible"
	ReasonInvalidJSON                  ReasonCode = "invalid_json"
	ReasonUnknownSchema                ReasonCode = "unknown_schema"
	ReasonUseOutsideContract           ReasonCode = "use_outside_contract"
	ReasonMissingSourceLicense         ReasonCode = "missing_source_license"
	ReasonMissingArtifactLicense       ReasonCode = "missing_artifact_license"
	ReasonMissingFormatLicense         ReasonCode = "missing_format_license"
	ReasonMissingQuantizerLicense      ReasonCode = "missing_quantizer_license"
	ReasonMissingRuntimeLicense        ReasonCode = "missing_runtime_license"
	ReasonMissingCompatibility         ReasonCode = "missing_compatibility_attestation"
	ReasonIncompatibleChain            ReasonCode = "incompatible_chain"
	ReasonUnknownLicense               ReasonCode = "unknown_license_evidence"
	ReasonSourceUseDenied              ReasonCode = "source_use_denied"
	ReasonSourceDerivationDenied       ReasonCode = "source_derivation_denied"
	ReasonSourceRedistributionDenied   ReasonCode = "source_redistribution_denied"
	ReasonArtifactUseDenied            ReasonCode = "artifact_use_denied"
	ReasonArtifactRedistributionDenied ReasonCode = "artifact_redistribution_denied"
	ReasonFormatUseDenied              ReasonCode = "format_use_denied"
	ReasonQuantizerUseDenied           ReasonCode = "quantizer_use_denied"
	ReasonRuntimeUseDenied             ReasonCode = "runtime_use_denied"
	ReasonMissingClaims                ReasonCode = "missing_claim_envelope"
)

// Permission is an evidence assertion, not fak's interpretation of license text.
type Permission string

const (
	PermissionAllowed Permission = "allowed"
	PermissionDenied  Permission = "denied"
	PermissionUnknown Permission = "unknown"
)

// Compatibility is an externally attested relationship between source and artifact terms.
type Compatibility string

const (
	CompatibilityCompatible   Compatibility = "compatible"
	CompatibilityIncompatible Compatibility = "incompatible"
	CompatibilityUnknown      Compatibility = "unknown"
)

// UseKind identifies operations understood by this contract version.
type UseKind string

const (
	UseDemo     UseKind = "demo"
	UseEvaluate UseKind = "evaluate"
	UseServe    UseKind = "serve"
)

// Terms records asserted permissions separately. Empty values are unknown.
type Terms struct {
	Use          Permission `json:"use"`
	Derive       Permission `json:"derive"`
	Redistribute Permission `json:"redistribute"`
}

// LicenseEvidence identifies terms and where a caller can independently inspect them.
type LicenseEvidence struct {
	ID       string `json:"id"`
	Evidence string `json:"evidence"`
	Terms    Terms  `json:"terms"`
}

// SourceWeights identifies the original weights separately from every derived artifact.
type SourceWeights struct {
	Identity string          `json:"identity"`
	License  LicenseEvidence `json:"license"`
}

// Component identifies a format, quantizer, or runtime and its own license evidence.
type Component struct {
	Name    string          `json:"name"`
	Version string          `json:"version"`
	License LicenseEvidence `json:"license"`
}

// Artifact identifies derived bytes, their artifact-specific terms, and public format.
type Artifact struct {
	Identity string          `json:"identity"`
	License  LicenseEvidence `json:"license"`
	Format   Component       `json:"format"`
}

// Recipe describes how an artifact was derived and the quantizer whose terms apply.
type Recipe struct {
	ID        string    `json:"id"`
	Quantizer Component `json:"quantizer"`
}

// ChainAttestation is caller-supplied evidence that source and artifact terms coexist.
type ChainAttestation struct {
	Status   Compatibility `json:"status"`
	Evidence string        `json:"evidence"`
}

// Request states the operation being gated.
type Request struct {
	Use                  UseKind `json:"use"`
	Convert              bool    `json:"convert"`
	RedistributeSource   bool    `json:"redistribute_source"`
	RedistributeArtifact bool    `json:"redistribute_artifact"`
}

// ClaimEnvelope prevents a license pass from implying unwitnessed artifact, quality,
// runtime, or hardware claims.
type ClaimEnvelope struct {
	Artifact          string `json:"artifact"`
	Recipe            string `json:"recipe"`
	RuntimeDelegation string `json:"runtime_delegation"`
	HardwareEnvelope  string `json:"hardware_envelope"`
}

// Manifest is the public quantlicense/v1 artifact.
type Manifest struct {
	Schema        string           `json:"schema"`
	SourceWeights SourceWeights    `json:"source_weights"`
	Artifact      Artifact         `json:"artifact"`
	Recipe        Recipe           `json:"recipe"`
	Runtime       Component        `json:"runtime"`
	Compatibility ChainAttestation `json:"compatibility"`
	Request       Request          `json:"request"`
	Claims        ClaimEnvelope    `json:"claims"`
}

// Result is always typed and includes a next action plus a non-legal-advice boundary.
type Result struct {
	Outcome        Outcome    `json:"outcome"`
	Reason         ReasonCode `json:"reason"`
	Action         string     `json:"action"`
	NonLegalAdvice string     `json:"non_legal_advice"`
}

const disclaimer = "This result checks supplied evidence fields; it is not legal advice or a license interpretation."

func result(outcome Outcome, reason ReasonCode, action string) Result {
	return Result{Outcome: outcome, Reason: reason, Action: action, NonLegalAdvice: disclaimer}
}

// ParseAndEvaluate decodes a public manifest and preserves malformed input as a typed result.
func ParseAndEvaluate(raw []byte) (Result, error) {
	var m Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return result(OutcomeAbstain, ReasonInvalidJSON, "Provide a valid quantlicense JSON manifest before evaluation."), fmt.Errorf("parse quantlicense manifest: %w", err)
	}
	return Evaluate(m), nil
}

// Evaluate checks the complete requested chain. It never guesses missing permissions,
// license compatibility, runtime support, quality, or performance.
func Evaluate(m Manifest) Result {
	if m.Schema != SchemaV1 {
		return result(OutcomeAbstain, ReasonUnknownSchema, "Provide a quantlicense/v1 manifest or delegate evaluation to a parser that supports this schema.")
	}
	switch m.Request.Use {
	case UseDemo, UseEvaluate, UseServe:
	default:
		return result(OutcomeDelegate, ReasonUseOutsideContract, "Delegate this operation to a policy evaluator that explicitly supports the requested use.")
	}

	if missing(m.SourceWeights.License) {
		return result(OutcomeRefuse, ReasonMissingSourceLicense, "Attach independently readable source-weight license evidence and explicit terms.")
	}
	if missing(m.Artifact.License) {
		return result(OutcomeRefuse, ReasonMissingArtifactLicense, "Attach independently readable derived-artifact license evidence and explicit terms.")
	}
	if missing(m.Artifact.Format.License) {
		return result(OutcomeRefuse, ReasonMissingFormatLicense, "Attach independently readable format implementation/specification license evidence.")
	}
	if missing(m.Recipe.Quantizer.License) {
		return result(OutcomeRefuse, ReasonMissingQuantizerLicense, "Attach independently readable quantizer license evidence.")
	}
	if missing(m.Runtime.License) {
		return result(OutcomeRefuse, ReasonMissingRuntimeLicense, "Attach independently readable delegated-runtime license evidence.")
	}
	if blankClaims(m.Claims) {
		return result(OutcomeRefuse, ReasonMissingClaims, "State artifact, recipe, runtime delegation, and measured hardware envelope claims separately.")
	}
	if m.Compatibility.Status == CompatibilityIncompatible {
		return result(OutcomeRefuse, ReasonIncompatibleChain, "Do not use this source/artifact chain; attach evidence for a compatible artifact or choose another artifact.")
	}
	if m.Compatibility.Status != CompatibilityCompatible || strings.TrimSpace(m.Compatibility.Evidence) == "" {
		return result(OutcomeAbstain, ReasonMissingCompatibility, "Obtain an independent source-to-artifact compatibility attestation before using the chain.")
	}

	checks := []struct {
		needed bool
		value  Permission
		denied ReasonCode
		action string
	}{
		{true, m.SourceWeights.License.Terms.Use, ReasonSourceUseDenied, "Use source weights whose terms explicitly allow the requested operation."},
		{true, m.Artifact.License.Terms.Use, ReasonArtifactUseDenied, "Use a derived artifact whose terms explicitly allow the requested operation."},
		{true, m.Artifact.Format.License.Terms.Use, ReasonFormatUseDenied, "Use a format whose supplied terms explicitly allow this operation."},
		{true, m.Runtime.License.Terms.Use, ReasonRuntimeUseDenied, "Delegate to a runtime whose supplied terms explicitly allow this operation."},
		{m.Request.Convert, m.SourceWeights.License.Terms.Derive, ReasonSourceDerivationDenied, "Do not convert these weights; obtain derivation permission or choose another source."},
		{m.Request.Convert, m.Recipe.Quantizer.License.Terms.Use, ReasonQuantizerUseDenied, "Use a quantizer whose supplied terms explicitly allow conversion."},
		{m.Request.RedistributeSource, m.SourceWeights.License.Terms.Redistribute, ReasonSourceRedistributionDenied, "Do not redistribute the source weights; obtain explicit permission or omit redistribution."},
		{m.Request.RedistributeArtifact, m.Artifact.License.Terms.Redistribute, ReasonArtifactRedistributionDenied, "Do not redistribute the derived artifact; obtain explicit permission or omit redistribution."},
	}
	for _, check := range checks {
		if !check.needed {
			continue
		}
		switch check.value {
		case PermissionDenied:
			return result(OutcomeRefuse, check.denied, check.action)
		case PermissionAllowed:
		default:
			return result(OutcomeAbstain, ReasonUnknownLicense, "Obtain an explicit allowed/denied permission assertion for every term required by the request.")
		}
	}
	return result(OutcomeAllow, ReasonCompatible, "Use only for the stated request and preserve the separate evidence and claim envelope.")
}

func missing(l LicenseEvidence) bool {
	return strings.TrimSpace(l.ID) == "" || strings.TrimSpace(l.Evidence) == ""
}

func blankClaims(c ClaimEnvelope) bool {
	return strings.TrimSpace(c.Artifact) == "" || strings.TrimSpace(c.Recipe) == "" || strings.TrimSpace(c.RuntimeDelegation) == "" || strings.TrimSpace(c.HardwareEnvelope) == ""
}
