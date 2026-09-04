// Package microscaleeval adjudicates microscaling descriptors without inferring
// runtime support or turning research results into locally observed claims.
package microscaleeval

import (
	"fmt"
	"strings"
)

const (
	SchemaV1 = "fak.microscale-eval/v1"

	Native  Disposition = "supported"
	Refuse  Disposition = "unsupported"
	Forward Disposition = "delegate"

	ReasonNativeMatch          Reason = "native_profile_match"
	ReasonUnknownSchema        Reason = "unknown_schema"
	ReasonInvalidProvenance    Reason = "invalid_provenance"
	ReasonInvalidDescriptor    Reason = "invalid_descriptor"
	ReasonUnsupportedFormat    Reason = "unsupported_format"
	ReasonRuntimeMismatch      Reason = "runtime_profile_mismatch"
	ReasonHeterogeneousRuntime Reason = "heterogeneous_runtime_required"

	EvidenceModeled  EvidenceKind = "modeled"
	EvidenceObserved EvidenceKind = "observed"
)

type Disposition string
type Reason string
type EvidenceKind string

// Pin identifies immutable input. Revision and SHA256 are both required: a
// friendly model/artifact name alone is not provenance.
type Pin struct {
	ID       string `json:"id"`
	Revision string `json:"revision"`
	SHA256   string `json:"sha256"`
}

// Provenance keeps the serialised artifact, quantization recipe, execution
// runtime, and evaluated model independently attributable.
type Provenance struct {
	Artifact Pin `json:"artifact"`
	Recipe   Pin `json:"recipe"`
	Runtime  Pin `json:"runtime"`
	Model    Pin `json:"model"`
}

// Operand describes one encoded operand. Recovery is the per-block precision
// recovery scheme, not an implied property of ElementFormat.
type Operand struct {
	ElementFormat string `json:"element_format"`
	Recovery      string `json:"recovery"`
}

// Descriptor is the neutral, non-fak artifact view needed to compare OCP MX,
// vendor block scaling, and heterogeneity-aware research formats.
type Descriptor struct {
	Schema       string  `json:"schema"`
	Family       string  `json:"family"`
	BlockSize    int     `json:"block_size"`
	ScaleFormat  string  `json:"scale_format"`
	Weights      Operand `json:"weights"`
	Activations  Operand `json:"activations"`
	PerBlockMode bool    `json:"per_block_mode"`
}

// NativeProfile is an exact runtime capability; empty fields are never
// wildcards. MixedOperands and PerBlockMode must be explicitly advertised.
type NativeProfile struct {
	Family        string  `json:"family"`
	BlockSize     int     `json:"block_size"`
	ScaleFormat   string  `json:"scale_format"`
	Weights       Operand `json:"weights"`
	Activations   Operand `json:"activations"`
	MixedOperands bool    `json:"mixed_operands"`
	PerBlockMode  bool    `json:"per_block_mode"`
}

type RuntimeCapabilities struct {
	Profiles []NativeProfile `json:"profiles"`
	Delegate string          `json:"delegate,omitempty"`
}

// Evidence labels quality/performance data by origin. Observed data requires
// a machine-readable run witness and hardware identity; modeled data requires
// a cited model/source. Evaluate never fabricates either kind.
type Evidence struct {
	Kind                EvidenceKind `json:"kind"`
	Source              string       `json:"source"`
	RunSHA256           string       `json:"run_sha256,omitempty"`
	HardwareFingerprint string       `json:"hardware_fingerprint,omitempty"`
}

type Request struct {
	Descriptor   Descriptor          `json:"descriptor"`
	Capabilities RuntimeCapabilities `json:"capabilities"`
	Provenance   Provenance          `json:"provenance"`
	Evidence     Evidence            `json:"evidence"`
}

type Verdict struct {
	Outcome    Disposition `json:"outcome"`
	Reason     Reason      `json:"reason"`
	Detail     string      `json:"detail"`
	Provenance Provenance  `json:"provenance"`
	Evidence   Evidence    `json:"evidence"`
	Delegate   string      `json:"delegate,omitempty"`
}

// Evaluate returns a closed typed outcome. It does not silently convert a
// descriptor, select a universal format, or promote paper-reported data to an
// observation.
//
// Invariant: microscale evaluations are fail-closed and deterministic.
// Guard: unknown schema versions, missing provenance digests, or unverified
// observed evidence immediately refuse evaluation rather than inferring runtime support.
func Evaluate(r Request) Verdict {
	v := Verdict{Provenance: r.Provenance, Evidence: r.Evidence}
	if r.Descriptor.Schema != SchemaV1 {
		return v.finish(Refuse, ReasonUnknownSchema, "descriptor schema is not supported")
	}
	if err := validateProvenance(r.Provenance); err != nil {
		return v.finish(Refuse, ReasonInvalidProvenance, err.Error())
	}
	if err := validateEvidence(r.Evidence); err != nil {
		return v.finish(Refuse, ReasonInvalidProvenance, err.Error())
	}
	if err := validateDescriptor(r.Descriptor); err != nil {
		return v.finish(Refuse, ReasonInvalidDescriptor, err.Error())
	}

	known := r.Descriptor.Family == "ocp-mx-v1" || r.Descriptor.Family == "nvidia-nvfp4" || r.Descriptor.Family == "adamx-paper-v1"
	if !known {
		return v.finish(Refuse, ReasonUnsupportedFormat, "format family is not in the bounded evaluation set")
	}
	for _, p := range r.Capabilities.Profiles {
		if exactMatch(r.Descriptor, p) {
			return v.finish(Native, ReasonNativeMatch, "descriptor exactly matches a pinned native runtime profile")
		}
	}

	if r.Capabilities.Delegate != "" {
		v.Delegate = r.Capabilities.Delegate
		if r.Descriptor.PerBlockMode || operandsDiffer(r.Descriptor.Weights, r.Descriptor.Activations) {
			return v.finish(Forward, ReasonHeterogeneousRuntime, "known descriptor requires per-block or per-operand behavior absent from native profiles")
		}
		return v.finish(Forward, ReasonRuntimeMismatch, "known descriptor has no exact native runtime profile")
	}
	return v.finish(Refuse, ReasonRuntimeMismatch, "known descriptor has no exact native runtime profile or declared delegate")
}

func (v Verdict) finish(o Disposition, r Reason, detail string) Verdict {
	v.Outcome, v.Reason, v.Detail = o, r, detail
	return v
}

func validateProvenance(p Provenance) error {
	for name, pin := range map[string]Pin{"artifact": p.Artifact, "recipe": p.Recipe, "runtime": p.Runtime, "model": p.Model} {
		if strings.TrimSpace(pin.ID) == "" || strings.TrimSpace(pin.Revision) == "" || !isSHA256(pin.SHA256) {
			return fmt.Errorf("%s provenance requires id, revision, and lowercase sha256", name)
		}
	}
	return nil
}

func validateEvidence(e Evidence) error {
	switch e.Kind {
	case EvidenceModeled:
		if strings.TrimSpace(e.Source) == "" || e.RunSHA256 != "" || e.HardwareFingerprint != "" {
			return fmt.Errorf("modeled evidence requires source and must not impersonate an observed witness")
		}
	case EvidenceObserved:
		if strings.TrimSpace(e.Source) == "" || !isSHA256(e.RunSHA256) || strings.TrimSpace(e.HardwareFingerprint) == "" {
			return fmt.Errorf("observed evidence requires source, witness sha256, and hardware fingerprint")
		}
	default:
		return fmt.Errorf("evidence kind must be modeled or observed")
	}
	return nil
}

func validateDescriptor(d Descriptor) error {
	if d.Family == "" || d.BlockSize <= 0 || d.ScaleFormat == "" {
		return fmt.Errorf("family, positive block size, and scale format are required")
	}
	for name, op := range map[string]Operand{"weights": d.Weights, "activations": d.Activations} {
		if op.ElementFormat == "" || op.Recovery == "" {
			return fmt.Errorf("%s element format and recovery are required", name)
		}
	}
	return nil
}

func exactMatch(d Descriptor, p NativeProfile) bool {
	if d.Family != p.Family || d.BlockSize != p.BlockSize || d.ScaleFormat != p.ScaleFormat || d.PerBlockMode != p.PerBlockMode {
		return false
	}
	if d.Weights != p.Weights || d.Activations != p.Activations {
		return false
	}
	return p.MixedOperands || !operandsDiffer(d.Weights, d.Activations)
}

func operandsDiffer(a, b Operand) bool { return a != b }

func isSHA256(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
