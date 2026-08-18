// Package quantcompat adjudicates whether a declared quantized artifact can run
// on a declared runtime and hardware envelope without an implicit conversion.
package quantcompat

import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

// Decision is the closed public compatibility vocabulary.
type Status string

const (
	StatusDirect             Status = "supported"
	StatusExternalRuntime    Status = "delegated"
	StatusConversionRequired Status = "conversion-required"
	StatusRejected           Status = "unsupported"
)

// Reason is a stable machine-readable explanation for a decision.
type Reason string

const (
	ReasonCompatible          Reason = "quantcompat.compatible"
	ReasonExternalRuntime     Reason = "quantcompat.runtime-delegated"
	ReasonConversionAvailable Reason = "quantcompat.conversion-available"
	ReasonArtifactInvalid     Reason = "quantcompat.artifact-invalid"
	ReasonFormatUnknown       Reason = "quantcompat.format-unknown"
	ReasonRuntimeUnknown      Reason = "quantcompat.runtime-unknown"
	ReasonHardwareUnknown     Reason = "quantcompat.hardware-unknown"
	ReasonFormatRejected      Reason = "quantcompat.format-unsupported"
	ReasonMethodRejected      Reason = "quantcompat.method-unsupported"
	ReasonHardwareRejected    Reason = "quantcompat.hardware-unsupported"
)

// Runtime declares one runtime's directly executable and delegated envelope.
type Runtime struct {
	ID                 string
	Formats            []quantmeta.Format
	Methods            []string
	Hardware           []string
	ExternalFormats    []quantmeta.Format
	ConvertibleFormats []quantmeta.Format
}

// Request binds the artifact, runtime, and hardware descriptors for one decision.
type Request struct {
	Artifact quantmeta.Descriptor
	Runtime  Runtime
	Hardware string
}

// Result is intentionally self-describing so callers never infer fallback behavior.
type Result struct {
	Status Status `json:"status"`
	Reason Reason `json:"reason"`
}

// Adjudicate returns one typed result. Conversion and delegation are never reported
// as direct support, and unknown declarations fail closed.
func Adjudicate(req Request) Result {
	artifact := quantmeta.Adjudicate(req.Artifact)
	switch artifact.Outcome {
	case quantmeta.OutcomeRefuse:
		return result(StatusRejected, ReasonArtifactInvalid)
	case quantmeta.OutcomeAbstain:
		return result(StatusRejected, ReasonFormatUnknown)
	}

	if strings.TrimSpace(req.Runtime.ID) == "" {
		return result(StatusRejected, ReasonRuntimeUnknown)
	}
	if strings.TrimSpace(req.Hardware) == "" {
		return result(StatusRejected, ReasonHardwareUnknown)
	}
	if !containsString(req.Runtime.Hardware, req.Hardware) {
		return result(StatusRejected, ReasonHardwareRejected)
	}

	format := quantmeta.Format(req.Artifact.Artifact.ContainerID)
	if containsFormat(req.Runtime.Formats, format) {
		if !containsMethod(req.Runtime.Methods, req.Artifact.Provenance.MethodID) {
			return result(StatusRejected, ReasonMethodRejected)
		}
		return result(StatusDirect, ReasonCompatible)
	}
	if containsFormat(req.Runtime.ExternalFormats, format) {
		return result(StatusExternalRuntime, ReasonExternalRuntime)
	}
	if containsFormat(req.Runtime.ConvertibleFormats, format) {
		return result(StatusConversionRequired, ReasonConversionAvailable)
	}
	return result(StatusRejected, ReasonFormatRejected)
}

func result(status Status, reason Reason) Result {
	return Result{Status: status, Reason: reason}
}

func containsFormat(values []quantmeta.Format, want quantmeta.Format) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func containsMethod(values []string, want string) bool {
	return containsString(values, want)
}

func containsString(values []string, want string) bool {
	want = strings.ToLower(strings.TrimSpace(want))
	values = append([]string(nil), values...)
	sort.Strings(values)
	for _, value := range values {
		if strings.ToLower(strings.TrimSpace(value)) == want {
			return true
		}
	}
	return false
}
