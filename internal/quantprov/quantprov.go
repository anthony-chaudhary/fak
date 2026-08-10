package quantprov

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ContractVersion is the only provenance schema this leaf currently adjudicates.
const ContractVersion = "quantprov/v1"

// Confidence is the explicit trust state of an adjudicated provenance record.
type Confidence string

const (
	ConfidenceConfirmed   Confidence = "confirmed"
	ConfidenceIncomplete  Confidence = "incomplete"
	ConfidenceUnsupported Confidence = "unsupported"
	ConfidenceTampered    Confidence = "tampered"
	ConfidenceInvalid     Confidence = "invalid"
)

// ReasonCode is stable machine-readable detail for a provenance decision.
type ReasonCode string

const (
	ReasonConfirmed            ReasonCode = "QUANTPROV_CONFIRMED"
	ReasonInvalidJSON          ReasonCode = "QUANTPROV_INVALID_JSON"
	ReasonUnknownSchema        ReasonCode = "QUANTPROV_UNKNOWN_SCHEMA"
	ReasonMissingField         ReasonCode = "QUANTPROV_MISSING_FIELD"
	ReasonUnsupportedQuantizer ReasonCode = "QUANTPROV_UNSUPPORTED_QUANTIZER"
	ReasonUnsupportedVersion   ReasonCode = "QUANTPROV_UNSUPPORTED_VERSION"
	ReasonUnsupportedFormat    ReasonCode = "QUANTPROV_UNSUPPORTED_FORMAT"
	ReasonBrokenChain          ReasonCode = "QUANTPROV_BROKEN_CONVERSION_CHAIN"
)

// ConversionStep records one reproducible transformation in artifact lineage.
type ConversionStep struct {
	Tool         string `json:"tool"`
	ToolVersion  string `json:"tool_version"`
	InputDigest  string `json:"input_digest"`
	OutputDigest string `json:"output_digest"`
}

// Record is a neutral description of a quantized artifact. It deliberately
// describes provenance rather than asserting quality or runtime support.
type Record struct {
	Schema              string            `json:"schema"`
	SourceModel         string            `json:"source_model"`
	SourceRevision      string            `json:"source_revision"`
	SourceDigest        string            `json:"source_digest"`
	ArtifactDigest      string            `json:"artifact_digest"`
	Quantizer           string            `json:"quantizer"`
	QuantizerVersion    string            `json:"quantizer_version"`
	CalibrationIdentity string            `json:"calibration_identity"`
	Format              string            `json:"format"`
	Parameters          map[string]string `json:"parameters"`
	License             string            `json:"license"`
	ConversionChain     []ConversionStep  `json:"conversion_chain"`
}

// Support is the caller's explicit interoperability envelope. Missing entries
// are unsupported, never silently accepted.
type Support struct {
	Quantizers map[string][]string
	Formats    []string
}

// Result is the typed outcome returned for every parse/adjudication path.
type Result struct {
	Confidence Confidence `json:"confidence"`
	Reason     ReasonCode `json:"reason"`
	Detail     string     `json:"detail,omitempty"`
	Record     *Record    `json:"record,omitempty"`
}

// ParseAndVerify parses a record and adjudicates it against support. The
// returned Result is authoritative even when err is non-nil; err is supplied
// only for callers that use Go's conventional malformed-input flow.
func ParseAndVerify(raw []byte, support Support) (Result, error) {
	var record Record
	if err := json.Unmarshal(raw, &record); err != nil {
		result := Result{Confidence: ConfidenceInvalid, Reason: ReasonInvalidJSON, Detail: err.Error()}
		return result, fmt.Errorf("parse quantization provenance: %w", err)
	}
	return Verify(record, support), nil
}

// Verify adjudicates required fields, declared support, and digest continuity.
func Verify(record Record, support Support) Result {
	result := Result{Record: &record}
	if record.Schema != ContractVersion {
		result.Confidence, result.Reason = ConfidenceUnsupported, ReasonUnknownSchema
		result.Detail = "schema " + record.Schema
		return result
	}
	if field := firstMissing(record); field != "" {
		result.Confidence, result.Reason = ConfidenceIncomplete, ReasonMissingField
		result.Detail = field
		return result
	}
	versions, ok := support.Quantizers[record.Quantizer]
	if !ok {
		result.Confidence, result.Reason = ConfidenceUnsupported, ReasonUnsupportedQuantizer
		result.Detail = record.Quantizer
		return result
	}
	if !contains(versions, record.QuantizerVersion) {
		result.Confidence, result.Reason = ConfidenceUnsupported, ReasonUnsupportedVersion
		result.Detail = record.Quantizer + "@" + record.QuantizerVersion
		return result
	}
	if !contains(support.Formats, record.Format) {
		result.Confidence, result.Reason = ConfidenceUnsupported, ReasonUnsupportedFormat
		result.Detail = record.Format
		return result
	}
	wantInput := record.SourceDigest
	for i, step := range record.ConversionChain {
		if step.InputDigest != wantInput || step.OutputDigest == "" {
			result.Confidence, result.Reason = ConfidenceTampered, ReasonBrokenChain
			result.Detail = fmt.Sprintf("step %d digest continuity", i)
			return result
		}
		wantInput = step.OutputDigest
	}
	if wantInput != record.ArtifactDigest {
		result.Confidence, result.Reason = ConfidenceTampered, ReasonBrokenChain
		result.Detail = "artifact digest does not terminate conversion chain"
		return result
	}
	result.Confidence, result.Reason = ConfidenceConfirmed, ReasonConfirmed
	return result
}

func firstMissing(r Record) string {
	fields := []struct{ name, value string }{
		{"source_model", r.SourceModel}, {"source_revision", r.SourceRevision},
		{"source_digest", r.SourceDigest}, {"artifact_digest", r.ArtifactDigest},
		{"quantizer", r.Quantizer}, {"quantizer_version", r.QuantizerVersion},
		{"calibration_identity", r.CalibrationIdentity}, {"format", r.Format},
		{"license", r.License},
	}
	for _, field := range fields {
		if strings.TrimSpace(field.value) == "" {
			return field.name
		}
	}
	if len(r.Parameters) == 0 {
		return "parameters"
	}
	if len(r.ConversionChain) == 0 {
		return "conversion_chain"
	}
	for i, step := range r.ConversionChain {
		if strings.TrimSpace(step.Tool) == "" {
			return fmt.Sprintf("conversion_chain[%d].tool", i)
		}
		if strings.TrimSpace(step.ToolVersion) == "" {
			return fmt.Sprintf("conversion_chain[%d].tool_version", i)
		}
		if strings.TrimSpace(step.InputDigest) == "" {
			return fmt.Sprintf("conversion_chain[%d].input_digest", i)
		}
		if strings.TrimSpace(step.OutputDigest) == "" {
			return fmt.Sprintf("conversion_chain[%d].output_digest", i)
		}
	}
	return ""
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
