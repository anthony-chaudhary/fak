package benchauthority

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ParserCompletenessSchema is the canonical schema version tag for parser-completeness
// receipts. Every sidecar artifact must carry this exact schema identifier.
const ParserCompletenessSchema = "fak.benchmark-parser-completeness/v1"

// Closed reason tokens for claim admission evaluation.
const (
	ReasonParserCompletenessDeficit = "PARSER_COMPLETENESS_DEFICIT"
	ReasonMalformedRecordBreach     = "MALFORMED_RECORD_BREACH"
	ReasonUnknownFamilyBreach       = "UNKNOWN_FAMILY_BREACH"
	ReasonTruncatedRecordBreach     = "TRUNCATED_RECORD_BREACH"
	ReasonDuplicateRecordBreach     = "DUPLICATE_RECORD_BREACH"
	ReasonEmptyInput                = "EMPTY_INPUT"
	ReasonAccountingIncongruent     = "ACCOUNTING_INCONGRUENT"
	ReasonInvalidReceipt            = "INVALID_RECEIPT"
)

// Standard error sentinels returned by receipt validation.
var (
	// ErrAccountingIncongruent is returned when TotalRecords != sum of all category records.
	ErrAccountingIncongruent = errors.New("parser completeness accounting is incongruent: total records does not equal sum of parts")
	// ErrInvalidSchema is returned when the receipt carries an unexpected or missing schema.
	ErrInvalidSchema = errors.New("invalid parser completeness receipt schema")
	// ErrNegativeCount is returned when any record count or byte count is negative.
	ErrNegativeCount = errors.New("record counts and byte counts cannot be negative")
	// ErrMissingSource is returned when source identity (artifact/path or SHA256) is missing.
	ErrMissingSource = errors.New("missing source identity: artifact/path and sha256 are required")
)

// SourceIdentity captures provenance identity for the raw benchmark artifact.
type SourceIdentity struct {
	// Path is the file path of the source artifact.
	Path string `json:"path,omitempty"`
	// Artifact is an alternate name/identifier for the source artifact.
	Artifact string `json:"artifact,omitempty"`
	// SHA256 is the hex-encoded SHA-256 digest of the raw source file.
	SHA256 string `json:"sha256"`
}

// Location returns the non-empty artifact name or file path.
func (s SourceIdentity) Location() string {
	if s.Artifact != "" {
		return s.Artifact
	}
	return s.Path
}

// RecordSample carries bounded, scrubbed evidence for an anomalous or unparsed record.
type RecordSample struct {
	// Line is the 1-based line number in the source file, or 0 if line numbers are unavailable.
	Line int64 `json:"line"`
	// Offset is the 0-based byte offset in the raw source stream.
	Offset int64 `json:"offset"`
	// Family is the record type or family classifier identified by the parser.
	Family string `json:"family,omitempty"`
	// Reason describes why the record could not be parsed or was categorized as anomalous.
	Reason string `json:"reason,omitempty"`
	// Snippet is a bounded, scrubbed preview of the raw input.
	Snippet string `json:"snippet"`
}

// ParserCompletenessReceipt is the typed sidecar receipt documenting raw benchmark
// parser completeness, conservation accounting, and anomalous record samples.
type ParserCompletenessReceipt struct {
	// Schema is the versioned schema string (must equal ParserCompletenessSchema).
	Schema string `json:"schema"`
	// Source identifies the raw artifact file and its digest.
	Source SourceIdentity `json:"source"`
	// ParserVersion is the semantic version or commit rev of the parser that produced the receipt.
	ParserVersion string `json:"parser_version"`
	// ParserSchema is the schema tag of the parser format specification.
	ParserSchema string `json:"parser_schema"`
	// TotalBytes is the total size of the raw input stream in bytes.
	TotalBytes int64 `json:"total_bytes"`
	// TotalRecords is the total count of distinct raw records encountered.
	TotalRecords int64 `json:"total_records"`

	// Conservation accounting categories:
	// ParsedRecords: records successfully parsed into target domain representations.
	ParsedRecords int64 `json:"parsed_records"`
	// IgnoredByPolicyRecords: records explicitly dropped or ignored per declared ingestion policy.
	IgnoredByPolicyRecords int64 `json:"ignored_by_policy_records"`
	// UnknownFamilyRecords: records belonging to an unrecognized benchmark/task family.
	UnknownFamilyRecords int64 `json:"unknown_family_records"`
	// MalformedRecords: structurally invalid records that failed syntactic decoding.
	MalformedRecords int64 `json:"malformed_records"`
	// TruncatedRecords: incomplete records cut off by stream boundary or EOF.
	TruncatedRecords int64 `json:"truncated_records"`
	// DuplicateRecords: records discarded because an identical record was already processed.
	DuplicateRecords int64 `json:"duplicate_records"`

	// Diagnostic samples (bounded and scrubbed):
	UnknownSamples   []RecordSample `json:"unknown_samples,omitempty"`
	MalformedSamples []RecordSample `json:"malformed_samples,omitempty"`
}

const (
	// DefaultMaxSampleCount is the default limit for recorded diagnostic samples.
	DefaultMaxSampleCount = 10
	// DefaultMaxSnippetLength is the maximum character length for scrubbed sample snippets.
	DefaultMaxSnippetLength = 256
)

// ScrubSnippet sanitizes control characters, flattens newlines, and truncates to maxLen.
func ScrubSnippet(raw string, maxLen int) string {
	if maxLen <= 0 {
		maxLen = DefaultMaxSnippetLength
	}
	s := strings.ReplaceAll(raw, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\t", " ")
	s = strings.TrimSpace(s)
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}

// AddUnknownSample appends a scrubbed sample for an unknown family record up to maxSamples.
func (r *ParserCompletenessReceipt) AddUnknownSample(sample RecordSample, maxSamples int) {
	if maxSamples <= 0 {
		maxSamples = DefaultMaxSampleCount
	}
	if len(r.UnknownSamples) >= maxSamples {
		return
	}
	sample.Snippet = ScrubSnippet(sample.Snippet, DefaultMaxSnippetLength)
	r.UnknownSamples = append(r.UnknownSamples, sample)
}

// AddMalformedSample appends a scrubbed sample for a malformed record up to maxSamples.
func (r *ParserCompletenessReceipt) AddMalformedSample(sample RecordSample, maxSamples int) {
	if maxSamples <= 0 {
		maxSamples = DefaultMaxSampleCount
	}
	if len(r.MalformedSamples) >= maxSamples {
		return
	}
	sample.Snippet = ScrubSnippet(sample.Snippet, DefaultMaxSnippetLength)
	r.MalformedSamples = append(r.MalformedSamples, sample)
}

// Validate checks the receipt for structural validity and conservation accounting.
//
// Invariant:
// TotalRecords == ParsedRecords + IgnoredByPolicyRecords + UnknownFamilyRecords +
//
//	MalformedRecords + TruncatedRecords + DuplicateRecords.
//
// If total does not equal sum, Validate returns ErrAccountingIncongruent.
func (r ParserCompletenessReceipt) Validate() error {
	// Check non-negative counts.
	if r.TotalBytes < 0 {
		return fmt.Errorf("%w: total bytes (%d)", ErrNegativeCount, r.TotalBytes)
	}
	if r.TotalRecords < 0 || r.ParsedRecords < 0 || r.IgnoredByPolicyRecords < 0 ||
		r.UnknownFamilyRecords < 0 || r.MalformedRecords < 0 || r.TruncatedRecords < 0 || r.DuplicateRecords < 0 {
		return ErrNegativeCount
	}

	// Conservation accounting invariant: TotalRecords must equal the sum of all parts.
	sum := r.ParsedRecords + r.IgnoredByPolicyRecords + r.UnknownFamilyRecords +
		r.MalformedRecords + r.TruncatedRecords + r.DuplicateRecords
	if r.TotalRecords != sum {
		return ErrAccountingIncongruent
	}

	// Schema verification.
	if r.Schema != ParserCompletenessSchema {
		return fmt.Errorf("%w: got %q, want %q", ErrInvalidSchema, r.Schema, ParserCompletenessSchema)
	}

	// Source identity verification.
	if strings.TrimSpace(r.Source.Location()) == "" || strings.TrimSpace(r.Source.SHA256) == "" {
		return ErrMissingSource
	}

	// Validate diagnostic sample pointers.
	if r.UnknownFamilyRecords == 0 && len(r.UnknownSamples) > 0 {
		return errors.New("unknown samples present but UnknownFamilyRecords is 0")
	}
	for i, s := range r.UnknownSamples {
		if s.Line < 0 || s.Offset < 0 {
			return fmt.Errorf("unknown sample %d has negative line or offset", i)
		}
	}
	if r.MalformedRecords == 0 && len(r.MalformedSamples) > 0 {
		return errors.New("malformed samples present but MalformedRecords is 0")
	}
	for i, s := range r.MalformedSamples {
		if s.Line < 0 || s.Offset < 0 {
			return fmt.Errorf("malformed sample %d has negative line or offset", i)
		}
	}

	return nil
}

// CompletenessRatio returns ParsedRecords / TotalRecords (or 0.0 when TotalRecords <= 0).
func (r ParserCompletenessReceipt) CompletenessRatio() float64 {
	if r.TotalRecords <= 0 {
		return 0.0
	}
	return float64(r.ParsedRecords) / float64(r.TotalRecords)
}

// MalformedRatio returns MalformedRecords / TotalRecords (or 0.0 when TotalRecords <= 0).
func (r ParserCompletenessReceipt) MalformedRatio() float64 {
	if r.TotalRecords <= 0 {
		return 0.0
	}
	return float64(r.MalformedRecords) / float64(r.TotalRecords)
}

// UnknownFamilyRatio returns UnknownFamilyRecords / TotalRecords (or 0.0 when TotalRecords <= 0).
func (r ParserCompletenessReceipt) UnknownFamilyRatio() float64 {
	if r.TotalRecords <= 0 {
		return 0.0
	}
	return float64(r.UnknownFamilyRecords) / float64(r.TotalRecords)
}

// TruncatedRatio returns TruncatedRecords / TotalRecords (or 0.0 when TotalRecords <= 0).
func (r ParserCompletenessReceipt) TruncatedRatio() float64 {
	if r.TotalRecords <= 0 {
		return 0.0
	}
	return float64(r.TruncatedRecords) / float64(r.TotalRecords)
}

// DuplicateRatio returns DuplicateRecords / TotalRecords (or 0.0 when TotalRecords <= 0).
func (r ParserCompletenessReceipt) DuplicateRatio() float64 {
	if r.TotalRecords <= 0 {
		return 0.0
	}
	return float64(r.DuplicateRecords) / float64(r.TotalRecords)
}

// JSON renders the receipt indented for persistence.
func (r ParserCompletenessReceipt) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// DecodeParserCompletenessReceipt decodes raw JSON bytes into a receipt.
func DecodeParserCompletenessReceipt(data []byte) (ParserCompletenessReceipt, error) {
	var r ParserCompletenessReceipt
	if err := json.Unmarshal(data, &r); err != nil {
		return ParserCompletenessReceipt{}, err
	}
	return r, nil
}

// CompletenessThresholds configures acceptance criteria for claim admission.
type CompletenessThresholds struct {
	// MinCompletenessRatio is the lower bound for ParsedRecords / TotalRecords (e.g. 1.0 or 0.95).
	MinCompletenessRatio float64 `json:"min_completeness_ratio"`
	// MaxMalformedRatio is the upper bound for MalformedRecords / TotalRecords (e.g. 0.0).
	MaxMalformedRatio float64 `json:"max_malformed_ratio"`
	// MaxUnknownRatio is an optional upper bound for UnknownFamilyRecords / TotalRecords.
	MaxUnknownRatio *float64 `json:"max_unknown_ratio,omitempty"`
	// MaxTruncatedRatio is an optional upper bound for TruncatedRecords / TotalRecords.
	MaxTruncatedRatio *float64 `json:"max_truncated_ratio,omitempty"`
	// MaxDuplicateRatio is an optional upper bound for DuplicateRecords / TotalRecords.
	MaxDuplicateRatio *float64 `json:"max_duplicate_ratio,omitempty"`
	// AllowEmpty determines whether a zero-record input may be admitted when MinCompletenessRatio is 0.
	AllowEmpty bool `json:"allow_empty"`
}

// AdmissionDecision represents the typed outcome of claim admission evaluation.
type AdmissionDecision string

const (
	// Admit indicates the receipt satisfies all completeness thresholds.
	Admit AdmissionDecision = "Admit"
	// Deny indicates the receipt breached one or more completeness thresholds.
	Deny AdmissionDecision = "Deny"
)

// AdmissionVerdict is the typed result of EvaluateClaimAdmission.
type AdmissionVerdict struct {
	// Verdict is either Admit or Deny.
	Verdict AdmissionDecision `json:"verdict"`
	// Reason is the typed failure token when Verdict is Deny, or empty when Admit.
	Reason string `json:"reason,omitempty"`
	// Message provides human-readable context for the verdict.
	Message string `json:"message,omitempty"`
	// CompletenessRatio is the observed completeness ratio.
	CompletenessRatio float64 `json:"completeness_ratio"`
	// MalformedRatio is the observed malformed record ratio.
	MalformedRatio float64 `json:"malformed_ratio"`
}

// Admitted reports whether the verdict is Admit.
func (v AdmissionVerdict) Admitted() bool {
	return v.Verdict == Admit
}

// IsAdmit reports whether the verdict is Admit.
func (v AdmissionVerdict) IsAdmit() bool {
	return v.Verdict == Admit
}

// IsDeny reports whether the verdict is Deny.
func (v AdmissionVerdict) IsDeny() bool {
	return v.Verdict == Deny
}

const floatEpsilon = 1e-9

// EvaluateClaimAdmission evaluates whether the completeness receipt satisfies declared thresholds.
func (r ParserCompletenessReceipt) EvaluateClaimAdmission(thresholds CompletenessThresholds) AdmissionVerdict {
	// 1. Fail-closed on receipt invalidity or broken conservation.
	if err := r.Validate(); err != nil {
		reason := ReasonInvalidReceipt
		if errors.Is(err, ErrAccountingIncongruent) {
			reason = ReasonAccountingIncongruent
		}
		return AdmissionVerdict{
			Verdict:           Deny,
			Reason:            reason,
			Message:           fmt.Sprintf("receipt failed validation: %v", err),
			CompletenessRatio: r.CompletenessRatio(),
			MalformedRatio:    r.MalformedRatio(),
		}
	}

	// 2. Zero-denominator and empty input handling.
	if r.TotalRecords == 0 {
		if !thresholds.AllowEmpty {
			return AdmissionVerdict{
				Verdict:           Deny,
				Reason:            ReasonEmptyInput,
				Message:           "empty benchmark input: total records is zero",
				CompletenessRatio: 0.0,
				MalformedRatio:    0.0,
			}
		}
		if thresholds.MinCompletenessRatio > 0 {
			return AdmissionVerdict{
				Verdict:           Deny,
				Reason:            ReasonParserCompletenessDeficit,
				Message:           fmt.Sprintf("completeness ratio 0.0000 below minimum threshold %.4f", thresholds.MinCompletenessRatio),
				CompletenessRatio: 0.0,
				MalformedRatio:    0.0,
			}
		}
		return AdmissionVerdict{
			Verdict:           Admit,
			CompletenessRatio: 0.0,
			MalformedRatio:    0.0,
			Message:           "empty input permitted by threshold policy",
		}
	}

	completenessRatio := r.CompletenessRatio()
	malformedRatio := r.MalformedRatio()

	// 3. Minimum completeness ratio check.
	if completenessRatio+floatEpsilon < thresholds.MinCompletenessRatio {
		return AdmissionVerdict{
			Verdict:           Deny,
			Reason:            ReasonParserCompletenessDeficit,
			Message:           fmt.Sprintf("completeness ratio %.4f below minimum threshold %.4f", completenessRatio, thresholds.MinCompletenessRatio),
			CompletenessRatio: completenessRatio,
			MalformedRatio:    malformedRatio,
		}
	}

	// 4. Maximum malformed ratio check.
	if malformedRatio-floatEpsilon > thresholds.MaxMalformedRatio {
		return AdmissionVerdict{
			Verdict:           Deny,
			Reason:            ReasonMalformedRecordBreach,
			Message:           fmt.Sprintf("malformed ratio %.4f exceeds maximum threshold %.4f", malformedRatio, thresholds.MaxMalformedRatio),
			CompletenessRatio: completenessRatio,
			MalformedRatio:    malformedRatio,
		}
	}

	// 5. Optional unknown family ratio check.
	if thresholds.MaxUnknownRatio != nil {
		unknownRatio := r.UnknownFamilyRatio()
		if unknownRatio-floatEpsilon > *thresholds.MaxUnknownRatio {
			return AdmissionVerdict{
				Verdict:           Deny,
				Reason:            ReasonUnknownFamilyBreach,
				Message:           fmt.Sprintf("unknown family ratio %.4f exceeds maximum threshold %.4f", unknownRatio, *thresholds.MaxUnknownRatio),
				CompletenessRatio: completenessRatio,
				MalformedRatio:    malformedRatio,
			}
		}
	}

	// 6. Optional truncated ratio check.
	if thresholds.MaxTruncatedRatio != nil {
		truncatedRatio := r.TruncatedRatio()
		if truncatedRatio-floatEpsilon > *thresholds.MaxTruncatedRatio {
			return AdmissionVerdict{
				Verdict:           Deny,
				Reason:            ReasonTruncatedRecordBreach,
				Message:           fmt.Sprintf("truncated ratio %.4f exceeds maximum threshold %.4f", truncatedRatio, *thresholds.MaxTruncatedRatio),
				CompletenessRatio: completenessRatio,
				MalformedRatio:    malformedRatio,
			}
		}
	}

	// 7. Optional duplicate ratio check.
	if thresholds.MaxDuplicateRatio != nil {
		duplicateRatio := r.DuplicateRatio()
		if duplicateRatio-floatEpsilon > *thresholds.MaxDuplicateRatio {
			return AdmissionVerdict{
				Verdict:           Deny,
				Reason:            ReasonDuplicateRecordBreach,
				Message:           fmt.Sprintf("duplicate ratio %.4f exceeds maximum threshold %.4f", duplicateRatio, *thresholds.MaxDuplicateRatio),
				CompletenessRatio: completenessRatio,
				MalformedRatio:    malformedRatio,
			}
		}
	}

	return AdmissionVerdict{
		Verdict:           Admit,
		CompletenessRatio: completenessRatio,
		MalformedRatio:    malformedRatio,
		Message:           "all completeness thresholds satisfied",
	}
}
