package disambiguation

import (
	"fmt"
	"time"
)

// Stable freshness reason codes. These values are part of the public JSON
// contract; callers should branch on them rather than human-readable text.
const (
	FreshnessReasonSourceCurrent     = "SOURCE_CURRENT"
	FreshnessReasonSourceOutdated    = "SOURCE_OUTDATED"
	FreshnessReasonProbeUnavailable  = "PROBE_UNAVAILABLE"
	FreshnessReasonEvidenceMalformed = "EVIDENCE_MALFORMED"
)

// FreshnessProbe is the public-only observation consumed by EvaluateFreshness.
// EvidenceValid applies only when Available is true. An unavailable probe has
// no evidence to validate and therefore yields unknown, never fresh.
type FreshnessProbe struct {
	Probe         string `json:"probe"`
	CheckedAt     string `json:"checked_at"`
	Available     bool   `json:"available"`
	EvidenceValid bool   `json:"evidence_valid"`
	Current       bool   `json:"current"`
}

// EvaluateFreshness converts one public probe observation into the four-state
// wire verdict. Invalid probe metadata or malformed returned evidence outranks
// age; probe unavailability is unknown; only valid, available, current evidence
// can be fresh. It performs no I/O and is not an index writer.
func EvaluateFreshness(observation FreshnessProbe) Freshness {
	result := Freshness{CheckedAt: observation.CheckedAt, Probe: observation.Probe}
	if validateFreshnessProbeMetadata(observation) != nil || (observation.Available && !observation.EvidenceValid) {
		result.Verdict = FreshnessInvalid
		result.ReasonCode = FreshnessReasonEvidenceMalformed
		return result
	}
	if !observation.Available {
		result.Verdict = FreshnessUnknown
		result.ReasonCode = FreshnessReasonProbeUnavailable
		return result
	}
	if observation.Current {
		result.Verdict = FreshnessFresh
		result.ReasonCode = FreshnessReasonSourceCurrent
		return result
	}
	result.Verdict = FreshnessStale
	result.ReasonCode = FreshnessReasonSourceOutdated
	return result
}

func validateFreshnessProbeMetadata(observation FreshnessProbe) error {
	if err := requireText("freshness.probe", observation.Probe); err != nil {
		return err
	}
	if err := requireText("freshness.checked_at", observation.CheckedAt); err != nil {
		return err
	}
	checkedAt, err := time.Parse(time.RFC3339, observation.CheckedAt)
	if err != nil || checkedAt.Format(time.RFC3339) != observation.CheckedAt {
		return fmt.Errorf("freshness.checked_at must be canonical RFC3339")
	}
	return nil
}

func expectedFreshnessReason(verdict FreshnessVerdict) (string, bool) {
	switch verdict {
	case FreshnessFresh:
		return FreshnessReasonSourceCurrent, true
	case FreshnessStale:
		return FreshnessReasonSourceOutdated, true
	case FreshnessUnknown:
		return FreshnessReasonProbeUnavailable, true
	case FreshnessInvalid:
		return FreshnessReasonEvidenceMalformed, true
	default:
		return "", false
	}
}

// FreshnessSelfCheckCase is one row of the hermetic four-state package witness.
type FreshnessSelfCheckCase struct {
	Verdict    FreshnessVerdict `json:"verdict"`
	ReasonCode string           `json:"reason_code"`
	Passed     bool             `json:"passed"`
}

// FreshnessSelfCheckReport is the stable CLI/package JSON witness.
type FreshnessSelfCheckReport struct {
	Schema string                   `json:"schema"`
	Cases  []FreshnessSelfCheckCase `json:"cases"`
	Passed bool                     `json:"passed"`
}

// FreshnessSelfCheck exercises every public state without filesystem, network,
// private-source, or writer dependencies.
func FreshnessSelfCheck() FreshnessSelfCheckReport {
	checkedAt := "2026-08-11T00:00:00Z"
	observations := []FreshnessProbe{
		{Probe: "public-self-test/1", CheckedAt: checkedAt, Available: true, EvidenceValid: true, Current: true},
		{Probe: "public-self-test/1", CheckedAt: checkedAt, Available: true, EvidenceValid: true, Current: false},
		{Probe: "public-self-test/1", CheckedAt: checkedAt, Available: false},
		{Probe: "public-self-test/1", CheckedAt: checkedAt, Available: true, EvidenceValid: false},
	}
	report := FreshnessSelfCheckReport{Schema: EntrySchemaVersion, Passed: true}
	for _, observation := range observations {
		got := EvaluateFreshness(observation)
		expectedReason, known := expectedFreshnessReason(got.Verdict)
		passed := known && got.ReasonCode == expectedReason
		report.Cases = append(report.Cases, FreshnessSelfCheckCase{Verdict: got.Verdict, ReasonCode: got.ReasonCode, Passed: passed})
		report.Passed = report.Passed && passed
	}
	return report
}
func validateFreshness(freshness Freshness) error {
	expectedReason, ok := expectedFreshnessReason(freshness.Verdict)
	if !ok {
		return fmt.Errorf("freshness.verdict %q is not one of fresh, stale, unknown, invalid", freshness.Verdict)
	}
	if err := requireText("freshness.reason_code", freshness.ReasonCode); err != nil {
		return err
	}
	if freshness.ReasonCode != expectedReason {
		return fmt.Errorf("freshness.reason_code %q does not match verdict %q (want %q)", freshness.ReasonCode, freshness.Verdict, expectedReason)
	}
	return validateFreshnessProbeMetadata(FreshnessProbe{Probe: freshness.Probe, CheckedAt: freshness.CheckedAt})
}
