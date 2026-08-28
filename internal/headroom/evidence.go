package headroom

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/big"
	"strings"
)

// LiveArmMetrics are provider/task measurements gathered outside the local
// compressor runner. Rates are in [0,1], token counts and milliseconds are
// non-negative, and TotalCostUSD is the observed all-in cost for the arm.
type LiveArmMetrics struct {
	TaskSuccess         float64 `json:"task_success"`
	MetricFactRecall    float64 `json:"retained_fact_recall"`
	ProviderInputTokens int64   `json:"provider_input_tokens"`
	TTFTMilliseconds    float64 `json:"ttft_ms"`
	RegrowthTaxTokens   int64   `json:"regrowth_tax_tokens"`
	TotalCostUSD        float64 `json:"total_cost_usd"`
}

// LiveComparisonEvidence is an independently captured provider/task read-back.
// Witness identifies the immutable artifact or run ledger from which the
// measurements came; an inline self-report is intentionally not enough.
type LiveComparisonEvidence struct {
	Schema         string                    `json:"schema"`
	Witness        string                    `json:"witness"`
	WorkloadDigest string                    `json:"workload_digest"`
	Model          string                    `json:"model"`
	Provider       string                    `json:"provider"`
	CacheState     string                    `json:"cache_state"`
	Grader         string                    `json:"grader"`
	Arms           map[string]LiveArmMetrics `json:"arms"`
	PromotionArms  []PromotionArmEvidence    `json:"promotion_arms,omitempty"`
	Decision       *PromotionDecision        `json:"promotion_decision,omitempty"`
}

// UnmarshalJSON makes the existing --evidence seam strict without requiring command-layer wiring.
// Legacy v1 receipts remain valid; when promotion_arms is present, every nested field is required.
func (e *LiveComparisonEvidence) UnmarshalJSON(raw []byte) error {
	if err := rejectDuplicateJSONKeys(raw); err != nil {
		return err
	}
	if err := requirePromotionJSONCompleteness(raw); err != nil {
		return err
	}
	type wire LiveComparisonEvidence
	var decoded wire
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&decoded); err != nil {
		return fmt.Errorf("headroom evidence: decode: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return fmt.Errorf("headroom evidence: trailing JSON")
	}
	*e = LiveComparisonEvidence(decoded)
	return nil
}

// ApplyLiveEvidence joins independently captured live metrics to a local
// same-corpus report. It refuses partial evidence: every successfully run arm
// must have all six finite, range-valid measurements under one declared
// workload/model/cache/grader contract.
func ApplyLiveEvidence(report ComparisonReport, evidence LiveComparisonEvidence) (ComparisonReport, error) {
	evidence.Decision = nil // output-only; never trust a caller-supplied decision
	if evidence.Schema != "fak-headroom-live-evidence/1" {
		return report, fmt.Errorf("headroom evidence: schema=%q", evidence.Schema)
	}
	for name, value := range map[string]string{
		"witness": evidence.Witness, "workload_digest": evidence.WorkloadDigest,
		"model": evidence.Model, "provider": evidence.Provider,
		"cache_state": evidence.CacheState, "grader": evidence.Grader,
	} {
		if strings.TrimSpace(value) == "" {
			return report, fmt.Errorf("headroom evidence: %s is required", name)
		}
	}
	if !report.ArmsComplete {
		return report, fmt.Errorf("headroom evidence: local arms are incomplete")
	}
	seen := make(map[string]struct{}, len(report.Arms))
	for _, arm := range report.Arms {
		if _, duplicate := seen[arm.Name]; duplicate {
			return report, fmt.Errorf("headroom evidence: duplicate local arm %q", arm.Name)
		}
		seen[arm.Name] = struct{}{}
		metrics, ok := evidence.Arms[arm.Name]
		if !ok {
			return report, fmt.Errorf("headroom evidence: arm %q is missing", arm.Name)
		}
		if err := validateLiveArmMetrics(arm.Name, metrics); err != nil {
			return report, err
		}
	}
	for name := range evidence.Arms {
		found := false
		for _, arm := range report.Arms {
			if arm.Name == name {
				found = true
				break
			}
		}
		if !found {
			return report, fmt.Errorf("headroom evidence: unexpected arm %q", name)
		}
	}
	if len(evidence.PromotionArms) > 0 {
		if _, _, reason := validatePromotionEvidence(evidence); reason != "" {
			return report, fmt.Errorf("headroom evidence: promotion %s", reason)
		}
		if err := validatePromotionLegacyCoherence(evidence); err != nil {
			return report, err
		}
		decision := DecideNativePromotion(evidence)
		evidence.Decision = &decision
	}
	report.Measured = append([]ComparisonMetric(nil), requiredComparisonMetrics...)
	report.Pending = nil
	report.LiveEvidence = &evidence
	report.Complete = true
	return report, nil
}

func validatePromotionLegacyCoherence(evidence LiveComparisonEvidence) error {
	if len(evidence.Arms) != 2 {
		return fmt.Errorf("headroom evidence: promotion and legacy arm sets differ")
	}
	for _, arm := range evidence.PromotionArms {
		legacy, ok := evidence.Arms[arm.Name]
		if !ok {
			return fmt.Errorf("headroom evidence: promotion arm %q has no legacy arm", arm.Name)
		}
		checks := []struct {
			name  string
			equal bool
		}{
			{"task_success", arm.Metrics.TaskSuccess == legacy.TaskSuccess},
			{"retained_fact_recall", arm.Metrics.RetainedFactRecall == legacy.MetricFactRecall},
			{"initial_input_tokens", arm.Metrics.InitialInputTokens == legacy.ProviderInputTokens},
			{"regrowth_input_tokens", arm.Metrics.RegrowthInputTokens == legacy.RegrowthTaxTokens},
		}
		for _, check := range checks {
			if !check.equal {
				return fmt.Errorf("headroom evidence: arm %q promotion/legacy %s mismatch", arm.Name, check.name)
			}
		}
	}
	return nil
}

func validateLiveArmMetrics(name string, metrics LiveArmMetrics) error {
	for metric, value := range map[string]float64{
		"task_success":         metrics.TaskSuccess,
		"retained_fact_recall": metrics.MetricFactRecall,
	} {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
			return fmt.Errorf("headroom evidence: arm %q %s=%v is outside [0,1]", name, metric, value)
		}
	}
	for metric, value := range map[string]float64{
		"ttft_ms":        metrics.TTFTMilliseconds,
		"total_cost_usd": metrics.TotalCostUSD,
	} {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			return fmt.Errorf("headroom evidence: arm %q %s=%v must be finite and non-negative", name, metric, value)
		}
	}
	if metrics.ProviderInputTokens < 0 || metrics.RegrowthTaxTokens < 0 {
		return fmt.Errorf("headroom evidence: arm %q token counts must be non-negative", name)
	}
	return nil
}

const PromotionControlName = "none"

// PromotionProvenance is the complete matched-run identity. Pointer-valued numeric fields make
// omitted JSON distinguishable from legitimate zero values without payload-bearing sentinels.
type PromotionProvenance struct {
	Witness        string   `json:"witness"`
	WorkloadDigest string   `json:"workload_digest"`
	Model          string   `json:"model"`
	Provider       string   `json:"provider"`
	Seed           *int64   `json:"seed"`
	Temperature    *float64 `json:"temperature"`
	OutputLimit    *int64   `json:"output_limit"`
	CacheState     string   `json:"cache_state"`
	GraderID       string   `json:"grader_id"`
	GraderVersion  string   `json:"grader_version"`
}

// PromotionArmMetrics contains aggregates and recovery accounting only. EffectiveInputTokens
// must exactly conserve the initial input plus every recovery tax.
type PromotionArmMetrics struct {
	TaskSuccess                      float64 `json:"task_success"`
	RetainedFactRecall               float64 `json:"retained_fact_recall"`
	InitialInputTokens               int64   `json:"initial_input_tokens"`
	RegrowthInputTokens              int64   `json:"regrowth_input_tokens"`
	RefetchInputTokens               int64   `json:"refetch_input_tokens"`
	OverrideInputTokens              int64   `json:"override_input_tokens"`
	EffectiveInputTokens             int64   `json:"effective_input_tokens"`
	P95ResultToResponseMilliseconds  float64 `json:"p95_result_to_response_ms"`
	ExactOriginalRestorationFailures int64   `json:"exact_original_restoration_failures"`
}

type PromotionArmEvidence struct {
	Name       string              `json:"name"`
	CaseIDs    []string            `json:"case_ids"`
	Provenance PromotionProvenance `json:"provenance"`
	Metrics    PromotionArmMetrics `json:"metrics"`
}

type PromotionVerdict string
type PromotionHoldReason string

const (
	PromotionPromote PromotionVerdict = "promote"
	PromotionHold    PromotionVerdict = "hold"

	HoldInvalidSchema         PromotionHoldReason = "invalid_schema"
	HoldIncompleteArms        PromotionHoldReason = "incomplete_arms"
	HoldDuplicateArms         PromotionHoldReason = "duplicate_arms"
	HoldReorderedArms         PromotionHoldReason = "reordered_arms"
	HoldMissingCases          PromotionHoldReason = "missing_cases"
	HoldInvalidCases          PromotionHoldReason = "invalid_cases"
	HoldDuplicateCases        PromotionHoldReason = "duplicate_cases"
	HoldReorderedCases        PromotionHoldReason = "reordered_cases"
	HoldMissingProvenance     PromotionHoldReason = "missing_provenance"
	HoldInvalidProvenance     PromotionHoldReason = "invalid_provenance"
	HoldProvenanceMismatch    PromotionHoldReason = "provenance_mismatch"
	HoldNegativeCounter       PromotionHoldReason = "negative_counter"
	HoldNonFiniteMetric       PromotionHoldReason = "non_finite_metric"
	HoldMetricOutOfRange      PromotionHoldReason = "metric_out_of_range"
	HoldCounterOverflow       PromotionHoldReason = "counter_overflow"
	HoldCounterNonConserving  PromotionHoldReason = "counter_non_conserving"
	HoldTaskSuccessRegression PromotionHoldReason = "task_success_regression"
	HoldFactRecallRegression  PromotionHoldReason = "retained_fact_recall_regression"
	HoldEffectiveInputTooHigh PromotionHoldReason = "effective_input_above_90_percent"
	HoldLatencyRegression     PromotionHoldReason = "latency_above_105_percent"
	HoldRestorationFailure    PromotionHoldReason = "exact_original_restoration_failure"
)

// PromotionDecision is deterministic and advisory. RuntimeCompressor remains noop until the
// separately authorized live-promotion follow-on changes selection.
type PromotionDecision struct {
	Verdict PromotionVerdict      `json:"verdict"`
	Reasons []PromotionHoldReason `json:"reasons"`
}

func holdPromotion(reason PromotionHoldReason) PromotionDecision {
	return PromotionDecision{Verdict: PromotionHold, Reasons: []PromotionHoldReason{reason}}
}

// DecideNativePromotion is a pure, fail-closed decision over a matched none/native receipt.
func DecideNativePromotion(evidence LiveComparisonEvidence) PromotionDecision {
	control, native, reason := validatePromotionEvidence(evidence)
	if reason != "" {
		return holdPromotion(reason)
	}

	var reasons []PromotionHoldReason
	if native.Metrics.TaskSuccess < control.Metrics.TaskSuccess {
		reasons = append(reasons, HoldTaskSuccessRegression)
	}
	if native.Metrics.RetainedFactRecall < control.Metrics.RetainedFactRecall {
		reasons = append(reasons, HoldFactRecallRegression)
	}
	if !fractionAtMost(native.Metrics.EffectiveInputTokens, control.Metrics.EffectiveInputTokens, 9, 10) {
		reasons = append(reasons, HoldEffectiveInputTooHigh)
	}
	if !floatFractionAtMost(native.Metrics.P95ResultToResponseMilliseconds, control.Metrics.P95ResultToResponseMilliseconds, 105, 100) {
		reasons = append(reasons, HoldLatencyRegression)
	}
	if control.Metrics.ExactOriginalRestorationFailures != 0 || native.Metrics.ExactOriginalRestorationFailures != 0 {
		reasons = append(reasons, HoldRestorationFailure)
	}
	if len(reasons) != 0 {
		return PromotionDecision{Verdict: PromotionHold, Reasons: reasons}
	}
	return PromotionDecision{Verdict: PromotionPromote, Reasons: []PromotionHoldReason{}}
}

func validatePromotionEvidence(evidence LiveComparisonEvidence) (control, native PromotionArmEvidence, reason PromotionHoldReason) {
	if evidence.Schema != "fak-headroom-live-evidence/1" {
		return control, native, HoldInvalidSchema
	}
	if len(evidence.PromotionArms) != 2 {
		return control, native, HoldIncompleteArms
	}
	if evidence.PromotionArms[0].Name == evidence.PromotionArms[1].Name {
		return control, native, HoldDuplicateArms
	}
	if evidence.PromotionArms[0].Name != PromotionControlName || evidence.PromotionArms[1].Name != NativeName {
		return control, native, HoldReorderedArms
	}
	control, native = evidence.PromotionArms[0], evidence.PromotionArms[1]
	if len(control.CaseIDs) == 0 || len(native.CaseIDs) == 0 {
		return control, native, HoldMissingCases
	}
	if hasDuplicate(control.CaseIDs) || hasDuplicate(native.CaseIDs) {
		return control, native, HoldDuplicateCases
	}
	if !validCaseIDs(control.CaseIDs) || !validCaseIDs(native.CaseIDs) {
		return control, native, HoldInvalidCases
	}
	if len(control.CaseIDs) != len(native.CaseIDs) {
		return control, native, HoldMissingCases
	}
	if !equalStrings(control.CaseIDs, native.CaseIDs) {
		return control, native, HoldReorderedCases
	}
	if !completePromotionProvenance(control.Provenance) || !completePromotionProvenance(native.Provenance) {
		return control, native, HoldMissingProvenance
	}
	for _, p := range []PromotionProvenance{control.Provenance, native.Provenance} {
		if math.IsNaN(*p.Temperature) || math.IsInf(*p.Temperature, 0) {
			return control, native, HoldNonFiniteMetric
		}
		if *p.Temperature < 0 {
			return control, native, HoldMetricOutOfRange
		}
		if *p.OutputLimit <= 0 {
			return control, native, HoldNegativeCounter
		}
		if !validPromotionProvenance(p) {
			return control, native, HoldInvalidProvenance
		}
	}
	grader := control.Provenance.GraderID + "@" + control.Provenance.GraderVersion
	if evidence.WorkloadDigest != control.Provenance.WorkloadDigest || evidence.Model != control.Provenance.Model ||
		evidence.Provider != control.Provenance.Provider || evidence.CacheState != control.Provenance.CacheState ||
		evidence.Grader != grader || !validEvidenceReference(evidence.Witness, 256) {
		return control, native, HoldProvenanceMismatch
	}
	if !matchedPromotionProvenance(control.Provenance, native.Provenance) {
		return control, native, HoldProvenanceMismatch
	}
	for _, arm := range []PromotionArmEvidence{control, native} {
		if reason := validatePromotionMetrics(arm.Metrics); reason != "" {
			return control, native, reason
		}
	}
	return control, native, ""
}

func completePromotionProvenance(p PromotionProvenance) bool {
	for _, value := range []string{p.Witness, p.WorkloadDigest, p.Model, p.Provider, p.CacheState, p.GraderID, p.GraderVersion} {
		if strings.TrimSpace(value) == "" {
			return false
		}
	}
	return p.Seed != nil && p.Temperature != nil && p.OutputLimit != nil
}

func validPromotionProvenance(p PromotionProvenance) bool {
	return validEvidenceReference(p.Witness, 256) && validSHA256Digest(p.WorkloadDigest) &&
		validEvidenceToken(p.Model, 128) && validEvidenceToken(p.Provider, 128) &&
		validEvidenceToken(p.CacheState, 64) && validEvidenceToken(p.GraderID, 128) &&
		validEvidenceToken(p.GraderVersion, 64)
}

func matchedPromotionProvenance(a, b PromotionProvenance) bool {
	return a.WorkloadDigest == b.WorkloadDigest && a.Model == b.Model && a.Provider == b.Provider &&
		*a.Seed == *b.Seed && *a.Temperature == *b.Temperature && *a.OutputLimit == *b.OutputLimit &&
		a.CacheState == b.CacheState && a.GraderID == b.GraderID && a.GraderVersion == b.GraderVersion
}

func validatePromotionMetrics(m PromotionArmMetrics) PromotionHoldReason {
	for _, value := range []float64{m.TaskSuccess, m.RetainedFactRecall, m.P95ResultToResponseMilliseconds} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return HoldNonFiniteMetric
		}
	}
	if m.TaskSuccess < 0 || m.RetainedFactRecall < 0 || m.P95ResultToResponseMilliseconds < 0 ||
		m.TaskSuccess > 1 || m.RetainedFactRecall > 1 {
		return HoldMetricOutOfRange
	}
	counters := []int64{m.InitialInputTokens, m.RegrowthInputTokens, m.RefetchInputTokens, m.OverrideInputTokens, m.EffectiveInputTokens, m.ExactOriginalRestorationFailures}
	for _, value := range counters {
		if value < 0 {
			return HoldNegativeCounter
		}
	}
	total := int64(0)
	for _, value := range counters[:4] {
		if value > math.MaxInt64-total {
			return HoldCounterOverflow
		}
		total += value
	}
	if total != m.EffectiveInputTokens {
		return HoldCounterNonConserving
	}
	return ""
}

func hasDuplicate(values []string) bool {
	seen := make(map[string]bool, len(values))
	for _, value := range values {
		if seen[value] {
			return true
		}
		seen[value] = true
	}
	return false
}

func validCaseIDs(values []string) bool {
	for _, value := range values {
		if !validEvidenceToken(value, 128) {
			return false
		}
	}
	return true
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func fractionAtMost(value, control int64, numerator, denominator int64) bool {
	if control <= 0 {
		return false
	}
	left := new(big.Int).Mul(big.NewInt(value), big.NewInt(denominator))
	right := new(big.Int).Mul(big.NewInt(control), big.NewInt(numerator))
	return left.Cmp(right) <= 0
}

func floatFractionAtMost(value, control float64, numerator, denominator int64) bool {
	valueRat := new(big.Rat).SetFloat64(value)
	controlRat := new(big.Rat).SetFloat64(control)
	if valueRat == nil || controlRat == nil {
		return false
	}
	valueRat.Mul(valueRat, big.NewRat(denominator, 1))
	controlRat.Mul(controlRat, big.NewRat(numerator, 1))
	return valueRat.Cmp(controlRat) <= 0
}

func validEvidenceReference(value string, max int) bool {
	return len(value) <= max && strings.Contains(value, "://") && validEvidenceToken(value, max)
}

func validEvidenceToken(value string, max int) bool {
	if value == "" || len(value) > max {
		return false
	}
	for i := 0; i < len(value); i++ {
		c := value[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || strings.ContainsRune("-._:/+@", rune(c)) {
			continue
		}
		return false
	}
	return true
}

func validSHA256Digest(value string) bool {
	if len(value) != len("sha256:")+64 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	for _, c := range value[len("sha256:"):] {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

func requirePromotionJSONCompleteness(raw []byte) error {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return fmt.Errorf("headroom promotion evidence: decode: %w", err)
	}
	rawArms, supplied := root["promotion_arms"]
	if !supplied {
		return nil
	}
	var arms []json.RawMessage
	if err := json.Unmarshal(rawArms, &arms); err != nil {
		return fmt.Errorf("headroom promotion evidence: promotion_arms: %w", err)
	}
	if len(arms) == 0 {
		return fmt.Errorf("headroom promotion evidence: promotion_arms is incomplete")
	}
	for i, rawArm := range arms {
		var arm map[string]json.RawMessage
		if err := json.Unmarshal(rawArm, &arm); err != nil {
			return fmt.Errorf("headroom promotion evidence: arm[%d]: %w", i, err)
		}
		if err := requireJSONFields(fmt.Sprintf("arm[%d]", i), arm, "name", "case_ids", "provenance", "metrics"); err != nil {
			return err
		}
		var provenance map[string]json.RawMessage
		if err := json.Unmarshal(arm["provenance"], &provenance); err != nil {
			return fmt.Errorf("headroom promotion evidence: arm[%d] provenance: %w", i, err)
		}
		if err := requireJSONFields(fmt.Sprintf("arm[%d] provenance", i), provenance,
			"witness", "workload_digest", "model", "provider", "seed", "temperature", "output_limit", "cache_state", "grader_id", "grader_version"); err != nil {
			return err
		}
		var metrics map[string]json.RawMessage
		if err := json.Unmarshal(arm["metrics"], &metrics); err != nil {
			return fmt.Errorf("headroom promotion evidence: arm[%d] metrics: %w", i, err)
		}
		if err := requireJSONFields(fmt.Sprintf("arm[%d] metrics", i), metrics,
			"task_success", "retained_fact_recall", "initial_input_tokens", "regrowth_input_tokens",
			"refetch_input_tokens", "override_input_tokens", "effective_input_tokens",
			"p95_result_to_response_ms", "exact_original_restoration_failures"); err != nil {
			return err
		}
	}
	return nil
}

func requireJSONFields(where string, object map[string]json.RawMessage, fields ...string) error {
	for _, field := range fields {
		if raw, ok := object[field]; !ok || len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
			return fmt.Errorf("headroom promotion evidence: %s field %q is required", where, field)
		}
	}
	return nil
}

func rejectDuplicateJSONKeys(raw []byte) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	var walk func() error
	walk = func() error {
		tok, err := dec.Token()
		if err != nil {
			return fmt.Errorf("headroom promotion evidence: decode: %w", err)
		}
		delim, ok := tok.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := map[string]bool{}
			for dec.More() {
				keyToken, err := dec.Token()
				if err != nil {
					return err
				}
				key, ok := keyToken.(string)
				if !ok || seen[key] {
					return fmt.Errorf("headroom promotion evidence: duplicate object key %q", key)
				}
				seen[key] = true
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = dec.Token()
			return err
		case '[':
			for dec.More() {
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = dec.Token()
			return err
		default:
			return fmt.Errorf("headroom promotion evidence: unexpected JSON delimiter %q", delim)
		}
	}
	if err := walk(); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return fmt.Errorf("headroom promotion evidence: trailing JSON")
	}
	return nil
}
