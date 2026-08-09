package headroom

import (
	"fmt"
	"math"
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
}

// ApplyLiveEvidence joins independently captured live metrics to a local
// same-corpus report. It refuses partial evidence: every successfully run arm
// must have all six finite, range-valid measurements under one declared
// workload/model/cache/grader contract.
func ApplyLiveEvidence(report ComparisonReport, evidence LiveComparisonEvidence) (ComparisonReport, error) {
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
	report.Measured = append([]ComparisonMetric(nil), requiredComparisonMetrics...)
	report.Pending = nil
	report.LiveEvidence = &evidence
	report.Complete = true
	return report, nil
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
