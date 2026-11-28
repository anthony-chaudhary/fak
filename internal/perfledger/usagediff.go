// Inspired by ai-dynamo/aiperf
// client-tokenized counts and reporting percentage discrepancy to surface
// billing-vs-actual tokenization mismatches.
package perfledger

import (
	"fmt"
)

// Curated summary tag names aligned with summary.go.
const (
	TagUsagePromptDiffPct     = "usage_prompt_tokens_diff_pct"
	TagUsageCompletionDiffPct = "usage_completion_tokens_diff_pct"
	TagUsageReasoningDiffPct  = "usage_reasoning_tokens_diff_pct"
	TagUsageDiscrepancyCount  = "usage_discrepancy_count"
	TagOSLMismatchDiffPct     = "osl_mismatch_diff_pct"
	TagOSLMismatchCount       = "osl_mismatch_count"
)

// UsageDiffInput holds paired client-tokenized and API-reported token counts
// for one request. Nil pointers mark dimensions not observed on that request.
type UsageDiffInput struct {
	ClientPromptTokens     *int `json:"client_prompt_tokens,omitempty"`
	APIPromptTokens        *int `json:"api_prompt_tokens,omitempty"`
	ClientCompletionTokens *int `json:"client_completion_tokens,omitempty"`
	APICompletionTokens    *int `json:"api_completion_tokens,omitempty"`
	ClientReasoningTokens  *int `json:"client_reasoning_tokens,omitempty"`
	APIReasoningTokens     *int `json:"api_reasoning_tokens,omitempty"`
}

// UsageDiffRecord captures per-request API-vs-client deltas and percentage
// discrepancies ((api - client) / client * 100).
type UsageDiffRecord struct {
	PromptDelta       *int     `json:"prompt_delta,omitempty"`
	CompletionDelta   *int     `json:"completion_delta,omitempty"`
	ReasoningDelta    *int     `json:"reasoning_delta,omitempty"`
	PromptDiffPct     *float64 `json:"prompt_diff_pct,omitempty"`
	CompletionDiffPct *float64 `json:"completion_diff_pct,omitempty"`
	ReasoningDiffPct  *float64 `json:"reasoning_diff_pct,omitempty"`
	HasDiscrepancy    bool     `json:"has_discrepancy"`
}

// UsageDiffSummary aggregates usage discrepancies across a run.
type UsageDiffSummary struct {
	Requests                 int             `json:"requests"`
	DiscrepancyCount         int             `json:"discrepancy_count"`
	OSLMismatchCount         int             `json:"osl_mismatch_count"`
	PromptTotalAPI           int64           `json:"prompt_total_api"`
	PromptTotalClient        int64           `json:"prompt_total_client"`
	CompletionTotalAPI       int64           `json:"completion_total_api"`
	CompletionTotalClient    int64           `json:"completion_total_client"`
	ReasoningTotalAPI        int64           `json:"reasoning_total_api"`
	ReasoningTotalClient     int64           `json:"reasoning_total_client"`
	PromptOverallDiffPct     *float64        `json:"prompt_overall_diff_pct,omitempty"`
	CompletionOverallDiffPct *float64        `json:"completion_overall_diff_pct,omitempty"`
	ReasoningOverallDiffPct  *float64        `json:"reasoning_overall_diff_pct,omitempty"`
	PromptDiffBand           *PercentileBand `json:"prompt_diff_band,omitempty"`
	CompletionDiffBand       *PercentileBand `json:"completion_diff_band,omitempty"`
	ReasoningDiffBand        *PercentileBand `json:"reasoning_diff_band,omitempty"`
	PromptMeanDiffPct        *float64        `json:"prompt_mean_diff_pct,omitempty"`
	CompletionMeanDiffPct    *float64        `json:"completion_mean_diff_pct,omitempty"`
	ReasoningMeanDiffPct     *float64        `json:"reasoning_mean_diff_pct,omitempty"`
}

// TokenDiffPct computes the signed percentage discrepancy
// (apiTokens - clientTokens) / clientTokens * 100.
// Negative counts return an error; apiTokens > 0 with clientTokens == 0 returns
// ErrNoMetricValue because percentage relative to zero client tokens is undefined.
func TokenDiffPct(apiTokens, clientTokens int) (float64, error) {
	if apiTokens < 0 || clientTokens < 0 {
		return 0, fmt.Errorf("perfledger: TokenDiffPct: negative token count api=%d client=%d", apiTokens, clientTokens)
	}
	if clientTokens == 0 {
		if apiTokens == 0 {
			return 0, nil
		}
		return 0, ErrNoMetricValue
	}
	return float64(apiTokens-clientTokens) / float64(clientTokens) * 100.0, nil
}

// ComputeUsageDiff evaluates API-vs-client count deltas and percentage
// discrepancies for one request record.
func ComputeUsageDiff(in UsageDiffInput) (UsageDiffRecord, error) {
	var rec UsageDiffRecord

	if in.APIPromptTokens != nil && in.ClientPromptTokens != nil {
		delta, pct, err := pairDiff(*in.APIPromptTokens, *in.ClientPromptTokens)
		if err != nil {
			return UsageDiffRecord{}, err
		}
		rec.PromptDelta = &delta
		if pct != nil {
			rec.PromptDiffPct = pct
		}
		if delta != 0 {
			rec.HasDiscrepancy = true
		}
	}

	if in.APICompletionTokens != nil && in.ClientCompletionTokens != nil {
		delta, pct, err := pairDiff(*in.APICompletionTokens, *in.ClientCompletionTokens)
		if err != nil {
			return UsageDiffRecord{}, err
		}
		rec.CompletionDelta = &delta
		if pct != nil {
			rec.CompletionDiffPct = pct
		}
		if delta != 0 {
			rec.HasDiscrepancy = true
		}
	}

	if in.APIReasoningTokens != nil && in.ClientReasoningTokens != nil {
		delta, pct, err := pairDiff(*in.APIReasoningTokens, *in.ClientReasoningTokens)
		if err != nil {
			return UsageDiffRecord{}, err
		}
		rec.ReasoningDelta = &delta
		if pct != nil {
			rec.ReasoningDiffPct = pct
		}
		if delta != 0 {
			rec.HasDiscrepancy = true
		}
	}

	return rec, nil
}

// AggregateUsageDiff computes run-level discrepancy counts, aggregate diff
// percentages, and per-request diff percentile bands across records.
func AggregateUsageDiff(inputs []UsageDiffInput) (UsageDiffSummary, error) {
	sum := UsageDiffSummary{Requests: len(inputs)}
	var promptPcts, compPcts, reasPcts []float64

	for _, in := range inputs {
		rec, err := ComputeUsageDiff(in)
		if err != nil {
			return UsageDiffSummary{}, err
		}
		if rec.HasDiscrepancy {
			sum.DiscrepancyCount++
		}
		if rec.CompletionDelta != nil && *rec.CompletionDelta != 0 {
			sum.OSLMismatchCount++
		}

		if in.APIPromptTokens != nil && in.ClientPromptTokens != nil {
			sum.PromptTotalAPI += int64(*in.APIPromptTokens)
			sum.PromptTotalClient += int64(*in.ClientPromptTokens)
			if rec.PromptDiffPct != nil {
				promptPcts = append(promptPcts, *rec.PromptDiffPct)
			}
		}
		if in.APICompletionTokens != nil && in.ClientCompletionTokens != nil {
			sum.CompletionTotalAPI += int64(*in.APICompletionTokens)
			sum.CompletionTotalClient += int64(*in.ClientCompletionTokens)
			if rec.CompletionDiffPct != nil {
				compPcts = append(compPcts, *rec.CompletionDiffPct)
			}
		}
		if in.APIReasoningTokens != nil && in.ClientReasoningTokens != nil {
			sum.ReasoningTotalAPI += int64(*in.APIReasoningTokens)
			sum.ReasoningTotalClient += int64(*in.ClientReasoningTokens)
			if rec.ReasoningDiffPct != nil {
				reasPcts = append(reasPcts, *rec.ReasoningDiffPct)
			}
		}
	}

	sum.PromptOverallDiffPct = totalDiffPctPtr(sum.PromptTotalAPI, sum.PromptTotalClient)
	sum.CompletionOverallDiffPct = totalDiffPctPtr(sum.CompletionTotalAPI, sum.CompletionTotalClient)
	sum.ReasoningOverallDiffPct = totalDiffPctPtr(sum.ReasoningTotalAPI, sum.ReasoningTotalClient)

	sum.PromptDiffBand, sum.PromptMeanDiffPct = summarizePctSeries(promptPcts)
	sum.CompletionDiffBand, sum.CompletionMeanDiffPct = summarizePctSeries(compPcts)
	sum.ReasoningDiffBand, sum.ReasoningMeanDiffPct = summarizePctSeries(reasPcts)

	return sum, nil
}

// ToSummaryMetrics converts UsageDiffSummary into tag-keyed metric dicts
// matching summary.go's curated tags.
func (s UsageDiffSummary) ToSummaryMetrics() map[string]any {
	out := map[string]any{
		TagUsageDiscrepancyCount: map[string]any{"avg": float64(s.DiscrepancyCount)},
		TagOSLMismatchCount:      map[string]any{"avg": float64(s.OSLMismatchCount)},
	}
	if s.PromptDiffBand != nil && s.PromptMeanDiffPct != nil {
		out[TagUsagePromptDiffPct] = bandStatDict(*s.PromptMeanDiffPct, *s.PromptDiffBand)
	}
	if s.CompletionDiffBand != nil && s.CompletionMeanDiffPct != nil {
		d := bandStatDict(*s.CompletionMeanDiffPct, *s.CompletionDiffBand)
		out[TagUsageCompletionDiffPct] = d
		out[TagOSLMismatchDiffPct] = bandStatDict(*s.CompletionMeanDiffPct, *s.CompletionDiffBand)
	}
	if s.ReasoningDiffBand != nil && s.ReasoningMeanDiffPct != nil {
		out[TagUsageReasoningDiffPct] = bandStatDict(*s.ReasoningMeanDiffPct, *s.ReasoningDiffBand)
	}
	return out
}

func pairDiff(api, client int) (int, *float64, error) {
	pct, err := TokenDiffPct(api, client)
	delta := api - client
	if err != nil {
		if err == ErrNoMetricValue {
			return delta, nil, nil
		}
		return 0, nil, err
	}
	return delta, &pct, nil
}

func totalDiffPctPtr(api, client int64) *float64 {
	if client == 0 {
		if api == 0 {
			z := 0.0
			return &z
		}
		return nil
	}
	v := float64(api-client) / float64(client) * 100.0
	return &v
}

func summarizePctSeries(vals []float64) (*PercentileBand, *float64) {
	if len(vals) == 0 {
		return nil, nil
	}
	band, err := ComputePercentileBand(vals)
	if err != nil {
		return nil, nil
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	mean := sum / float64(len(vals))
	return &band, &mean
}

func bandStatDict(mean float64, b PercentileBand) map[string]any {
	return map[string]any{
		"avg":   mean,
		"p50":   b.P50,
		"p95":   b.P95,
		"p99":   b.P99,
		"count": b.Count,
	}
}
