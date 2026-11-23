// Adapted from ai-dynamo/aiperf
// src/aiperf/kubernetes/crd_models.py@ab610add4e8876fca223197596c728e17365f381
// (Apache-2.0, Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES;
// modified from the original): the curated status-summary projection
// (MetricsSummary.from_metrics). A full per-tag metrics payload is narrowed to
// a fixed list of tags written through unchanged, so a reader reaches every
// metric by summary[tag][stat], plus two derived scalars. A tag carrying any
// non-finite stat is dropped whole rather than written half-scrubbed.
package perfledger

import (
	"encoding/json"
	"math"
)

// Derived top-level keys of a projected summary. They are not metric tags.
const (
	SummaryTotalRequests = "total_requests"
	SummaryErrorRate     = "error_rate"
)

var summaryTags = []string{
	// Throughput
	"request_throughput",
	"output_token_throughput",
	"total_token_throughput",
	"output_token_throughput_per_user",
	"prefill_throughput_per_user",
	"e2e_output_token_throughput",
	"image_throughput",
	"goodput",
	// Latency
	"request_latency",
	"time_to_first_token",
	"time_to_first_output_token",
	"time_to_second_token",
	"inter_token_latency",
	"inter_chunk_latency",
	"image_latency",
	// Tokens and sequence lengths
	"input_sequence_length",
	"output_sequence_length",
	"output_token_count",
	"reasoning_token_count",
	"error_isl",
	"osl_mismatch_diff_pct",
	"osl_mismatch_count",
	"usage_prompt_tokens",
	"usage_completion_tokens",
	"usage_total_tokens",
	"usage_reasoning_tokens",
	"usage_prompt_tokens_diff_pct",
	"usage_completion_tokens_diff_pct",
	"usage_reasoning_tokens_diff_pct",
	"usage_discrepancy_count",
	// Counts and totals
	"request_count",
	"good_request_count",
	"error_request_count",
	"completed_request_count",
	"total_isl",
	"total_osl",
	"total_error_isl",
	"total_output_tokens",
	"total_usage_prompt_tokens",
	"total_usage_completion_tokens",
	"total_usage_total_tokens",
	"total_reasoning_tokens",
	"benchmark_duration",
	"num_images",
	// Video
	"video_inference_time",
	"video_peak_memory",
	// HTTP trace
	"http_req_duration",
	"http_req_total",
	"http_req_waiting",
	"http_req_blocked",
	"http_req_connecting",
	"http_req_dns_lookup",
	"http_req_sending",
	"http_req_receiving",
	"http_req_connection_overhead",
	"http_req_connection_reused",
	"http_req_chunks_sent",
	"http_req_chunks_received",
	"http_req_data_sent",
	"http_req_data_received",
}

// DefaultSummaryTags returns a fresh copy of the curated tag list.
func DefaultSummaryTags() []string {
	return append([]string(nil), summaryTags...)
}

// ProjectSummary narrows a metrics payload to tags (the curated list when
// nil) and adds the derived total_requests and error_rate scalars.
//
// The payload may be {"metrics": {tag: stats}}, {"metrics": [{"tag": tag,
// ...}]} or the tags at the top level. Each kept tag's stats are copied
// through as they are; a tag is kept only if none of its numeric stats is NaN
// or infinite. The result is never nil and never contains a non-finite float
// this function produced.
func ProjectSummary(metrics map[string]any, tags []string) map[string]any {
	out := map[string]any{}
	if len(metrics) == 0 {
		return out
	}
	if tags == nil {
		tags = summaryTags
	}
	byTag := summaryByTag(metrics)
	for _, tag := range tags {
		stats, ok := byTag[tag]
		if !ok || !summaryStatsFinite(stats) {
			continue
		}
		kept := make(map[string]any, len(stats))
		for k, v := range stats {
			kept[k] = v
		}
		out[tag] = kept
	}
	summaryDerivedScalars(out, metrics, byTag)
	return out
}

// ProjectStatusSummary is ProjectSummary over the curated tags, rounded for
// display the way RoundSummaryTree rounds any summary.
func ProjectStatusSummary(metrics map[string]any) map[string]any {
	rounded, _ := RoundSummaryTree(ProjectSummary(metrics, nil), DefaultSummaryFractionKeys()).(map[string]any)
	if rounded == nil {
		return map[string]any{}
	}
	return rounded
}

func summaryByTag(metrics map[string]any) map[string]map[string]any {
	raw, ok := metrics["metrics"]
	if !ok || raw == nil {
		raw = metrics
	}
	out := map[string]map[string]any{}
	switch shaped := raw.(type) {
	case map[string]any:
		for tag, v := range shaped {
			if stats, ok := v.(map[string]any); ok {
				out[tag] = stats
			}
		}
	case []any:
		for _, v := range shaped {
			stats, ok := v.(map[string]any)
			if !ok {
				continue
			}
			if tag, ok := stats["tag"].(string); ok {
				out[tag] = stats
			}
		}
	}
	return out
}

// summaryNumber reads v as a number. Booleans and strings are not numbers,
// so a "unit" string or a flag never gates a tag.
func summaryNumber(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int8:
		return float64(n), true
	case int16:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case uint:
		return float64(n), true
	case uint8:
		return float64(n), true
	case uint16:
		return float64(n), true
	case uint32:
		return float64(n), true
	case uint64:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		if err != nil {
			return math.NaN(), true
		}
		return f, true
	}
	return 0, false
}

func summaryFinite(v any) (float64, bool) {
	f, ok := summaryNumber(v)
	if !ok || math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, false
	}
	return f, true
}

func summaryStatsFinite(stats map[string]any) bool {
	for _, v := range stats {
		if f, ok := summaryNumber(v); ok && (math.IsNaN(f) || math.IsInf(f, 0)) {
			return false
		}
	}
	return true
}

// summaryDerivedScalars adds total_requests (successes plus errors) and
// error_rate (the 0..1 share of that total that failed).
//
// A counter that is absent counts as zero, but one that is present and not a
// finite number is unknown: the error rate is then left out rather than
// computed from a NaN read as zero, which would report no failures.
func summaryDerivedScalars(out, metrics map[string]any, byTag map[string]map[string]any) {
	avg := func(tag string) any { return byTag[tag]["avg"] }
	rc, ec := avg("request_count"), avg("error_request_count")
	successes, rcOK := summaryFinite(rc)
	errors, ecOK := summaryFinite(ec)
	rcBad := rc != nil && !rcOK
	ecBad := ec != nil && !ecOK

	// completed_request_count is successes plus errors already added
	// upstream, and the only total a still-running job reports.
	total := successes + errors
	if completed, ok := summaryFinite(avg("completed_request_count")); ok && completed > 0 {
		total = completed
	}
	if total > 0 {
		out[SummaryTotalRequests] = summaryCount(total)
	}

	if percent, ok := summaryFinite(avg("request_error_rate")); ok {
		out[SummaryErrorRate] = min(1.0, max(0.0, percent/100))
	} else if total > 0 && !rcBad && !ecBad {
		out[SummaryErrorRate] = errors / total
	}

	// A finished run's round-tripped summary keeps only the scalars.
	if _, ok := out[SummaryErrorRate]; !ok {
		if rate, ok := summaryFinite(metrics[SummaryErrorRate]); ok {
			out[SummaryErrorRate] = rate
		}
	}
	if _, ok := out[SummaryTotalRequests]; !ok {
		if count, ok := summaryFinite(metrics["request_count"]); ok {
			out[SummaryTotalRequests] = summaryCount(count)
		}
	}
}

// summaryCount truncates a finite count to an integer. A float past the int64
// range has no defined conversion, so it is held at the nearest bound.
func summaryCount(v float64) int64 {
	switch {
	case v >= math.MaxInt64:
		return math.MaxInt64
	case v <= math.MinInt64:
		return math.MinInt64
	}
	return int64(v)
}
