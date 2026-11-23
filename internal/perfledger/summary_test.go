// fak-test:runtime fast est=10ms lane=default
// fak-test:justify why=contract when=changed:internal/perfledger/**
package perfledger

import (
	"encoding/json"
	"math"
	"reflect"
	"sort"
	"testing"
)

func summaryKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func TestSummaryProjectionKeepsOnlyCuratedTags(t *testing.T) {
	payload := map[string]any{
		"metrics": map[string]any{
			"request_latency":         map[string]any{"avg": 120.5, "p99": 300.0, "unit": "ms", "count": 40},
			"output_token_throughput": map[string]any{"avg": 55.25, "unit": "tokens/sec"},
			// Internal and experimental tags are not part of the curated view.
			"credit_drop_latency": map[string]any{"avg": 1.0},
			"thinking_efficiency": map[string]any{"avg": 0.5},
			"not_a_metric_at_all": "string",
			"request_count":       map[string]any{"avg": 38.0},
			"error_request_count": map[string]any{"avg": 2.0},
		},
		"unrelated_top_level": 7,
	}
	got := ProjectSummary(payload, nil)
	want := []string{"error_rate", "error_request_count", "output_token_throughput", "request_count", "request_latency", "total_requests"}
	if keys := summaryKeys(got); !reflect.DeepEqual(keys, want) {
		t.Fatalf("projected keys = %v, want %v", keys, want)
	}
	wantLatency := map[string]any{"avg": 120.5, "p99": 300.0, "unit": "ms", "count": 40}
	if !reflect.DeepEqual(got["request_latency"], wantLatency) {
		t.Fatalf("request_latency = %v, want the stats unchanged %v", got["request_latency"], wantLatency)
	}
	if got["total_requests"] != int64(40) || got["error_rate"] != 0.05 {
		t.Fatalf("total_requests/error_rate = %v/%v, want 40/0.05", got["total_requests"], got["error_rate"])
	}

	// The projection copies: editing it must not reach back into the payload.
	got["request_latency"].(map[string]any)["avg"] = -1.0
	if payload["metrics"].(map[string]any)["request_latency"].(map[string]any)["avg"] != 120.5 {
		t.Fatal("editing the projection changed the source payload")
	}
}

func TestSummaryProjectionNaNGateDropsWholeTag(t *testing.T) {
	payload := map[string]any{
		"request_latency":         map[string]any{"avg": 120.0, "p99": math.NaN(), "unit": "ms"},
		"time_to_first_token":     map[string]any{"avg": math.Inf(1)},
		"inter_token_latency":     map[string]any{"avg": 9.0, "min": math.Inf(-1)},
		"output_token_throughput": map[string]any{"avg": json.Number("1e999")},
		"request_throughput":      map[string]any{"avg": float32(math.NaN())},
		// A non-finite-looking string and a boolean are not numeric stats.
		"total_token_throughput": map[string]any{"avg": 70.0, "unit": "NaN", "streaming": true},
		"goodput":                map[string]any{"avg": json.Number("12.5")},
	}
	got := ProjectSummary(payload, nil)
	want := []string{"goodput", "total_token_throughput"}
	if keys := summaryKeys(got); !reflect.DeepEqual(keys, want) {
		t.Fatalf("projected keys = %v, want only the fully finite tags %v", keys, want)
	}
	// request_latency had one bad stat among good ones: none of it survives.
	if _, ok := got["request_latency"]; ok {
		t.Fatal("a tag with one NaN stat was written half-scrubbed instead of dropped")
	}
}

func TestSummaryProjectionAcceptsThreePayloadShapes(t *testing.T) {
	stats := map[string]any{"avg": 42.0}
	shapes := map[string]map[string]any{
		"live_tag_map": {"metrics": map[string]any{"request_latency": stats}},
		"legacy_list": {"metrics": []any{
			map[string]any{"tag": "request_latency", "avg": 42.0},
			"not a metric",
			map[string]any{"avg": 1.0},
			map[string]any{"tag": 7, "avg": 1.0},
		}},
		"top_level_tags": {"request_latency": stats},
	}
	for name, payload := range shapes {
		got := ProjectSummary(payload, nil)
		latency, ok := got["request_latency"].(map[string]any)
		if !ok || latency["avg"] != 42.0 || len(got) != 1 {
			t.Fatalf("%s: projection = %v, want only request_latency with avg 42", name, got)
		}
	}
	for name, payload := range map[string]map[string]any{
		"nil":               nil,
		"empty":             {},
		"metrics_is_scalar": {"metrics": 3},
	} {
		if got := ProjectSummary(payload, nil); got == nil || len(got) != 0 {
			t.Fatalf("%s payload projected to %v, want an empty non-nil map", name, got)
		}
	}
}

func TestSummaryProjectionDerivedScalars(t *testing.T) {
	tagAvg := func(pairs ...any) map[string]any {
		m := map[string]any{}
		for i := 0; i < len(pairs); i += 2 {
			m[pairs[i].(string)] = map[string]any{"avg": pairs[i+1]}
		}
		return m
	}
	cases := []struct {
		name      string
		payload   map[string]any
		wantTotal any
		wantRate  any
	}{
		{"successes_plus_errors", tagAvg("request_count", 95.0, "error_request_count", 5.0), int64(100), 0.05},
		{"all_failed", tagAvg("error_request_count", 12.0), int64(12), 1.0},
		{"no_errors_counter", tagAvg("request_count", 50.0), int64(50), 0.0},
		// The live path has no error counter; the completed total carries it.
		{"completed_total_preferred", tagAvg("request_count", 90.0, "completed_request_count", 100.0), int64(100), 0.0},
		// The published percent wins over the ratio, converted to a fraction.
		{"published_percent_wins", tagAvg("request_count", 90.0, "error_request_count", 10.0, "request_error_rate", 12.5), int64(100), 0.125},
		{"percent_clamped_high", tagAvg("request_count", 1.0, "request_error_rate", 250.0), int64(1), 1.0},
		{"percent_clamped_low", tagAvg("request_count", 1.0, "request_error_rate", -3.0), int64(1), 0.0},
		// A zero or unknown completed total falls back to the sum.
		{"completed_zero_falls_back", tagAvg("request_count", 90.0, "completed_request_count", 0.0), int64(90), 0.0},
		{"completed_nan_falls_back", tagAvg("request_count", 90.0, "completed_request_count", math.NaN()), int64(90), 0.0},
		{"small_integer_kinds_count", tagAvg("request_count", int16(30), "error_request_count", uint8(10)), int64(40), 0.25},
		{"count_past_int64_is_held_at_the_bound", tagAvg("request_count", 1e19), int64(math.MaxInt64), 0.0},
		// Counters from the tags win over stale top-level scalars.
		{"tags_beat_top_level_scalars", map[string]any{
			"metrics":       tagAvg("request_count", 95.0, "error_request_count", 5.0),
			"error_rate":    0.9,
			"request_count": 7,
		}, int64(100), 0.05},
		// A rate with no counters at all: the all-error live case.
		{"rate_without_counters", tagAvg("request_error_rate", 100.0), nil, 1.0},
		// An unknown error count must not become "zero errors".
		{"nan_error_count_drops_rate", tagAvg("request_count", 40.0, "error_request_count", math.NaN()), int64(40), nil},
		{"non_numeric_success_count_drops_rate", tagAvg("request_count", "many", "error_request_count", 4.0), int64(4), nil},
		{"nothing_published", tagAvg("request_latency", 5.0), nil, nil},
		// A finished run's summary keeps only the top-level scalars.
		{"top_level_fallback", map[string]any{"error_rate": 0.25, "request_count": 80}, int64(80), 0.25},
		{"top_level_non_finite_ignored", map[string]any{"error_rate": math.NaN(), "request_count": math.Inf(1), "goodput": map[string]any{"avg": 1.0}}, nil, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ProjectSummary(tc.payload, nil)
			total, hasTotal := got[SummaryTotalRequests]
			rate, hasRate := got[SummaryErrorRate]
			if (tc.wantTotal == nil) == hasTotal || (hasTotal && total != tc.wantTotal) {
				t.Fatalf("total_requests = %v (present %v), want %v", total, hasTotal, tc.wantTotal)
			}
			if (tc.wantRate == nil) == hasRate || (hasRate && rate != tc.wantRate) {
				t.Fatalf("error_rate = %v (present %v), want %v", rate, hasRate, tc.wantRate)
			}
		})
	}
}

func TestSummaryProjectionCustomTagsAndRounding(t *testing.T) {
	payload := map[string]any{
		"request_latency":     map[string]any{"avg": 120.456789, "unit": "ms"},
		"request_count":       map[string]any{"avg": 2099.0},
		"error_request_count": map[string]any{"avg": 1.0},
		"credit_drop_latency": map[string]any{"avg": 3.14159},
	}
	custom := ProjectSummary(payload, []string{"credit_drop_latency"})
	if keys := summaryKeys(custom); !reflect.DeepEqual(keys, []string{"credit_drop_latency", "error_rate", "total_requests"}) {
		t.Fatalf("custom tag list projected %v", keys)
	}

	status := ProjectStatusSummary(payload)
	if got := status["request_latency"].(map[string]any)["avg"]; got != 120.46 {
		t.Fatalf("rounded latency avg = %v, want 120.46", got)
	}
	// 1/2100 = 0.000476190476...: two decimals would report no errors at all.
	if got := status[SummaryErrorRate]; got != 0.000476190 {
		t.Fatalf("rounded error_rate = %v, want 0.00047619 kept at nine decimals", got)
	}
	if got := status[SummaryTotalRequests]; got != int64(2100) {
		t.Fatalf("total_requests = %v, want 2100", got)
	}
	if got := ProjectStatusSummary(nil); got == nil || len(got) != 0 {
		t.Fatalf("status summary of nil = %v, want an empty non-nil map", got)
	}

	tags := DefaultSummaryTags()
	seen := map[string]bool{}
	for _, tag := range tags {
		if seen[tag] {
			t.Fatalf("curated tag %q is listed twice", tag)
		}
		seen[tag] = true
	}
	if len(tags) != 60 || tags[0] != "request_throughput" || tags[len(tags)-1] != "http_req_data_received" {
		t.Fatalf("curated list has %d tags from %q to %q, want the 60 upstream tags", len(tags), tags[0], tags[len(tags)-1])
	}
	for _, tag := range []string{"goodput", "request_latency", "time_to_first_token", "inter_token_latency", "good_request_count",
		"error_request_count", "completed_request_count", "benchmark_duration", "video_peak_memory", "http_req_connection_reused"} {
		if !seen[tag] {
			t.Fatalf("curated list is missing %q", tag)
		}
	}
	for _, tag := range []string{"credit_drop_latency", "thinking_efficiency", "request_error_rate"} {
		if seen[tag] {
			t.Fatalf("curated list includes %q, which is not mirrored", tag)
		}
	}
	tags[0] = "mutated"
	if DefaultSummaryTags()[0] == "mutated" {
		t.Fatal("DefaultSummaryTags handed out the shared list")
	}
}
