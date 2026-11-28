package gateway

import (
	"fmt"
	"testing"
)

// fak-test:runtime fast est=1s
func TestPromUnitsPriorityChainExplicitUnitBeatsHelpAndSuffix(t *testing.T) {
	// Explicit # UNIT overrides contradictory Help scale/text and metric suffix.
	got := InferPrometheusUnit(PromMetricMeta{
		Name:         "vllm:kv_cache_usage_bytes",
		Type:         "gauge",
		ExplicitUnit: "ms",
		Help:         "Measured in seconds (0-100)",
	})
	if got != PromUnitMilliseconds {
		t.Fatalf("explicit unit priority: got %q, want %q", got, PromUnitMilliseconds)
	}

	// OpenMetrics # UNIT header parser normalizes unit aliases cleanly.
	metric, unit, ok := ParseOpenMetricsUnitLine("# UNIT http_request_duration_seconds s")
	if !ok || metric != "http_request_duration_seconds" || unit != PromUnitSeconds {
		t.Fatalf("ParseOpenMetricsUnitLine = (%q, %q, %v), want (http_request_duration_seconds, seconds, true)", metric, unit, ok)
	}
	if _, _, ok := ParseOpenMetricsUnitLine("# TYPE foo counter"); ok {
		t.Fatal("non-UNIT comment line must not parse as # UNIT")
	}
}

// fak-test:runtime fast est=1s
func TestPromUnitsHelpScaleAndTextPriority(t *testing.T) {
	// Scale cue "(0-1)" beats both description text and misleading "_perc" suffix.
	got := InferPrometheusUnit(PromMetricMeta{
		Name: "vllm:kv_cache_usage_perc",
		Type: "gauge",
		Help: "KV-cache usage (0-1) reported in percent equivalent",
	})
	if got != PromUnitRatio {
		t.Fatalf("scale cue (0-1): got %q, want %q", got, PromUnitRatio)
	}

	// Scale cue "(0-100)" beats "_ratio" suffix.
	got = InferPrometheusUnit(PromMetricMeta{
		Name: "gpu_utilization_ratio",
		Type: "gauge",
		Help: "SM utilization (0-100)",
	})
	if got != PromUnitPercent {
		t.Fatalf("scale cue (0-100): got %q, want %q", got, PromUnitPercent)
	}

	// Description text beats suffix; longer unit phrase "in milliseconds" beats "in seconds".
	got = InferPrometheusUnit(PromMetricMeta{
		Name: "queue_wait_total",
		Type: "gauge",
		Help: "Queue wait time in milliseconds",
	})
	if got != PromUnitMilliseconds {
		t.Fatalf("help text longest phrase: got %q, want %q", got, PromUnitMilliseconds)
	}
}

// fak-test:runtime fast est=1s
func TestPromUnitsSuffixCounterRatioInfoHistogramRules(t *testing.T) {
	cases := []struct {
		name string
		meta PromMetricMeta
		want PromUnit
	}{
		{
			name: "longest suffix _milliseconds beats _seconds",
			meta: PromMetricMeta{Name: "prefill_latency_milliseconds", Type: "gauge"},
			want: PromUnitMilliseconds,
		},
		{
			name: "counter _seconds_total resolves underlying seconds",
			meta: PromMetricMeta{Name: "process_cpu_seconds_total", Type: "counter"},
			want: PromUnitSeconds,
		},
		{
			name: "counter _bytes_total resolves underlying bytes",
			meta: PromMetricMeta{Name: "network_transmit_bytes_total", Type: "counter"},
			want: PromUnitBytes,
		},
		{
			name: "counter _tokens_total resolves tokens",
			meta: PromMetricMeta{Name: "vllm:prompt_tokens_total", Type: "counter"},
			want: PromUnitTokens,
		},
		{
			name: "generic _total counter resolves count",
			meta: PromMetricMeta{Name: "cache_evictions_total", Type: "counter"},
			want: PromUnitCount,
		},
		{
			name: "counter _reqs suffix resolves requests",
			meta: PromMetricMeta{Name: "sglang:num_retracted_reqs", Type: "counter"},
			want: PromUnitRequests,
		},
		{
			name: "typed counter without _total suffix resolves count",
			meta: PromMetricMeta{Name: "vllm:num_preemptions", Type: "counter"},
			want: PromUnitCount,
		},
		{
			name: "ratio suffix",
			meta: PromMetricMeta{Name: "prefix_cache_hit_ratio", Type: "gauge"},
			want: PromUnitRatio,
		},
		{
			name: "info suffix and type",
			meta: PromMetricMeta{Name: "build_info", Type: "gauge"},
			want: PromUnitInfo,
		},
		{
			name: "info type without suffix",
			meta: PromMetricMeta{Name: "vllm_engine_metadata", Type: "info"},
			want: PromUnitInfo,
		},
		{
			name: "histogram _bucket inherits base unit",
			meta: PromMetricMeta{Name: "vllm:time_to_first_token_seconds_bucket", Type: "histogram"},
			want: PromUnitSeconds,
		},
		{
			name: "histogram _sum inherits base unit",
			meta: PromMetricMeta{Name: "vllm:e2e_request_latency_milliseconds_sum", Type: "histogram"},
			want: PromUnitMilliseconds,
		},
		{
			name: "histogram _count reports count",
			meta: PromMetricMeta{Name: "vllm:time_to_first_token_seconds_count", Type: "histogram"},
			want: PromUnitCount,
		},
		{
			name: "unknown unsuffixed gauge",
			meta: PromMetricMeta{Name: "vllm:num_requests_running", Type: "gauge"},
			want: PromUnitUnknown,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := InferPrometheusUnit(tc.meta); got != tc.want {
				t.Fatalf("InferPrometheusUnit(%+v) = %q, want %q", tc.meta, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestPromUnitsBoundedLRUCache(t *testing.T) {
	inf := NewPromUnitInferrer(3)
	for i := 0; i < 10; i++ {
		u := inf.Infer(PromMetricMeta{Name: fmt.Sprintf("metric_%d_seconds", i)})
		if u != PromUnitSeconds {
			t.Fatalf("iter %d got %q, want %q", i, u, PromUnitSeconds)
		}
	}
	inf.mu.Lock()
	cacheLen := len(inf.cache)
	listLen := inf.ll.Len()
	inf.mu.Unlock()
	if cacheLen != 3 || listLen != 3 {
		t.Fatalf("LRU cache grew past capacity 3: map=%d list=%d", cacheLen, listLen)
	}
}
