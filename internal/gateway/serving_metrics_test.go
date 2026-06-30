package gateway

import (
	"strings"
	"testing"
	"time"
)

func TestServingScrapeEmitterRelabelsVLLMIntoFakSchema(t *testing.T) {
	srv := newTestServer(t)
	emitter := NewServingScrapeEmitter(ServingMetricLabels{
		Worker: "vllm-a",
		Engine: "vllm",
	})

	t0 := time.Unix(100, 0)
	if err := emitter.IngestPrometheusAt(vllmServingFixture("10"), t0); err != nil {
		t.Fatal(err)
	}
	if err := emitter.IngestPrometheusAt(vllmServingFixture("30"), t0.Add(10*time.Second)); err != nil {
		t.Fatal(err)
	}
	srv.SetServingMetricsEmitters(emitter)

	out := srv.renderMetrics()
	labels := `worker="vllm-a",engine="vllm",model="llama-70b"`
	for _, want := range []string{
		`# TYPE fak_serving_time_to_first_token_seconds histogram`,
		`fak_serving_time_to_first_token_seconds_bucket{` + labels + `,le="0.1"} 3`,
		`fak_serving_time_to_first_token_seconds_sum{` + labels + `} 0.8`,
		`fak_serving_time_to_first_token_seconds_count{` + labels + `} 5`,
		`fak_serving_time_per_output_token_seconds_bucket{` + labels + `,le="0.02"} 4`,
		`fak_serving_time_per_output_token_seconds_count{` + labels + `} 5`,
		`fak_serving_num_requests_running{` + labels + `} 4`,
		`fak_serving_num_requests_waiting{` + labels + `} 7`,
		`fak_serving_gpu_cache_usage_perc{` + labels + `} 0.75`,
		`fak_serving_gpu_prefix_cache_hit_rate{` + labels + `} 0.6`,
		`fak_serving_goodput_requests_per_second{` + labels + `} 2`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("serving scrape surface missing %q\n--- metrics ---\n%s", want, out)
		}
	}
	if strings.Contains(out, "vllm:time_to_first_token_seconds") {
		t.Fatalf("upstream vLLM metric name leaked instead of relabeling:\n%s", out)
	}
	for _, family := range []string{
		"fak_serving_time_to_first_token_seconds",
		"fak_serving_time_per_output_token_seconds",
		"fak_serving_goodput_requests_per_second",
		"fak_serving_num_requests_running",
		"fak_serving_num_requests_waiting",
		"fak_serving_gpu_cache_usage_perc",
		"fak_serving_gpu_prefix_cache_hit_rate",
	} {
		if got := strings.Count(out, "# HELP "+family+" "); got != 1 {
			t.Fatalf("%s HELP count = %d, want 1", family, got)
		}
		if got := strings.Count(out, "# TYPE "+family+" "); got != 1 {
			t.Fatalf("%s TYPE count = %d, want 1", family, got)
		}
	}
}

func TestNativeServingEmitterWritesSameSchema(t *testing.T) {
	srv := newTestServer(t)
	native := NewNativeServingMetricsEmitter(ServingMetricLabels{
		Worker: "native-a",
		Engine: "native",
		Model:  "glm-5.2",
	})
	native.ObserveTTFT(200 * time.Millisecond)
	native.ObserveTPOT(20 * time.Millisecond)
	native.SetGoodputRequestsPerSecond(3.5)
	native.SetQueue(2, 5)
	native.SetKVUtilization(0.81)
	native.SetPrefixCacheHitRate(0.42)
	srv.SetServingMetricsEmitters(native)

	out := srv.renderMetrics()
	labels := `worker="native-a",engine="native",model="glm-5.2"`
	for _, want := range []string{
		`fak_serving_time_to_first_token_seconds_count{` + labels + `} 1`,
		`fak_serving_time_to_first_token_seconds_bucket{` + labels + `,le="0.25"} 1`,
		`fak_serving_time_per_output_token_seconds_count{` + labels + `} 1`,
		`fak_serving_inter_token_latency_seconds_count{` + labels + `} 1`,
		`fak_serving_goodput_requests_per_second{` + labels + `} 3.5`,
		`fak_serving_num_requests_running{` + labels + `} 2`,
		`fak_serving_num_requests_waiting{` + labels + `} 5`,
		`fak_serving_gpu_cache_usage_perc{` + labels + `} 0.81`,
		`fak_serving_gpu_prefix_cache_hit_rate{` + labels + `} 0.42`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("native serving surface missing %q\n--- metrics ---\n%s", want, out)
		}
	}
}

func TestGatewayNativeInferenceFeedsServingSchema(t *testing.T) {
	srv := newTestServer(t)
	ctl := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, TokenBudget: 1000, MaxWaiting: 10})
	if got := ctl.Offer(SeqRequest{TraceID: "run", Tokens: 10}); got != VerdictAdmitted {
		t.Fatalf("first offer = %s, want admitted", got)
	}
	if got := ctl.Offer(SeqRequest{TraceID: "wait", Tokens: 10}); got != VerdictQueued {
		t.Fatalf("second offer = %s, want queued", got)
	}
	srv.SetAdmissionController(ctl)
	srv.metrics.observeInferenceTimed(100, 10, 0, 0, "stop", time.Second, 100*time.Millisecond)

	out := srv.renderMetrics()
	labels := `worker="local",engine="test",model="test-model"`
	for _, want := range []string{
		`fak_serving_time_to_first_token_seconds_count{` + labels + `} 1`,
		`fak_serving_time_per_output_token_seconds_count{` + labels + `} 1`,
		`fak_serving_goodput_requests_per_second{` + labels + `} 1`,
		`fak_serving_num_requests_running{` + labels + `} 1`,
		`fak_serving_num_requests_waiting{` + labels + `} 1`,
		`fak_serving_gpu_prefix_cache_hit_rate{` + labels + `} 0`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("gateway native serving schema missing %q\n--- metrics ---\n%s", want, out)
		}
	}
}

func vllmServingFixture(success string) string {
	return `# HELP vllm:time_to_first_token_seconds TTFT
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="llama-70b",le="0.1"} 3
vllm:time_to_first_token_seconds_bucket{model_name="llama-70b",le="0.5"} 5
vllm:time_to_first_token_seconds_bucket{model_name="llama-70b",le="+Inf"} 5
vllm:time_to_first_token_seconds_sum{model_name="llama-70b"} 0.8
vllm:time_to_first_token_seconds_count{model_name="llama-70b"} 5
# HELP vllm:time_per_output_token_seconds TPOT
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{model_name="llama-70b",le="0.02"} 4
vllm:time_per_output_token_seconds_bucket{model_name="llama-70b",le="+Inf"} 5
vllm:time_per_output_token_seconds_sum{model_name="llama-70b"} 0.11
vllm:time_per_output_token_seconds_count{model_name="llama-70b"} 5
vllm:num_requests_running{model_name="llama-70b"} 4
vllm:num_requests_waiting{model_name="llama-70b"} 7
vllm:gpu_cache_usage_perc{model_name="llama-70b"} 0.75
vllm:gpu_prefix_cache_hit_rate{model_name="llama-70b"} 0.6
vllm:request_success_total{model_name="llama-70b"} ` + success + "\n"
}
