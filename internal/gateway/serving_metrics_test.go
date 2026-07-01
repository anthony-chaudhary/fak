package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
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
		`fak_serving_inter_token_latency_seconds_bucket{` + labels + `,le="0.015"} 4`,
		`fak_serving_inter_token_latency_seconds_count{` + labels + `} 5`,
		`fak_serving_num_requests_running{` + labels + `} 4`,
		`fak_serving_num_requests_waiting{` + labels + `} 7`,
		`fak_serving_kv_cache_usage_perc{` + labels + `} 0.75`,
		`fak_serving_prefix_cache_hit_rate{` + labels + `} 0.6`,
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
		"fak_serving_inter_token_latency_seconds",
		"fak_serving_goodput_requests_per_second",
		"fak_serving_num_requests_running",
		"fak_serving_num_requests_waiting",
		"fak_serving_kv_cache_usage_perc",
		"fak_serving_prefix_cache_hit_rate",
	} {
		if got := strings.Count(out, "# HELP "+family+" "); got != 1 {
			t.Fatalf("%s HELP count = %d, want 1", family, got)
		}
		if got := strings.Count(out, "# TYPE "+family+" "); got != 1 {
			t.Fatalf("%s TYPE count = %d, want 1", family, got)
		}
	}
}

func TestServingScrapeEmitterRelabelsSGLangIntoFakSchema(t *testing.T) {
	srv := newTestServer(t)
	emitter := NewServingScrapeEmitter(ServingMetricLabels{
		Worker: "sglang-a",
		Engine: "sglang",
	})

	t0 := time.Unix(200, 0)
	if err := emitter.IngestPrometheusAt(sglangServingFixture("20"), t0); err != nil {
		t.Fatal(err)
	}
	if err := emitter.IngestPrometheusAt(sglangServingFixture("44"), t0.Add(12*time.Second)); err != nil {
		t.Fatal(err)
	}
	srv.SetServingMetricsEmitters(emitter)

	out := srv.renderMetrics()
	labels := `worker="sglang-a",engine="sglang",model="meta-llama/Llama-3.1-8B-Instruct"`
	for _, want := range []string{
		`fak_serving_time_to_first_token_seconds_bucket{` + labels + `,le="0.04"} 1`,
		`fak_serving_time_to_first_token_seconds_count{` + labels + `} 6`,
		`fak_serving_time_per_output_token_seconds_bucket{` + labels + `,le="0.02"} 5`,
		`fak_serving_time_per_output_token_seconds_count{` + labels + `} 6`,
		`fak_serving_inter_token_latency_seconds_bucket{` + labels + `,le="0.02"} 5`,
		`fak_serving_inter_token_latency_seconds_count{` + labels + `} 6`,
		`fak_serving_num_requests_running{` + labels + `} 3`,
		`fak_serving_num_requests_waiting{` + labels + `} 9`,
		`fak_serving_kv_cache_usage_perc{` + labels + `} 0.28`,
		`fak_serving_prefix_cache_hit_rate{` + labels + `} 0.125`,
		`fak_serving_goodput_requests_per_second{` + labels + `} 2`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("SGLang serving scrape surface missing %q\n--- metrics ---\n%s", want, out)
		}
	}
	if strings.Contains(out, "sglang:") {
		t.Fatalf("upstream SGLang metric name leaked instead of relabeling:\n%s", out)
	}
}

func TestServingScrapeEmitterKeepsModelRowsSeparate(t *testing.T) {
	emitter := NewServingScrapeEmitter(ServingMetricLabels{
		Worker: "vllm-shared",
		Engine: "vllm",
	})
	if err := emitter.IngestPrometheus(`vllm:num_requests_running{model_name="llama-70b"} 1
vllm:num_requests_running{model_name="mistral-8x7b"} 2
`); err != nil {
		t.Fatal(err)
	}

	rows := emitter.SnapshotServingMetrics()
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want 2: %+v", len(rows), rows)
	}
	got := map[string]float64{}
	for _, row := range rows {
		got[row.Labels.Model] = row.Running.Value
		if row.Labels.Worker != "vllm-shared" || row.Labels.Engine != "vllm" {
			t.Fatalf("labels = %+v, want worker/engine preserved", row.Labels)
		}
	}
	if got["llama-70b"] != 1 || got["mistral-8x7b"] != 2 {
		t.Fatalf("running rows = %+v, want llama=1 mistral=2", got)
	}
}

func TestServingScrapeEmitterScrapesEndpointAndGatewayExposesNonHTTPFamily(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		_, _ = w.Write([]byte(vllmServingFixture("10")))
	}))
	defer upstream.Close()

	emitter := NewServingScrapeEmitter(ServingMetricLabels{
		Worker: "vllm-http",
		Engine: "vllm",
	})
	if err := emitter.ScrapeWithClient(context.Background(), upstream.Client(), upstream.URL); err != nil {
		t.Fatal(err)
	}

	srv := newTestServer(t)
	srv.SetServingMetricsEmitters(emitter)
	gateway := httptest.NewServer(srv.Handler())
	defer gateway.Close()

	out := getMetrics(t, gateway.URL+"/metrics", "")
	labels := `worker="vllm-http",engine="vllm",model="llama-70b"`
	for _, want := range []string{
		`fak_serving_num_requests_running{` + labels + `} 4`,
		`fak_serving_kv_cache_usage_perc{` + labels + `} 0.75`,
		`fak_serving_prefix_cache_hit_rate{` + labels + `} 0.6`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("gateway /metrics scrape missing non-HTTP serving metric %q\n--- metrics ---\n%s", want, out)
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
		`fak_serving_kv_cache_usage_perc{` + labels + `} 0.81`,
		`fak_serving_prefix_cache_hit_rate{` + labels + `} 0.42`,
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
		`fak_serving_prefix_cache_hit_rate{` + labels + `} 0`,
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
# HELP vllm:inter_token_latency_seconds ITL
# TYPE vllm:inter_token_latency_seconds histogram
vllm:inter_token_latency_seconds_bucket{model_name="llama-70b",le="0.015"} 4
vllm:inter_token_latency_seconds_bucket{model_name="llama-70b",le="+Inf"} 5
vllm:inter_token_latency_seconds_sum{model_name="llama-70b"} 0.09
vllm:inter_token_latency_seconds_count{model_name="llama-70b"} 5
vllm:num_requests_running{model_name="llama-70b"} 4
vllm:num_requests_waiting{model_name="llama-70b"} 7
vllm:kv_cache_usage_perc{model_name="llama-70b"} 0.75
vllm:prefix_cache_queries{model_name="llama-70b"} 10
vllm:prefix_cache_hits{model_name="llama-70b"} 6
vllm:request_success_total{model_name="llama-70b"} ` + success + "\n"
}

func sglangServingFixture(completed string) string {
	return `# HELP sglang:token_usage The token usage
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.28
# HELP sglang:cache_hit_rate The cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.125
# HELP sglang:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.9
sglang:time_to_first_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1
sglang:time_to_first_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6
sglang:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 6
# HELP sglang:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE sglang:time_per_output_token_seconds histogram
sglang:time_per_output_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 5
sglang:time_per_output_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6
sglang:time_per_output_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.18
sglang:time_per_output_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 6
# HELP sglang:inter_token_latency_seconds Histogram of inter-token latency in seconds.
# TYPE sglang:inter_token_latency_seconds histogram
sglang:inter_token_latency_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 5
sglang:inter_token_latency_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6
sglang:inter_token_latency_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.18
sglang:inter_token_latency_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 6
# HELP sglang:num_running_reqs The number of running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 3
# HELP sglang:num_queue_reqs The number of requests in the waiting queue
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 9
# HELP sglang:func_latency_seconds Function latency in seconds
# TYPE sglang:func_latency_seconds histogram
sglang:func_latency_seconds_count{name="generate_request"} ` + completed + "\n"
}
