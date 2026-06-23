package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

func TestHTTPMetricsEndpointExposesGatewayAndKernelCounters(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp SyscallResponse
	code := postJSON(t, ts.URL+"/v1/fak/syscall", SyscallRequest{
		Tool:      "allow_read",
		Arguments: json.RawMessage(`{"x":1}`),
		ReadOnly:  true,
	}, &resp)
	if code != http.StatusOK {
		t.Fatalf("syscall status = %d, want 200", code)
	}
	if resp.Verdict.Kind != "ALLOW" {
		t.Fatalf("syscall verdict = %q, want ALLOW", resp.Verdict.Kind)
	}

	text := getMetrics(t, ts.URL+"/metrics", "")
	for _, want := range []string{
		"# TYPE fak_gateway_up gauge",
		"fak_gateway_up 1",
		`fak_gateway_build_info{version="dev",engine="test",model="test-model",vdso="true"} 1`,
		`fak_gateway_http_requests_total{route="/v1/fak/syscall",method="POST",status="200"} 1`,
		`fak_gateway_http_request_duration_seconds_count{route="/v1/fak/syscall",method="POST",status="200"} 1`,
		`fak_gateway_operations_total{operation="syscall",verdict="ALLOW",reason="",disposition="",by="test"} 1`,
		`fak_gateway_operation_duration_seconds_count{operation="syscall",verdict="ALLOW",reason="",disposition="",by="test"} 1`,
		"fak_kernel_submits_total 1",
		"fak_kernel_engine_calls_total 1",
		"fak_kernel_admitted_total 1",
		"fak_gateway_vdso_hit_ratio 0",
		// The scrape request is excluded from the live-request registry, so an idle
		// gateway reports zero max age while still emitting the metric family.
		"# TYPE fak_gateway_inflight_max_age_seconds gauge",
		"fak_gateway_inflight_max_age_seconds 0",
		"# TYPE fak_gateway_inflight_requests_by_route gauge",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}
}

// TestInflightSnapshotSurfacesLiveRequests exercises the live-request registry
// directly: it must report what is running RIGHT NOW (per route) and the age of
// the oldest in-flight request, neither of which the completion-time histograms
// can show.
func TestInflightSnapshotSurfacesLiveRequests(t *testing.T) {
	m := newGatewayMetrics(time.Now())

	if byRoute, maxAge := m.inflightSnapshot(time.Now()); len(byRoute) != 0 || maxAge != 0 {
		t.Fatalf("idle snapshot = (%v, %v), want (empty, 0)", byRoute, maxAge)
	}

	base := time.Now()
	id1 := m.beginInflight("/v1/chat/completions", base.Add(-5*time.Second))
	id2 := m.beginInflight("/v1/chat/completions", base.Add(-1*time.Second))
	id3 := m.beginInflight("/v1/fak/syscall", base.Add(-2*time.Second))

	byRoute, maxAge := m.inflightSnapshot(base)
	if byRoute["/v1/chat/completions"] != 2 {
		t.Fatalf("chat inflight = %d, want 2", byRoute["/v1/chat/completions"])
	}
	if byRoute["/v1/fak/syscall"] != 1 {
		t.Fatalf("syscall inflight = %d, want 1", byRoute["/v1/fak/syscall"])
	}
	if maxAge < 4.99 || maxAge > 5.01 {
		t.Fatalf("oldest age = %v, want ~5s", maxAge)
	}

	// Retiring the oldest request shifts maxAge to the next-oldest still running.
	m.endInflight(id1)
	if _, maxAge := m.inflightSnapshot(base); maxAge < 1.99 || maxAge > 2.01 {
		t.Fatalf("oldest age after retiring id1 = %v, want ~2s", maxAge)
	}

	m.endInflight(id2)
	m.endInflight(id3)
	if byRoute, maxAge := m.inflightSnapshot(base); len(byRoute) != 0 || maxAge != 0 {
		t.Fatalf("drained snapshot = (%v, %v), want (empty, 0)", byRoute, maxAge)
	}
}

// TestAdjudicationSummaryClassifiesEveryVerdict proves the exit roll-up `fak guard`
// prints buckets each verdict honestly: a DEFER (a non-blocking admit — what an
// inbound tool_result earns on a tool-bearing turn) is "deferred", a REQUIRE_WITNESS
// is "escalated", and ONLY a genuine ERROR (or unknown future kind) is "errored".
// Regression for the live blemish where a healthy `fak guard -- claude` tool-use turn
// reported its proxy_admit DEFER as "1 errored".
func TestAdjudicationSummaryClassifiesEveryVerdict(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	d := time.Millisecond
	m.observeOperation("adjudicate", WireVerdict{Kind: "ALLOW"}, nil, d)
	m.observeOperation("adjudicate", WireVerdict{Kind: "TRANSFORM"}, nil, d)
	m.observeOperation("adjudicate", WireVerdict{Kind: "DENY", Reason: "POLICY_BLOCK"}, nil, d)
	m.observeOperation("proxy_admit", WireVerdict{Kind: "QUARANTINE", Reason: "SECRET"}, nil, d)
	m.observeOperation("proxy_admit", WireVerdict{Kind: "DEFER"}, nil, d) // a normal inbound-result admit
	m.observeOperation("adjudicate", WireVerdict{Kind: "REQUIRE_WITNESS"}, nil, d)
	m.observeOperation("adjudicate", WireVerdict{Kind: "ERROR"}, nil, d)                 // a genuine failure
	m.observeOperation("adjudicate", WireVerdict{Kind: "ALLOW"}, io.ErrUnexpectedEOF, d) // err overrides kind -> ERROR

	sum := m.adjudicationSummary()
	if sum.Total != 8 {
		t.Fatalf("Total = %d, want 8", sum.Total)
	}
	if sum.Allowed != 1 || sum.Transformed != 1 || sum.Denied != 1 || sum.Quarantined != 1 {
		t.Errorf("allow/transform/deny/quarantine = %d/%d/%d/%d, want 1/1/1/1",
			sum.Allowed, sum.Transformed, sum.Denied, sum.Quarantined)
	}
	if sum.Deferred != 1 {
		t.Errorf("Deferred = %d, want 1 (a DEFER must NOT count as errored)", sum.Deferred)
	}
	if sum.Escalated != 1 {
		t.Errorf("Escalated = %d, want 1 (a REQUIRE_WITNESS must NOT count as errored)", sum.Escalated)
	}
	if sum.Errored != 2 { // the explicit ERROR verdict + the err-tagged ALLOW
		t.Errorf("Errored = %d, want 2 (only genuine failures)", sum.Errored)
	}
	if sum.ByReason["POLICY_BLOCK"] != 1 || sum.ByReason["SECRET"] != 1 {
		t.Errorf("ByReason = %v, want POLICY_BLOCK=1 SECRET=1", sum.ByReason)
	}
}

// TestRenderMetricsEmitsLiveInflightSignals proves the scrape surface reflects a
// request that is still running; the case the user hit where a live request was
// otherwise a black box.
func TestRenderMetricsEmitsLiveInflightSignals(t *testing.T) {
	srv := newTestServer(t)

	idle := srv.renderMetrics()
	if !strings.Contains(idle, "fak_gateway_inflight_max_age_seconds 0") {
		t.Fatalf("idle scrape should report 0 max age:\n%s", idle)
	}

	id := srv.metrics.beginInflight("/v1/chat/completions", time.Now().Add(-3*time.Second))
	defer srv.metrics.endInflight(id)

	live := srv.renderMetrics()
	if !strings.Contains(live, `fak_gateway_inflight_requests_by_route{route="/v1/chat/completions"} 1`) {
		t.Fatalf("live scrape missing per-route inflight gauge:\n%s", live)
	}
	if strings.Contains(live, "fak_gateway_inflight_max_age_seconds 0\n") {
		t.Fatalf("live scrape should report a nonzero max age while a request runs:\n%s", live)
	}
}

// TestAdjudicationSummaryReportsProviderCache proves the exit summary `fak guard`
// prints carries the provider prompt-cache reuse: the cumulative cache_read tokens and
// the count of turns that got a hit, folded from the same inference counters /metrics
// exposes — so the guard line can never overstate the saving.
func TestAdjudicationSummaryReportsProviderCache(t *testing.T) {
	m := newGatewayMetrics(time.Now())

	// No turns yet → no cache reuse reported.
	if s := m.adjudicationSummary(); s.CachedPromptTokens != 0 || s.CachedTurns != 0 {
		t.Fatalf("idle summary must report no cache reuse, got %d tok / %d turns", s.CachedPromptTokens, s.CachedTurns)
	}

	// Two served turns: the first reads 4096 tokens from the provider cache, the second
	// reads 1024; a third reads nothing.
	m.observeInference(10, 5, 4096, "end_turn", time.Second)
	m.observeInference(10, 5, 1024, "end_turn", time.Second)
	m.observeInference(900, 5, 0, "end_turn", time.Second)

	s := m.adjudicationSummary()
	if s.CachedPromptTokens != 5120 {
		t.Errorf("CachedPromptTokens = %d, want 5120 (4096+1024)", s.CachedPromptTokens)
	}
	if s.CachedTurns != 2 {
		t.Errorf("CachedTurns = %d, want 2 (the two turns that hit the cache)", s.CachedTurns)
	}
}

// TestInferenceMetricsAccumulateAcrossTurns exercises the model-generation family
// directly: the kernel/vDSO counters stay 0 on a pure chat workload, so this is the
// signal that makes a busy gateway look busy. Two turns must sum the token totals,
// bucket the requests by finish reason, and derive a positive output tok/s.
func TestInferenceMetricsAccumulateAcrossTurns(t *testing.T) {
	m := newGatewayMetrics(time.Now())

	// Idle: the family renders with no per-reason series and a zero derived rate —
	// never a phantom throughput before the first turn.
	idle := renderInference(m)
	if !strings.Contains(idle, "# TYPE fak_gateway_inference_requests_total counter") {
		t.Fatalf("idle inference scrape missing family header:\n%s", idle)
	}
	if !strings.Contains(idle, "fak_gateway_inference_output_tokens_per_second 0\n") {
		t.Fatalf("idle inference scrape should report 0 tok/s:\n%s", idle)
	}
	if !strings.Contains(idle, "fak_gateway_inference_completion_tokens_total 0\n") {
		t.Fatalf("idle inference scrape should report 0 completion tokens:\n%s", idle)
	}

	m.observeInference(100, 40, 8, "stop", 2*time.Second)
	m.observeInference(50, 10, 0, "tool_calls", 1*time.Second)

	live := renderInference(m)
	for _, want := range []string{
		`fak_gateway_inference_requests_total{finish_reason="stop"} 1`,
		`fak_gateway_inference_requests_total{finish_reason="tool_calls"} 1`,
		"fak_gateway_inference_prompt_tokens_total 150",
		"fak_gateway_inference_completion_tokens_total 50",
		"fak_gateway_inference_cached_prompt_tokens_total 8",
		// Only the first turn got a provider cache read (8 tokens); the second got 0.
		"fak_gateway_inference_cached_prompt_hits_total 1",
		// 1 cache hit over 2 served turns.
		"fak_gateway_inference_cached_prompt_hit_ratio 0.5",
		"fak_gateway_inference_duration_seconds_total 3",
		// 50 completion tokens / 3s wall-clock.
		"fak_gateway_inference_output_tokens_per_second 16.666666666666668",
	} {
		if !strings.Contains(live, want) {
			t.Fatalf("inference scrape missing %q\n--- inference ---\n%s", want, live)
		}
	}
}

// TestChatCompletionsPopulatesInferenceMetrics proves the complete() seam wires a
// served /v1/chat/completions turn into the inference family end to end: a single
// turn must show up as one request with its reported completion tokens, the exact
// black box the user hit where a box decoding real tokens scraped all-zero panels.
func TestChatCompletionsPopulatesInferenceMetrics(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
		Model:    "test-model",
		Messages: []agent.Message{{Role: "user", Content: "book me a flight from SFO to JFK"}},
	}, &resp)
	if code != http.StatusOK {
		t.Fatalf("chat status = %d, want 200", code)
	}

	text := getMetrics(t, ts.URL+"/metrics", "")
	for _, want := range []string{
		"# TYPE fak_gateway_inference_requests_total counter",
		// The mock's first turn proposes a tool call (completion=24 tokens).
		`fak_gateway_inference_requests_total{finish_reason="tool_calls"} 1`,
		"fak_gateway_inference_completion_tokens_total 24",
		// The wall-clock counter is emitted on every scrape; the derived tok/s math is
		// proven against explicit durations in TestInferenceMetricsAccumulateAcrossTurns
		// (a mock turn can round to 0s on a coarse OS clock, so we don't assert it here).
		"# TYPE fak_gateway_inference_duration_seconds_total counter",
		"# TYPE fak_gateway_inference_output_tokens_per_second gauge",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("served chat turn missing %q\n--- metrics ---\n%s", want, text)
		}
	}
}

func renderInference(m *gatewayMetrics) string {
	var b strings.Builder
	m.writeInferenceMetrics(&b)
	return b.String()
}

// TestFleetValueMetricsDeriveHeroKPIs pins the hero-axis KPIs the live gateway
// derives from its own kernel counters: turns saved (vDSO dedup + grammar repair),
// context-window pollutions blocked, the pollution rate, and total agent-serving
// wall-clock. These are the per-node ingredients of fak's product headline, and
// kernel/vDSO traffic that does fire them must surface here non-zero.
func TestFleetValueMetricsDeriveHeroKPIs(t *testing.T) {
	var b strings.Builder
	writeFleetValueMetrics(&b, kernel.Counters{
		Submits:     20,
		VDSOHits:    7,
		Transforms:  3,
		Quarantines: 4,
	}, 12.5)
	out := b.String()

	for _, want := range []string{
		"# TYPE fak_gateway_turns_saved_total counter",
		// 7 engine round-trips skipped + 3 retry turns repaired away.
		`fak_gateway_turns_saved_total{mechanism="vdso_dedup"} 7`,
		`fak_gateway_turns_saved_total{mechanism="grammar_repair"} 3`,
		"fak_gateway_context_pollutions_blocked_total 4",
		// 4 quarantines / 20 submissions.
		"fak_gateway_context_pollution_rate 0.2",
		"fak_gateway_agent_seconds_total 12.5",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("fleet-value metrics missing %q\n--- metrics ---\n%s", want, out)
		}
	}

	// An idle kernel (no submissions) must not divide by zero in the pollution rate.
	var idle strings.Builder
	writeFleetValueMetrics(&idle, kernel.Counters{}, 0)
	if !strings.Contains(idle.String(), "fak_gateway_context_pollution_rate 0\n") {
		t.Fatalf("idle pollution rate should be 0:\n%s", idle.String())
	}
}

// TestMetricsEndpointExposesFleetValueFamily proves the hero-KPI family reaches the
// live scrape surface after real served traffic, alongside the kernel counters.
func TestMetricsEndpointExposesFleetValueFamily(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp SyscallResponse
	if code := postJSON(t, ts.URL+"/v1/fak/syscall", SyscallRequest{
		Tool:      "allow_read",
		Arguments: json.RawMessage(`{"x":1}`),
		ReadOnly:  true,
	}, &resp); code != http.StatusOK {
		t.Fatalf("syscall status = %d, want 200", code)
	}

	text := getMetrics(t, ts.URL+"/metrics", "")
	for _, want := range []string{
		"# TYPE fak_gateway_turns_saved_total counter",
		`fak_gateway_turns_saved_total{mechanism="vdso_dedup"} 0`,
		`fak_gateway_turns_saved_total{mechanism="grammar_repair"} 0`,
		"# TYPE fak_gateway_context_pollutions_blocked_total counter",
		"# TYPE fak_gateway_context_pollution_rate gauge",
		"# TYPE fak_gateway_agent_seconds_total counter",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("live scrape missing fleet-value KPI %q\n--- metrics ---\n%s", want, text)
		}
	}
}

func TestRouteForMetricsIncludesAnthropicMessages(t *testing.T) {
	for _, path := range []string{"/v1/messages", "/v1/messages/count_tokens"} {
		if got := routeForMetrics(path); got != path {
			t.Fatalf("routeForMetrics(%q) = %q, want %q", path, got, path)
		}
	}
}

func TestMetricsEndpointUsesGatewayAuth(t *testing.T) {
	abiResetTestServer := func() *Server {
		t.Helper()
		srv := newTestServer(t)
		srv.requireKey = "sekret"
		return srv
	}
	srv := abiResetTestServer()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unauthenticated /metrics status = %d, want 401", r.StatusCode)
	}

	text := getMetrics(t, ts.URL+"/metrics", "Bearer sekret")
	if !strings.Contains(text, "fak_gateway_up 1") {
		t.Fatalf("authenticated /metrics did not return exposition:\n%s", text)
	}
}

func TestDebugVarsEndpointExposesRuntimeGatewayKernelAndMetrics(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp SyscallResponse
	code := postJSON(t, ts.URL+"/v1/fak/syscall", SyscallRequest{
		Tool:      "allow_read",
		Arguments: json.RawMessage(`{"x":1}`),
		ReadOnly:  true,
	}, &resp)
	if code != http.StatusOK {
		t.Fatalf("syscall status = %d, want 200", code)
	}

	var vars debugVarsResponse
	r, err := http.Get(ts.URL + "/debug/vars")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("/debug/vars status = %d, want 200", r.StatusCode)
	}
	if ct := r.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Fatalf("/debug/vars content-type = %q, want application/json", ct)
	}
	if err := json.NewDecoder(r.Body).Decode(&vars); err != nil {
		t.Fatalf("decode /debug/vars: %v", err)
	}

	if !vars.Gateway.Up || vars.Gateway.Engine != "test" || vars.Gateway.Model != "test-model" || !vars.Gateway.VDSO {
		t.Fatalf("gateway vars mismatch: %+v", vars.Gateway)
	}
	if vars.Gateway.StartTimeUnix == 0 || vars.Gateway.UptimeSeconds < 0 || vars.Gateway.InflightRequests < 1 {
		t.Fatalf("gateway timing/inflight vars mismatch: %+v", vars.Gateway)
	}
	if vars.Runtime.GoVersion == "" || vars.Runtime.NumCPU < 1 || vars.Runtime.NumGoroutine < 1 {
		t.Fatalf("runtime vars mismatch: %+v", vars.Runtime)
	}
	if vars.Kernel.Submits != 1 || vars.Kernel.EngineCalls != 1 || vars.Kernel.Admitted != 1 {
		t.Fatalf("kernel counters mismatch: %+v", vars.Kernel)
	}
	if !hasDebugHTTPRow(vars.Metrics.HTTP, "/v1/fak/syscall", "POST", "200") {
		t.Fatalf("/debug/vars missing completed HTTP row for syscall: %+v", vars.Metrics.HTTP)
	}
	if !hasDebugOperationRow(vars.Metrics.Operations, "syscall", "ALLOW") {
		t.Fatalf("/debug/vars missing syscall operation row: %+v", vars.Metrics.Operations)
	}

	text := getMetrics(t, ts.URL+"/metrics", "")
	if !strings.Contains(text, `fak_gateway_http_requests_total{route="/debug/vars",method="GET",status="200"} 1`) {
		t.Fatalf("/debug/vars route was not counted in Prometheus metrics:\n%s", text)
	}
}

func TestDebugVarsEndpointUsesGatewayAuth(t *testing.T) {
	srv := newTestServer(t)
	srv.requireKey = "sekret"
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Get(ts.URL + "/debug/vars")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unauthenticated /debug/vars status = %d, want 401", r.StatusCode)
	}

	req, err := http.NewRequest(http.MethodGet, ts.URL+"/debug/vars", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer sekret")
	r2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer r2.Body.Close()
	if r2.StatusCode != http.StatusOK {
		t.Fatalf("authenticated /debug/vars status = %d, want 200", r2.StatusCode)
	}
	var vars debugVarsResponse
	if err := json.NewDecoder(r2.Body).Decode(&vars); err != nil {
		t.Fatalf("decode authenticated /debug/vars: %v", err)
	}
	if !vars.Gateway.AuthRequired {
		t.Fatalf("gateway auth_required = false, want true: %+v", vars.Gateway)
	}
}

func TestHTTPAccessLogIsStructured(t *testing.T) {
	srv := newTestServer(t)
	var lines []string
	srv.logf = func(format string, args ...any) {
		lines = append(lines, formatLog(format, args...))
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req, err := http.NewRequest(http.MethodGet, ts.URL+"/healthz", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("X-Trace-Id", "trace-123")
	r, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("/healthz status = %d, want 200", r.StatusCode)
	}
	if len(lines) != 1 {
		t.Fatalf("got %d access log lines, want 1: %v", len(lines), lines)
	}
	var ev map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &ev); err != nil {
		t.Fatalf("access log is not JSON: %v\n%s", err, lines[0])
	}
	for k, want := range map[string]any{
		"event":    "gateway_http_request",
		"method":   "GET",
		"route":    "/healthz",
		"path":     "/healthz",
		"status":   float64(http.StatusOK),
		"trace_id": "trace-123",
	} {
		if got := ev[k]; got != want {
			t.Fatalf("access log %s = %#v, want %#v (event=%v)", k, got, want, ev)
		}
	}
	if _, ok := ev["duration_ms"].(float64); !ok {
		t.Fatalf("access log duration_ms missing/non-number: %v", ev)
	}
	if _, ok := ev["bytes"].(float64); !ok {
		t.Fatalf("access log bytes missing/non-number: %v", ev)
	}
}

func TestHTTPTraceIsMintedAndThreaded(t *testing.T) {
	srv := newTestServer(t)
	rec := &eventRecorder{}
	abi.RegisterEmitter(rec)
	var lines []string
	srv.logf = func(format string, args ...any) {
		lines = append(lines, formatLog(format, args...))
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(SyscallRequest{
		Tool:      "allow_read",
		Arguments: json.RawMessage(`{"x":1}`),
		ReadOnly:  true,
	})
	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/fak/syscall", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	r, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("syscall status = %d, want 200", r.StatusCode)
	}
	traceID := r.Header.Get(traceHeader)
	if traceID == "" {
		t.Fatalf("response did not include %s", traceHeader)
	}
	access := findLogEvent(t, lines, "gateway_http_request")
	if access["trace_id"] != traceID {
		t.Fatalf("access log trace_id = %#v, want response trace %q (event=%v)", access["trace_id"], traceID, access)
	}
	op := findLogEvent(t, lines, "gateway_operation")
	for k, want := range map[string]any{
		"operation": "syscall",
		"tool":      "allow_read",
		"trace_id":  traceID,
		"verdict":   "ALLOW",
	} {
		if got := op[k]; got != want {
			t.Fatalf("operation log %s = %#v, want %#v (event=%v)", k, got, want, op)
		}
	}
	seen := false
	for _, emitted := range rec.events {
		if emitted.Call != nil && emitted.Call.TraceID == traceID {
			seen = true
			break
		}
	}
	if !seen {
		t.Fatalf("no emitted kernel event carried minted trace %q: %+v", traceID, rec.events)
	}
}

func getMetrics(t *testing.T, url, auth string) string {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	if auth != "" {
		req.Header.Set("Authorization", auth)
	}
	r, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatal(err)
	}
	if r.StatusCode != http.StatusOK {
		t.Fatalf("GET %s status = %d, body:\n%s", url, r.StatusCode, body)
	}
	ct := r.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "text/plain") {
		t.Fatalf("/metrics content-type = %q, want text/plain", ct)
	}
	return string(body)
}

func formatLog(format string, args ...any) string {
	return fmt.Sprintf(format, args...)
}

func findLogEvent(t *testing.T, lines []string, event string) map[string]any {
	t.Helper()
	for _, line := range lines {
		var ev map[string]any
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("log line is not JSON: %v\n%s", err, line)
		}
		if ev["event"] == event {
			return ev
		}
	}
	t.Fatalf("log event %q not found in %v", event, lines)
	return nil
}

func hasDebugHTTPRow(rows []debugHTTPMetricVars, route, method, status string) bool {
	for _, row := range rows {
		if row.Route == route && row.Method == method && row.Status == status && row.Latency.Count > 0 {
			return true
		}
	}
	return false
}

func hasDebugOperationRow(rows []debugOperationMetricVars, operation, verdict string) bool {
	for _, row := range rows {
		if row.Operation == operation && row.Verdict == verdict && row.Latency.Count > 0 {
			return true
		}
	}
	return false
}
