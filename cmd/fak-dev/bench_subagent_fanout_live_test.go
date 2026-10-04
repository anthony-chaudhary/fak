package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

const (
	kvReused    = "fak_gateway_kv_prefix_reused_tokens"
	kvPrompt    = "fak_gateway_kv_prefix_prompt_tokens"
	proxyCached = "fak_gateway_inference_cached_prompt_tokens"
	proxyPrompt = "fak_gateway_inference_prompt_tokens"
)

func TestParsePrometheusCounters(t *testing.T) {
	body := "# HELP fak_gateway_kv_prefix_reused_tokens_total x\n" +
		"# TYPE fak_gateway_kv_prefix_reused_tokens_total counter\n" +
		"fak_gateway_kv_prefix_reused_tokens_total 0\n" +
		"fak_gateway_inference_cached_prompt_tokens_total 109848\n" +
		"fak_gateway_inference_cached_prompt_hits_total 22\n" +
		"other_reused_tokens_total{arm=\"a\"} 100\n" +
		"other_reused_tokens_total{arm=\"b\"} 50\n" +
		"bare_reused_tokens 7\n" +
		"garbage_line\n"
	got := parsePrometheusCounters(body)
	want := map[string]int64{
		kvReused:    0,
		proxyCached: 109848,
		"fak_gateway_inference_cached_prompt_hits": 22,
		"other_reused_tokens":                      150,
		"bare_reused_tokens":                       7,
	}
	if len(got) != len(want) {
		t.Fatalf("families = %v, want %v", got, want)
	}
	for k, v := range want {
		if g, ok := got[k]; !ok || g != v {
			t.Errorf("%s = (%d, %v), want %d", k, g, ok, v)
		}
	}
}

func TestSelectReuseDelta(t *testing.T) {
	t.Run("ProxyPathReadsCachedPromptDelta", func(t *testing.T) {
		before := map[string]int64{kvReused: 0, kvPrompt: 0, proxyCached: 109848, proxyPrompt: 95583}
		after := map[string]int64{kvReused: 0, kvPrompt: 0, proxyCached: 109848 + 1167, proxyPrompt: 95583 + 461}
		sc, err := selectReuseDelta(before, after)
		if err != nil {
			t.Fatalf("selectReuseDelta: %v", err)
		}
		if sc.Counter != proxyCached || sc.Delta != 1167 {
			t.Fatalf("got counter %q delta %d, want %q 1167", sc.Counter, sc.Delta, proxyCached)
		}
		if sc.OfferedTokens != 1167+461 {
			t.Errorf("offered = %d, want %d (proxy prompt counter excludes cached)", sc.OfferedTokens, 1167+461)
		}
	})
	t.Run("InKernelPathPromptIncludesReused", func(t *testing.T) {
		before := map[string]int64{kvReused: 10, kvPrompt: 100, proxyCached: 0, proxyPrompt: 0}
		after := map[string]int64{kvReused: 310, kvPrompt: 500, proxyCached: 0, proxyPrompt: 0}
		sc, err := selectReuseDelta(before, after)
		if err != nil {
			t.Fatalf("selectReuseDelta: %v", err)
		}
		if sc.Counter != kvReused || sc.Delta != 300 || sc.OfferedTokens != 400 {
			t.Fatalf("got %+v, want kv counter delta 300 offered 400", sc)
		}
	})
	t.Run("BothMovedReadsOneFamilyOnly", func(t *testing.T) {
		before := map[string]int64{kvReused: 0, proxyCached: 0}
		after := map[string]int64{kvReused: 300, proxyCached: 300}
		sc, err := selectReuseDelta(before, after)
		if err != nil {
			t.Fatalf("selectReuseDelta: %v", err)
		}
		if sc.Counter != kvReused || sc.Delta != 300 {
			t.Fatalf("got %q delta %d, want %q 300 (never summed across families)", sc.Counter, sc.Delta, kvReused)
		}
	})
	t.Run("MeasuredZeroIsReported", func(t *testing.T) {
		before := map[string]int64{kvReused: 5, proxyCached: 9}
		after := map[string]int64{kvReused: 5, proxyCached: 9}
		sc, err := selectReuseDelta(before, after)
		if err != nil {
			t.Fatalf("selectReuseDelta: %v", err)
		}
		if sc.Delta != 0 || sc.Counter != kvReused {
			t.Fatalf("got %+v, want measured zero on %q", sc, kvReused)
		}
	})
	t.Run("ZeroReusePicksActivePath", func(t *testing.T) {
		before := map[string]int64{kvReused: 0, kvPrompt: 0, proxyCached: 50, proxyPrompt: 1000}
		after := map[string]int64{kvReused: 0, kvPrompt: 0, proxyCached: 50, proxyPrompt: 1400}
		sc, err := selectReuseDelta(before, after)
		if err != nil {
			t.Fatalf("selectReuseDelta: %v", err)
		}
		if sc.Counter != proxyCached || sc.Delta != 0 || sc.OfferedTokens != 400 {
			t.Fatalf("got %+v, want proxy family measured zero over 400 offered", sc)
		}
	})
	t.Run("GenericFamilyFallback", func(t *testing.T) {
		sc, err := selectReuseDelta(map[string]int64{"x_reused_tokens": 1}, map[string]int64{"x_reused_tokens": 4})
		if err != nil {
			t.Fatalf("selectReuseDelta: %v", err)
		}
		if sc.Counter != "x_reused_tokens" || sc.Delta != 3 || sc.OfferedTokens != 0 {
			t.Fatalf("got %+v, want generic delta 3 with no denominator", sc)
		}
	})
	t.Run("CounterResetRefuses", func(t *testing.T) {
		_, err := selectReuseDelta(map[string]int64{proxyCached: 100}, map[string]int64{proxyCached: 10})
		if !errors.Is(err, ErrReuseCounterReset) {
			t.Fatalf("err = %v, want ErrReuseCounterReset", err)
		}
	})
	t.Run("AbsentRefuses", func(t *testing.T) {
		_, err := selectReuseDelta(map[string]int64{"x": 1}, map[string]int64{"x": 2})
		if !errors.Is(err, ErrReuseCounterAbsent) {
			t.Fatalf("err = %v, want ErrReuseCounterAbsent", err)
		}
	})
	t.Run("OneSidedScrapeRefuses", func(t *testing.T) {
		_, err := selectReuseDelta(map[string]int64{}, map[string]int64{proxyCached: 10})
		if !errors.Is(err, ErrReuseScrapeIncomplete) {
			t.Fatalf("err = %v, want ErrReuseScrapeIncomplete", err)
		}
	})
}

// fanoutProxyGateway fakes a fak gateway on the proxy path: /props is absent,
// /metrics carries a large pre-existing running total, and each completion
// adds cached and uncached prompt tokens.
func fanoutProxyGateway(t *testing.T) (*httptest.Server, *int64, *int64) {
	t.Helper()
	var completions, chatHits int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/metrics":
			c := atomic.LoadInt64(&completions)
			fmt.Fprintf(w, "fak_gateway_kv_prefix_reused_tokens_total 0\n"+
				"fak_gateway_inference_cached_prompt_tokens_total %d\n"+
				"fak_gateway_inference_prompt_tokens_total %d\n", 109848+300*c, 95583+100*c)
		case "/v1/chat/completions", "/v1/completions":
			if r.URL.Path == "/v1/chat/completions" {
				atomic.AddInt64(&chatHits, 1)
			}
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"text\":\"a\"}]}\n\ndata: {\"choices\":[{\"text\":\"b\"}]}\n\ndata: [DONE]\n\n")
			atomic.AddInt64(&completions, 1)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, &completions, &chatHits
}

func fanoutUpstreamProps(t *testing.T, slotCtx int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/props" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprintf(w, `{"total_slots":4,"default_generation_settings":{"n_ctx":%d}}`, slotCtx)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func liveFanoutConfig(endpoint string) *FanoutBenchConfig {
	return &FanoutBenchConfig{
		Model:          "m",
		Quantization:   "q",
		MemoryFraction: FixedMemoryFraction,
		FanoutSweep:    []int{4},
		Arms:           []string{ArmFAK},
		PrefixTokens:   64,
		SuffixTokens:   16,
		DecodeTokens:   4,
		Trials:         1,
		Live:           true,
		Endpoints:      map[string]string{ArmFAK: endpoint},
		RequestShape:   RequestShapeChatSystem,
		Schedule:       ScheduleParentFirst,
	}
}

// fak-test:runtime fast est=1s
func TestFanoutLiveCellReadsProxyCounterDelta(t *testing.T) {
	gw, completions, chatHits := fanoutProxyGateway(t)
	up := fanoutUpstreamProps(t, 32768)
	cfg := liveFanoutConfig(gw.URL)
	cfg.UpstreamPropsURL = up.URL

	receipt, err := NewFanoutBenchmarkHarness(cfg).Run(context.Background())
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	res := receipt.Results[0]
	if res.Error != "" {
		t.Fatalf("cell error: %s", res.Error)
	}
	if got := atomic.LoadInt64(completions); got != 4 {
		t.Fatalf("completions = %d, want 4", got)
	}
	if got := atomic.LoadInt64(chatHits); got != 4 {
		t.Errorf("chat-system shape must use /v1/chat/completions, got %d chat hits", got)
	}
	if res.ReuseObservationSource != FanoutReuseSourcePrometheus || res.ReuseScrape == nil {
		t.Fatalf("source = %q scrape = %v, want prometheus delta", res.ReuseObservationSource, res.ReuseScrape)
	}
	if res.ReuseScrape.Counter != proxyCached {
		t.Errorf("counter = %q, want %q", res.ReuseScrape.Counter, proxyCached)
	}
	if res.ObservedReuseTokens != 1200 {
		t.Errorf("observed reuse = %d, want the per-cell delta 1200, not the running total", res.ObservedReuseTokens)
	}
	if res.ReuseScrape.OfferedTokens != 1600 || res.ObservedReuseMultiplier != 4 {
		t.Errorf("offered %d multiplier %v, want 1600 and 4", res.ReuseScrape.OfferedTokens, res.ObservedReuseMultiplier)
	}
	if res.ReuseDivergence {
		t.Error("observed reuse must not flag divergence")
	}
	if res.Schedule != ScheduleParentFirst || res.RequestsIssued != 4 {
		t.Errorf("schedule %q requests %d, want parent-first 4", res.Schedule, res.RequestsIssued)
	}
}

// fak-test:runtime fast est=1s
func TestFanoutPropsFallback(t *testing.T) {
	t.Run("UpstreamPropsAdmitsProxyGateway", func(t *testing.T) {
		gw, _, _ := fanoutProxyGateway(t)
		up := fanoutUpstreamProps(t, 32768)
		h := NewFanoutBenchmarkHarness(&FanoutBenchConfig{UpstreamPropsURL: up.URL})
		sc, err := h.verifyServedGeometry(context.Background(), ArmFAK, gw.URL, 64, 16, 4)
		if err != nil {
			t.Fatalf("verify: %v", err)
		}
		if !sc.ControlArmServed || sc.CapacityDeclared || sc.ObservedPerSlotTokens != 32768 || sc.ObservedTotalSlots != 4 {
			t.Fatalf("got %+v, want observed upstream capacity 32768 x4", sc)
		}
		if sc.PrimaryProbe == "" {
			t.Error("the unusable primary /props must be recorded")
		}
	})
	t.Run("GatewayPropsWithoutSlotCapacityFallsBack", func(t *testing.T) {
		gw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, `{"default_generation_settings":{"n_ctx":0}}`)
		}))
		t.Cleanup(gw.Close)
		up := fanoutUpstreamProps(t, 8192)
		h := NewFanoutBenchmarkHarness(&FanoutBenchConfig{UpstreamPropsURL: up.URL})
		sc, err := h.verifyServedGeometry(context.Background(), ArmFAK, gw.URL, 64, 16, 4)
		if err != nil || sc.ObservedPerSlotTokens != 8192 {
			t.Fatalf("got %+v err %v, want upstream 8192", sc, err)
		}
	})
	t.Run("UpstreamAlsoMissingFailsClosed", func(t *testing.T) {
		gw, completions, _ := fanoutProxyGateway(t)
		dead := httptest.NewServer(http.NotFoundHandler())
		t.Cleanup(dead.Close)
		cfg := liveFanoutConfig(gw.URL)
		cfg.UpstreamPropsURL = dead.URL
		receipt, err := NewFanoutBenchmarkHarness(cfg).Run(context.Background())
		if err != nil {
			t.Fatalf("run: %v", err)
		}
		sc := receipt.Results[0].ServeCompleteness
		if sc == nil || sc.ControlArmServed || sc.Refusal != FanoutServeRefuseCapacityUnobserved || !sc.EvidenceGap {
			t.Fatalf("got %+v, want fail-closed CapacityUnobserved evidence gap", sc)
		}
		if got := atomic.LoadInt64(completions); got != 0 {
			t.Errorf("unobserved capacity must not spend requests, got %d", got)
		}
	})
	t.Run("DeclaredCapacityIsRecordedAsDeclared", func(t *testing.T) {
		gw, _, _ := fanoutProxyGateway(t)
		h := NewFanoutBenchmarkHarness(&FanoutBenchConfig{DeclaredPerSlotCtx: 4096})
		sc, err := h.verifyServedGeometry(context.Background(), ArmFAK, gw.URL, 64, 16, 4)
		if err != nil {
			t.Fatalf("verify: %v", err)
		}
		if !sc.ControlArmServed || !sc.CapacityDeclared || sc.DeclaredPerSlotTokens != 4096 || sc.ObservedPerSlotTokens != 0 {
			t.Fatalf("got %+v, want declared 4096 with observed 0", sc)
		}
	})
	t.Run("DeclaredTooSmallOverflows", func(t *testing.T) {
		gw, _, _ := fanoutProxyGateway(t)
		h := NewFanoutBenchmarkHarness(&FanoutBenchConfig{DeclaredPerSlotCtx: 50})
		sc, err := h.verifyServedGeometry(context.Background(), ArmFAK, gw.URL, 64, 16, 4)
		if err == nil || sc.Refusal != FanoutServeRefusePromptOverflow || !sc.CapacityDeclared {
			t.Fatalf("got %+v err %v, want declared PromptOverflow", sc, err)
		}
	})
}

func TestFanoutSweepSubset(t *testing.T) {
	cases := []struct {
		sweep      []int
		scope      string
		violations int
	}{
		{[]int{1, 4, 8, 16, 32}, FanoutSweepFull, 0},
		{[]int{32, 16, 8, 4, 1}, FanoutSweepFull, 0},
		{[]int{4}, FanoutSweepSmokeSubset, 0},
		{[]int{1, 4}, FanoutSweepSmokeSubset, 0},
		{[]int{2, 4}, FanoutSweepNonCanonical, 1},
		{[]int{64}, FanoutSweepNonCanonical, 1},
	}
	for _, c := range cases {
		scope, v := classifyFanoutSweep(c.sweep)
		if scope != c.scope || len(v) != c.violations {
			t.Errorf("%v: scope %q violations %d, want %q %d", c.sweep, scope, len(v), c.scope, c.violations)
		}
	}

	arms := "no_reuse,sglang,vllm,fak"
	t.Run("SmokeSubsetRunsButIsPartial", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runBenchSubagentFanout(&stdout, &stderr, []string{"-fanout", "1,4", "-arms", arms, "-trials", "1", "-decode-tokens", "2", "-json"})
		if code != 0 {
			t.Fatalf("smoke subset must run, exit %d stderr=%s", code, stderr.String())
		}
		var r SubagentFanoutReceipt
		if err := json.Unmarshal(stdout.Bytes(), &r); err != nil {
			t.Fatalf("receipt: %v", err)
		}
		if r.Contract.Compliant || !r.Contract.Partial || r.Contract.SweepScope != FanoutSweepSmokeSubset || r.Contract.FanoutSweepVerified {
			t.Fatalf("contract = %+v, want non-compliant partial smoke subset", r.Contract)
		}
	})
	t.Run("NonCanonicalValueRefused", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := runBenchSubagentFanout(&stdout, &stderr, []string{"-fanout", "2,4", "-arms", arms, "-trials", "1"}); code != 1 {
			t.Fatalf("non-canonical sweep must be refused, exit %d", code)
		}
	})
}

func TestParseFanoutFlagsRejectsUnknownModes(t *testing.T) {
	cases := []struct {
		args []string
		want error
	}{
		{[]string{"-request-shape", "rpc"}, ErrFanoutRequestShape},
		{[]string{"-schedule", "random"}, ErrFanoutSchedule},
		{[]string{"-declared-per-slot-ctx", "-1"}, ErrFanoutDeclaredCtx},
	}
	for _, c := range cases {
		var stderr bytes.Buffer
		if _, err := parseFanoutFlags(&stderr, c.args); !errors.Is(err, c.want) {
			t.Errorf("%v: err = %v, want %v", c.args, err, c.want)
		}
	}
}
