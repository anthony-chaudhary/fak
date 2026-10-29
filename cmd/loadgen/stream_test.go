package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestAggregate_PhaseSeparatedDecodeMetrics pins the closed contract on synthetic
// per-request results: TTFT percentiles, mean ITL (TPOT), decode tok/s = 1000/TPOT,
// and the unchanged end-to-end completion/wall rate.
func TestAggregate_PhaseSeparatedDecodeMetrics(t *testing.T) {
	results := []reqResult{
		{latency: 300 * time.Millisecond, completionTokens: 10, streamed: true,
			ttft: 20 * time.Millisecond, itls: []time.Duration{10 * time.Millisecond, 10 * time.Millisecond}},
		{latency: 500 * time.Millisecond, completionTokens: 10, streamed: true,
			ttft: 40 * time.Millisecond, itls: []time.Duration{20 * time.Millisecond}},
	}
	pr := aggregate(1, 2, 2*time.Second, results)
	if !pr.Streamed {
		t.Fatal("streamed=false want true")
	}
	if pr.TTFTP50MS != 20 || pr.TTFTP95MS != 40 {
		t.Fatalf("ttft p50/p95=%v/%v want 20/40", pr.TTFTP50MS, pr.TTFTP95MS)
	}
	wantTPOT := 40.0 / 3.0 // mean of 10, 10, 20 ms
	if math.Abs(pr.TPOTMS-wantTPOT) > 1e-9 {
		t.Fatalf("tpot=%v want %v", pr.TPOTMS, wantTPOT)
	}
	if pr.DecodeTokensPerSecond != 1000/pr.TPOTMS {
		t.Fatalf("decode_tps=%v want 1000/tpot=%v", pr.DecodeTokensPerSecond, 1000/pr.TPOTMS)
	}
	if math.Abs(pr.DecodeTokensPerSecond-75) > 1e-9 {
		t.Fatalf("decode_tps=%v want ~75", pr.DecodeTokensPerSecond)
	}
	if pr.TokensPerSecond != 10 { // 20 completion tokens / 2s sweep wall, formula unchanged
		t.Fatalf("tok/s=%v want 10 (completion/wall)", pr.TokensPerSecond)
	}
}

// TestAggregate_MixedStreamedAndFallback keeps phase fields sourced from streamed
// successes only while fallback tokens still count toward the end-to-end rate.
func TestAggregate_MixedStreamedAndFallback(t *testing.T) {
	results := []reqResult{
		{latency: 300 * time.Millisecond, completionTokens: 10, streamed: true,
			ttft: 25 * time.Millisecond, itls: []time.Duration{5 * time.Millisecond}},
		{latency: 400 * time.Millisecond, completionTokens: 10}, // plain-JSON fallback
	}
	pr := aggregate(1, 2, 2*time.Second, results)
	if !pr.Streamed {
		t.Fatal("streamed=false want true (one request streamed)")
	}
	if pr.TTFTP50MS != 25 || pr.TTFTP95MS != 25 {
		t.Fatalf("ttft p50/p95=%v/%v want 25/25 (fallback contributes none)", pr.TTFTP50MS, pr.TTFTP95MS)
	}
	if pr.TPOTMS != 5 || pr.DecodeTokensPerSecond != 200 {
		t.Fatalf("tpot/decode=%v/%v want 5/200", pr.TPOTMS, pr.DecodeTokensPerSecond)
	}
	if pr.CompletionTokens != 20 || pr.TokensPerSecond != 10 {
		t.Fatalf("tokens/tok-s=%d/%v want 20/10 (fallback tokens still end-to-end)", pr.CompletionTokens, pr.TokensPerSecond)
	}
}

// TestAggregate_AllErrorsLeavesPhaseFieldsZero guards the omitempty shape: nothing
// succeeded, so every additive field stays at its zero value.
func TestAggregate_AllErrorsLeavesPhaseFieldsZero(t *testing.T) {
	results := []reqResult{{err: fmt.Errorf("boom")}, {err: fmt.Errorf("boom")}}
	pr := aggregate(1, 2, time.Second, results)
	if pr.Streamed || pr.TTFTP50MS != 0 || pr.TTFTP95MS != 0 || pr.TPOTMS != 0 || pr.DecodeTokensPerSecond != 0 {
		t.Fatalf("phase fields nonzero on all-error point: %+v", pr)
	}
}

// fak-test:runtime fast est=2s
func TestRun_StreamingSSEMeasuresTTFTAndTPOT(t *testing.T) {
	const (
		ttftDelay = 60 * time.Millisecond
		itlDelay  = 40 * time.Millisecond
		chunks    = 5 // content chunks per request
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if !req.Stream {
			t.Error("request stream=false want true")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		time.Sleep(ttftDelay)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n")
		flusher.Flush()
		for i := 1; i < chunks; i++ {
			time.Sleep(itlDelay)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n")
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":%d}}\n\n", chunks)
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	cfg := &Config{
		URL: srv.URL + "/v1/chat/completions", Model: "stub", Stack: "test",
		Prompt: "hi", MaxTokens: 8, Concurrencies: []int{1}, RequestsPer: 2,
	}
	mr := Run(context.Background(), cfg)
	if len(mr.Points) != 1 {
		t.Fatalf("points=%d want 1", len(mr.Points))
	}
	p := mr.Points[0]
	if p.Errors != 0 {
		t.Fatalf("errors=%d want 0", p.Errors)
	}
	if !p.Streamed {
		t.Fatal("streamed=false want true")
	}
	if p.CompletionTokens != 2*chunks {
		t.Fatalf("tokens=%d want %d (usage.completion_tokens per request)", p.CompletionTokens, 2*chunks)
	}
	// TTFT: the first content chunk arrives ~ttftDelay after request start.
	if p.TTFTP50MS < float64(ttftDelay.Milliseconds())/2 {
		t.Fatalf("ttft p50=%vms below half the injected %vms", p.TTFTP50MS, ttftDelay)
	}
	if p.TTFTP95MS > float64(ttftDelay.Milliseconds())*4+250 {
		t.Fatalf("ttft p95=%vms far above the injected %vms", p.TTFTP95MS, ttftDelay)
	}
	// TPOT: mean inter-content-chunk arrival ~itlDelay.
	if p.TPOTMS < float64(itlDelay.Milliseconds())/2 {
		t.Fatalf("tpot=%vms below half the injected %vms", p.TPOTMS, itlDelay)
	}
	if p.TPOTMS > float64(itlDelay.Milliseconds())*4+250 {
		t.Fatalf("tpot=%vms far above the injected %vms", p.TPOTMS, itlDelay)
	}
	if p.DecodeTokensPerSecond != 1000/p.TPOTMS {
		t.Fatalf("decode_tps=%v want 1000/tpot=%v", p.DecodeTokensPerSecond, 1000/p.TPOTMS)
	}
	// End-to-end rate keeps its MATRIX.json contract: completion tokens over sweep wall.
	if p.TokensPerSecond != float64(p.CompletionTokens)/p.WallSeconds {
		t.Fatalf("tok/s=%v want completion/wall=%v", p.TokensPerSecond, float64(p.CompletionTokens)/p.WallSeconds)
	}
}

// TestRun_NonStreamingEndpointFallsBackWithoutPhaseFields covers the robustness
// contract: an endpoint that answers plain JSON (ignoring stream:true) yields today's
// non-streaming behavior with streamed=false and every new field left zero.
func TestRun_NonStreamingEndpointFallsBackWithoutPhaseFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"completion_tokens":7}}`)
	}))
	defer srv.Close()

	cfg := &Config{
		URL: srv.URL + "/v1/chat/completions", Model: "stub", Prompt: "hi",
		MaxTokens: 8, Concurrencies: []int{1}, RequestsPer: 2,
	}
	mr := Run(context.Background(), cfg)
	p := mr.Points[0]
	if p.Errors != 0 {
		t.Fatalf("errors=%d want 0", p.Errors)
	}
	if p.Streamed {
		t.Fatal("streamed=true want false (endpoint answered plain JSON)")
	}
	if p.TTFTP50MS != 0 || p.TTFTP95MS != 0 || p.TPOTMS != 0 || p.DecodeTokensPerSecond != 0 {
		t.Fatalf("phase fields nonzero on fallback: %+v", p)
	}
	if p.CompletionTokens != 14 {
		t.Fatalf("tokens=%d want 14", p.CompletionTokens)
	}
	if p.TokensPerSecond <= 0 {
		t.Fatalf("tok/s=%v want >0", p.TokensPerSecond)
	}
}

// TestRun_MalformedSSENeverPanicsAndPrefersUsage feeds garbage lines between valid
// chunks: malformed payloads are skipped, the request still succeeds, and
// usage.completion_tokens wins over the observed content-chunk count.
func TestRun_MalformedSSENeverPanicsAndPrefersUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "event: ping\n\n")
		fmt.Fprint(w, "data: {not-json\n\n")
		fmt.Fprint(w, "\n\n")
		// Deterministic TTFT: hold the first content chunk back past timer granularity so
		// p50 is far above sub-microsecond noise instead of racing the loopback round-trip.
		time.Sleep(5 * time.Millisecond)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"y\"}}]}\n\n")
		fmt.Fprint(w, "garbage line without prefix\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"z\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"completion_tokens\":5}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	cfg := &Config{
		URL: srv.URL + "/v1/chat/completions", Model: "stub", Prompt: "hi",
		MaxTokens: 8, Concurrencies: []int{1}, RequestsPer: 1,
	}
	mr := Run(context.Background(), cfg)
	p := mr.Points[0]
	if p.Errors != 0 {
		t.Fatalf("errors=%d want 0 (malformed lines must be skipped, not fatal)", p.Errors)
	}
	if !p.Streamed {
		t.Fatal("streamed=false want true")
	}
	if p.CompletionTokens != 5 {
		t.Fatalf("tokens=%d want 5 (usage.completion_tokens preferred over 3 content chunks)", p.CompletionTokens)
	}
	if p.TTFTP50MS <= 0 {
		t.Fatalf("ttft p50=%v want >0 (content chunks did arrive)", p.TTFTP50MS)
	}
	// Burst delivery can yield a zero mean ITL; pin the contract either way:
	// decode = 1000/tpot when tpot > 0, else 0.
	if p.TPOTMS > 0 && p.DecodeTokensPerSecond != 1000/p.TPOTMS {
		t.Fatalf("decode_tps=%v want 1000/tpot=%v", p.DecodeTokensPerSecond, 1000/p.TPOTMS)
	}
	if p.TPOTMS == 0 && p.DecodeTokensPerSecond != 0 {
		t.Fatalf("decode_tps=%v want 0 when tpot=0", p.DecodeTokensPerSecond)
	}
}

// TestRun_StreamErrorEventCountsAsError pins that an in-band OpenAI error object ends
// the request as an error, leaving the point's phase fields unset.
func TestRun_StreamErrorEventCountsAsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"error\":{\"message\":\"overloaded\"}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	cfg := &Config{
		URL: srv.URL + "/v1/chat/completions", Model: "stub", Prompt: "hi",
		MaxTokens: 8, Concurrencies: []int{1}, RequestsPer: 1,
	}
	mr := Run(context.Background(), cfg)
	p := mr.Points[0]
	if p.Errors != 1 || p.ErrorRate != 1.0 {
		t.Fatalf("errors/rate=%d/%v want 1/1.0", p.Errors, p.ErrorRate)
	}
	if p.Streamed || p.TTFTP50MS != 0 || p.TPOTMS != 0 || p.DecodeTokensPerSecond != 0 {
		t.Fatalf("phase fields set on errored point: %+v", p)
	}
}
