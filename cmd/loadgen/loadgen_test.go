package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestPercentile(t *testing.T) {
	s := []float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	cases := map[float64]float64{50: 50, 95: 100, 100: 100}
	for p, want := range cases {
		if got := percentile(s, p); got != want {
			t.Errorf("percentile(%v)=%v want %v", p, got, want)
		}
	}
	if got := percentile(nil, 50); got != 0 {
		t.Errorf("empty percentile=%v want 0", got)
	}
	if got := percentile([]float64{42}, 95); got != 42 {
		t.Errorf("single percentile=%v want 42", got)
	}
}

func TestAggregate_TokensPerSecond(t *testing.T) {
	// 4 requests, each 10 completion tokens, wall = 2s => 40 tok / 2s = 20 tok/s.
	results := []reqResult{
		{latency: 100 * time.Millisecond, completionTokens: 10},
		{latency: 200 * time.Millisecond, completionTokens: 10},
		{latency: 300 * time.Millisecond, completionTokens: 10},
		{latency: 400 * time.Millisecond, completionTokens: 10},
	}
	pr := aggregate(2, 4, 2*time.Second, results)
	if pr.CompletionTokens != 40 {
		t.Fatalf("tokens=%d want 40", pr.CompletionTokens)
	}
	if pr.TokensPerSecond != 20 {
		t.Fatalf("tok/s=%v want 20", pr.TokensPerSecond)
	}
	if pr.ErrorRate != 0 {
		t.Fatalf("errrate=%v want 0", pr.ErrorRate)
	}
	if pr.P50LatencyMS != 200 || pr.P95LatencyMS != 400 {
		t.Fatalf("p50=%v p95=%v want 200/400", pr.P50LatencyMS, pr.P95LatencyMS)
	}
}

func TestAggregate_CountsErrors(t *testing.T) {
	results := []reqResult{
		{latency: 100 * time.Millisecond, completionTokens: 5},
		{err: fmt.Errorf("boom"), latency: 50 * time.Millisecond},
		{err: fmt.Errorf("boom"), latency: 50 * time.Millisecond},
		{latency: 100 * time.Millisecond, completionTokens: 5},
	}
	pr := aggregate(4, 4, time.Second, results)
	if pr.Errors != 2 || pr.ErrorRate != 0.5 {
		t.Fatalf("errors=%d rate=%v want 2/0.5", pr.Errors, pr.ErrorRate)
	}
	if pr.CompletionTokens != 10 {
		t.Fatalf("tokens=%d want 10 (errors excluded)", pr.CompletionTokens)
	}
}

func TestRun_AgainstStubServer(t *testing.T) {
	var hits int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&hits, 1)
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("missing content-type")
		}
		var req chatRequest
		json.NewDecoder(r.Body).Decode(&req)
		resp := chatResponse{}
		resp.Choices = []struct {
			Message chatMessage `json:"message"`
		}{{Message: chatMessage{Role: "assistant", Content: "ok"}}}
		resp.Usage.CompletionTokens = 7
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cfg := &Config{
		URL: srv.URL + "/v1/chat/completions", Model: "stub", Stack: "test",
		Prompt: "hi", MaxTokens: 8, Concurrencies: []int{1, 4}, RequestsPer: 8,
	}
	mr := Run(context.Background(), cfg)
	if len(mr.Points) != 2 {
		t.Fatalf("points=%d want 2", len(mr.Points))
	}
	for _, p := range mr.Points {
		if p.Errors != 0 {
			t.Errorf("c=%d errors=%d", p.Concurrency, p.Errors)
		}
		if p.CompletionTokens != 8*7 {
			t.Errorf("c=%d tokens=%d want 56", p.Concurrency, p.CompletionTokens)
		}
		if p.TokensPerSecond <= 0 {
			t.Errorf("c=%d tok/s=%v want >0", p.Concurrency, p.TokensPerSecond)
		}
	}
	if got := atomic.LoadInt64(&hits); got != 16 {
		t.Fatalf("total hits=%d want 16", got)
	}
}

func TestRun_HandlesServerErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "upstream boom", http.StatusBadGateway)
	}))
	defer srv.Close()
	cfg := &Config{
		URL: srv.URL + "/v1/chat/completions", Model: "x",
		Prompt: "hi", Concurrencies: []int{2}, RequestsPer: 4,
	}
	mr := Run(context.Background(), cfg)
	if mr.Points[0].Errors != 4 || mr.Points[0].ErrorRate != 1.0 {
		t.Fatalf("expected all 4 errors, got %+v", mr.Points[0])
	}
}

func TestParseInts(t *testing.T) {
	got, err := parseInts("1, 4 ,16,")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 4, 16}
	if len(got) != 3 || got[0] != 1 || got[1] != 4 || got[2] != 16 {
		t.Fatalf("got %v want %v", got, want)
	}
	if _, err := parseInts("abc"); err == nil {
		t.Fatal("expected error")
	}
}
