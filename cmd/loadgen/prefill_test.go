package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// casPeak raises *peak to v when v exceeds the current value.
func casPeak(peak *atomic.Int64, v int64) {
	for {
		old := peak.Load()
		if v <= old || peak.CompareAndSwap(old, v) {
			return
		}
	}
}

// fak-test:runtime fast est=1s
// TestPrefillSlotRelease pins the two-tier concurrency contract adapted from
// ai-dynamo/aiperf (src/aiperf/credit/issuer.py:311 @ ab610add4e8876fca223197596c728e17365f381):
// a session slot per in-flight request plus a prefill slot released at TTFT,
// released symmetrically on failure, and tied to the concurrency by default.
func TestPrefillSlotRelease(t *testing.T) {
	t.Run("validateSentinels", func(t *testing.T) {
		tests := []struct {
			name string
			cfg  Config
			want error
		}{
			{name: "prefill above maximum concurrency", cfg: Config{Concurrencies: []int{4}, RequestsPer: 4, PrefillConcurrency: 5}, want: errPrefillAboveConcurrency},
			{name: "prefill tied to maximum concurrency", cfg: Config{Concurrencies: []int{4}, RequestsPer: 4, PrefillConcurrency: 4}, want: nil},
			{name: "prefill default zero", cfg: Config{Concurrencies: []int{4}, RequestsPer: 4}, want: nil},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var networkCalls atomic.Int64
				tt.cfg.HTTPClient = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
					networkCalls.Add(1)
					return nil, errors.New("unexpected network call")
				})}
				if err := validateConfig(&tt.cfg); !errors.Is(err, tt.want) {
					t.Fatalf("validateConfig error=%v want category %v", err, tt.want)
				}
				if calls := networkCalls.Load(); calls != 0 {
					t.Fatalf("network calls=%d want 0", calls)
				}
			})
		}
	})

	t.Run("releasedAtTTFT", func(t *testing.T) {
		var srvSession, srvSessionPeak, srvPrefill, srvPrefillPeak atomic.Int64
		entered := make(chan struct{}, 2)
		firstChunkGate := make(chan struct{})
		decodeHold := make(chan struct{})

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			casPeak(&srvSessionPeak, srvSession.Add(1))
			casPeak(&srvPrefillPeak, srvPrefill.Add(1))
			entered <- struct{}{}
			w.Header().Set("Content-Type", "text/event-stream")
			fl := w.(http.Flusher)
			<-firstChunkGate
			srvPrefill.Add(-1) // prefill phase ends before the first content byte leaves
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n")
			fl.Flush()
			<-decodeHold
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}],\"usage\":{\"completion_tokens\":2}}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			fl.Flush()
			srvSession.Add(-1)
		}))
		defer srv.Close()

		cfg := &Config{
			URL: srv.URL, Model: "stub", Prompt: "hi", MaxTokens: 8,
			Concurrencies: []int{2}, RequestsPer: 2, PrefillConcurrency: 1,
		}
		done := make(chan MatrixResult, 1)
		go func() { done <- Run(context.Background(), cfg) }()

		testCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		// Request A reaches the server; request B holds its prefill slot behind A.
		select {
		case <-entered:
		case <-testCtx.Done():
			t.Fatalf("waiting for first request to enter server: %v", testCtx.Err())
		}
		close(firstChunkGate)
		// B may enter only after A's first content chunk freed the prefill slot.
		select {
		case <-entered:
		case <-testCtx.Done():
			t.Fatalf("waiting for second request to enter server: %v", testCtx.Err())
		}
		close(decodeHold)

		var got MatrixResult
		select {
		case got = <-done:
		case <-testCtx.Done():
			t.Fatalf("waiting for loadgen: %v", testCtx.Err())
		}
		if len(got.Points) != 1 {
			t.Fatalf("points=%d want 1", len(got.Points))
		}
		point := got.Points[0]
		if point.Errors != 0 {
			t.Fatalf("errors=%d want 0", point.Errors)
		}
		if point.CompletionTokens != 4 {
			t.Fatalf("tokens=%d want 4", point.CompletionTokens)
		}
		if point.PeakInFlight != 2 {
			t.Fatalf("session peak=%d want 2", point.PeakInFlight)
		}
		if point.PrefillConcurrency != 1 {
			t.Fatalf("reported prefill cap=%d want 1", point.PrefillConcurrency)
		}
		if point.PeakPrefillInFlight != 1 {
			t.Fatalf("client prefill peak=%d want 1", point.PeakPrefillInFlight)
		}
		if got := srvSessionPeak.Load(); got != 2 {
			t.Fatalf("server session peak=%d want 2", got)
		}
		if got := srvPrefillPeak.Load(); got != 1 {
			t.Fatalf("server prefill peak=%d want 1", got)
		}
	})

	t.Run("releasedOnFailure", func(t *testing.T) {
		var hits atomic.Int64
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			if hits.Add(1) == 1 {
				http.Error(w, "upstream boom", http.StatusBadGateway)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}],\"usage\":{\"completion_tokens\":3}}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}))
		defer srv.Close()

		cfg := &Config{
			URL: srv.URL, Model: "stub", Prompt: "hi", MaxTokens: 8,
			Concurrencies: []int{2}, RequestsPer: 2, PrefillConcurrency: 1,
		}
		done := make(chan MatrixResult, 1)
		go func() { done <- Run(context.Background(), cfg) }()
		testCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		var got MatrixResult
		select {
		case got = <-done:
		case <-testCtx.Done():
			t.Fatalf("run did not finish: prefill slot leaked after failed request? %v", testCtx.Err())
		}
		if len(got.Points) != 1 {
			t.Fatalf("points=%d want 1", len(got.Points))
		}
		point := got.Points[0]
		if point.Errors != 1 || point.CompletionTokens != 3 {
			t.Fatalf("errors/tokens=%d/%d want 1/3", point.Errors, point.CompletionTokens)
		}
		if got := hits.Load(); got != 2 {
			t.Fatalf("server hits=%d want 2", got)
		}
	})

	t.Run("defaultTierMatchesConcurrency", func(t *testing.T) {
		entered := make(chan struct{}, 2)
		firstChunkGate := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			entered <- struct{}{}
			w.Header().Set("Content-Type", "text/event-stream")
			fl := w.(http.Flusher)
			<-firstChunkGate
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}],\"usage\":{\"completion_tokens\":1}}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			fl.Flush()
		}))
		defer srv.Close()

		cfg := &Config{
			URL: srv.URL, Model: "stub", Prompt: "hi", MaxTokens: 8,
			Concurrencies: []int{2}, RequestsPer: 2,
		}
		done := make(chan MatrixResult, 1)
		go func() { done <- Run(context.Background(), cfg) }()
		testCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		for i := 0; i < 2; i++ {
			select {
			case <-entered:
			case <-testCtx.Done():
				t.Fatalf("waiting for request %d to enter server: %v", i+1, testCtx.Err())
			}
		}
		close(firstChunkGate)

		var got MatrixResult
		select {
		case got = <-done:
		case <-testCtx.Done():
			t.Fatalf("waiting for loadgen: %v", testCtx.Err())
		}
		if len(got.Points) != 1 {
			t.Fatalf("points=%d want 1", len(got.Points))
		}
		point := got.Points[0]
		if point.Errors != 0 {
			t.Fatalf("errors=%d want 0", point.Errors)
		}
		if point.PrefillConcurrency != 2 {
			t.Fatalf("default prefill cap=%d want 2 (tied to concurrency)", point.PrefillConcurrency)
		}
		if point.PeakPrefillInFlight != 2 {
			t.Fatalf("default prefill peak=%d want 2 (both requests prefill concurrently)", point.PeakPrefillInFlight)
		}
	})
}
