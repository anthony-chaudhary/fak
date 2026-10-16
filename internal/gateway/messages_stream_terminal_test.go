package gateway

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=1s
func TestMessagesStreamMissingTerminalIsNotServed(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	t.Setenv("FAK_WARM_CONTINUE", "0")
	t.Setenv("FAK_STREAM_CHECKPOINT_DIR", "")
	t.Setenv("FAK_STREAM_INCIDENT_DIR", "")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	for _, completed := range []bool{false, true} {
		name := "missing_stop"
		if completed {
			name = "completed"
		}
		t.Run(name, func(t *testing.T) {
			var requests atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				w.Header().Set("Content-Type", "text/event-stream")
				// A clean HTTP EOF after partial output is distinct from the
				// existing broken-chunk warm-continue transport fixture.
				frames := warmContinueDeadFrames
				if completed {
					frames += "event: content_block_stop\n" +
						`data: {"type":"content_block_stop","index":0}` + "\n\n" +
						"event: message_delta\n" +
						`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}` + "\n\n" +
						"event: message_stop\n" + `data: {"type":"message_stop"}` + "\n\n"
				}
				_, _ = io.WriteString(w, frames)
			}))
			defer upstream.Close()
			srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "configured-key"})
			if err != nil {
				t.Fatal(err)
			}
			defer srv.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
				`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hello"}]}`)).WithContext(ctx)
			req.RemoteAddr = "127.0.0.1:54321"
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("x-api-key", "caller-key")
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, req)
			if response.Code != http.StatusOK || !strings.Contains(response.Header().Get("Content-Type"), "event-stream") {
				t.Fatalf("stream did not open: status=%d body=%s", response.Code, response.Body)
			}
			var errors, stops int
			for _, frame := range readAnthropicSSE(t, strings.NewReader(response.Body.String())) {
				if frame.event == "error" {
					errors++
				}
				if frame.event == "message_stop" {
					stops++
				}
			}
			if got := clientTextDeltas(response.Body.String()); got != "Hel" {
				t.Fatalf("partial output lost or replayed: %q", got)
			}
			srv.metrics.inferenceMu.Lock()
			var served uint64
			for _, count := range srv.metrics.inferReqs {
				served += count
			}
			srv.metrics.inferenceMu.Unlock()
			srv.metrics.upstreamErrMu.Lock()
			var failures uint64
			for _, count := range srv.metrics.upstreamErrors {
				failures += count
			}
			srv.metrics.upstreamErrMu.Unlock()
			if completed {
				if errors != 0 || stops != 1 || served != 1 || failures != 0 {
					t.Fatalf("completed stream errors=%d stops=%d served=%d failures=%d", errors, stops, served, failures)
				}
			} else if errors != 1 || stops != 0 || served != 0 || failures != 1 {
				t.Fatalf("unterminated stream errors=%d stops=%d served=%d failures=%d", errors, stops, served, failures)
			}
			if requests.Load() != 1 {
				t.Fatalf("upstream requests=%d, want one with warm-continue disabled", requests.Load())
			}
		})
	}
}
