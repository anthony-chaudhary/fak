package gateway

import (
	"context"
	"encoding/json"
	"fmt"
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
func TestMessagesStreamProviderErrorAccounting(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	t.Setenv("FAK_STREAM_CHECKPOINT_DIR", "")
	t.Setenv("FAK_STREAM_INCIDENT_DIR", "")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	const providerData = `{"type":"error","error":{"type":"overloaded_error","message":"provider is busy","request_id":"fixture-request"}}`
	providerFrame := "event: error\ndata: " + providerData + "\n\n"
	tailAt := strings.Index(warmContinueTailFrames, "event: message_delta\n")
	stopAt := strings.Index(warmContinueTailFrames, "event: content_block_stop\n")
	if tailAt < 0 || stopAt < 0 {
		t.Fatal("existing continuation fixture lost its terminal frames")
	}
	completed := warmContinueDeadFrames + warmContinueTailFrames[stopAt:]
	for _, tc := range []struct {
		name      string
		warm      string
		wantCalls int32
		wantText  string
		provider  bool
		completed bool
		prestart  bool
	}{
		{name: "error_off", warm: "0", wantCalls: 1, wantText: "Hel", provider: true},
		{name: "error_on", warm: "1", wantCalls: 1, wantText: "Hel", provider: true},
		{name: "error_then_frames", warm: "1", wantCalls: 1, wantText: "Hel", provider: true},
		{name: "continuation_error", warm: "1", wantCalls: 2, wantText: "Hello world", provider: true},
		{name: "completed", warm: "0", wantCalls: 1, wantText: "Hel", completed: true},
		{name: "transport_error", warm: "0", wantCalls: 1, wantText: "Hel"},
		{name: "prestart_refusal", warm: "0", wantCalls: 1, prestart: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_WARM_CONTINUE", tc.warm)
			var calls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				n := calls.Add(1)
				if tc.name == "transport_error" || (tc.name == "continuation_error" && n == 1) {
					writeDeadPartial(w, warmContinueDeadFrames)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				frames := warmContinueDeadFrames + providerFrame
				switch tc.name {
				case "prestart_refusal":
					frames = "event: error\ndata: " + `{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}` + "\n\n"
				case "completed":
					frames = completed
				case "error_then_frames":
					// A terminal provider error must stop consumption even if a peer
					// appends more text, a second error, or a success terminator.
					frames += warmContinueTailFrames + providerFrame
				case "continuation_error":
					frames = warmContinueTailFrames[:tailAt] + providerFrame
				}
				_, _ = io.WriteString(w, frames)
			}))
			defer upstream.Close()
			srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "configured-key"})
			if err != nil {
				t.Fatal(err)
			}
			defer srv.Close()
			var debits int
			var debited SessionUsage
			srv.debitSession = func(_ context.Context, trace string, usage SessionUsage) SessionState {
				if trace == "" {
					t.Error("served usage lost its session identity")
				}
				debits++
				debited = usage
				return SessionState{}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(warmContinueInbound()))).WithContext(ctx)
			req.RemoteAddr = "127.0.0.1:54321"
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("x-api-key", "caller-key")
			req.Header.Set("X-Trace-Id", "provider-accounting-"+tc.name)
			response := httptest.NewRecorder()
			srv.Handler().ServeHTTP(response, req)
			wantStatus := http.StatusOK
			if tc.prestart {
				wantStatus = http.StatusBadRequest
			}
			if response.Code != wantStatus {
				t.Fatalf("status=%d want=%d body=%s", response.Code, wantStatus, response.Body)
			}
			var errorFrames []string
			var stops int
			for _, frame := range readAnthropicSSE(t, strings.NewReader(response.Body.String())) {
				if frame.event == "error" {
					errorFrames = append(errorFrames, frame.data)
				}
				if frame.event == "message_stop" {
					stops++
				}
			}
			wantErrors, wantStops := 1, 0
			if tc.completed || tc.prestart {
				wantErrors = 0
			}
			if tc.completed {
				wantStops = 1
			}
			if len(errorFrames) != wantErrors || stops != wantStops {
				t.Fatalf("error frames=%v stops=%d, want errors=%d stops=%d", errorFrames, stops, wantErrors, wantStops)
			}
			if tc.provider {
				var got, want any
				if err := json.Unmarshal([]byte(errorFrames[0]), &got); err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal([]byte(providerData), &want); err != nil {
					t.Fatal(err)
				}
				gotJSON, _ := json.Marshal(got)
				wantJSON, _ := json.Marshal(want)
				if string(gotJSON) != string(wantJSON) {
					t.Fatalf("provider error changed: got=%s want=%s", gotJSON, wantJSON)
				}
			}
			if text := clientTextDeltas(response.Body.String()); text != tc.wantText {
				t.Fatalf("relayed text=%q want=%q", text, tc.wantText)
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
			if tc.completed {
				if served != 1 || failures != 0 || debits != 1 || debited.CompletionTokens != 3 || debited.ContextTokens != 10 {
					t.Fatalf("completed served=%d failures=%d debits=%d usage=%+v", served, failures, debits, debited)
				}
			} else if served != 0 || failures != 1 || debits != 0 {
				t.Fatalf("failed stream served=%d failures=%d debits=%d", served, failures, debits)
			}
			if got := calls.Load(); got != tc.wantCalls {
				t.Fatalf("upstream calls=%d want=%d", got, tc.wantCalls)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestWarmContinueRejectsRelayedProviderError(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{name: "provider", err: errUpstreamStreamRefusal},
		{name: "wrapped_provider", err: fmt.Errorf("relay: %w", errUpstreamStreamRefusal)},
		{name: "transport", err: io.ErrUnexpectedEOF, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := warmContinuableErr(tc.err); got != tc.want {
				t.Fatalf("warmContinuableErr(%v)=%v want=%v", tc.err, got, tc.want)
			}
		})
	}
}
