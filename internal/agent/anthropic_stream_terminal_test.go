package agent

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime fast est=1s
func TestStreamAnthropicRawRequiresTerminal(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	const start = "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_terminal","role":"assistant","content":[]}}` + "\n\n"
	const text = "event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}` + "\n\n"
	const finish = "event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}` + "\n\n"
	const stop = "event: message_stop\n" + `data: {"type":"message_stop"}`
	const failure = "event: error\n" +
		`data: {"type":"error","error":{"type":"overloaded_error","message":"mid-stream failure"}}` + "\n\n"
	callbackError := errors.New("callback stopped")
	for _, tc := range []struct {
		name, body                       string
		missing, transport, stopCallback bool
		wantEvents                       string
	}{
		{"start_only", start, true, false, false, "message_start"},
		{"partial_text", start + text, true, false, false, "message_start,content_block_delta"},
		{"finish_without_stop", start + text + finish, true, false, false, "message_start,content_block_delta,message_delta"},
		{"completed_final_frame", start + text + finish + stop, false, false, false, "message_start,content_block_delta,message_delta,message_stop"},
		{"relayed_error", start + failure, false, false, false, "message_start,error"},
		{"prestart_empty", ": keepalive\n\n", false, false, false, ""},
		{"callback_error", start + text + finish + stop, false, false, true, "message_start"},
		{"transport_error", start + text + finish + stop + "\n\n", false, true, false, "message_start,content_block_delta,message_delta,message_stop"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var requests atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				w.Header().Set("Content-Type", "text/event-stream")
				if tc.transport {
					// Finish the HTTP response short of its declared body length:
					// the read error must still win after a terminal SSE frame.
					w.Header().Set("Content-Length", strconv.Itoa(len(tc.body)+1))
				}
				_, _ = io.WriteString(w, tc.body)
			}))
			defer upstream.Close()
			planner := inBandTestPlanner(t, upstream.URL)
			planner.Client = upstream.Client()
			var events []string
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			err := planner.StreamAnthropicRaw(ctx, inBandRawBody, "k", "", func(ev AnthropicSSEEvent) error {
				events = append(events, ev.Event)
				if tc.stopCallback {
					return callbackError
				}
				return nil
			})
			switch {
			case tc.stopCallback:
				if err != callbackError {
					t.Fatalf("callback error identity changed: %v", err)
				}
			case tc.transport:
				if err != io.ErrUnexpectedEOF {
					t.Fatalf("transport error replaced after terminal: %v", err)
				}
			case tc.missing:
				if !errors.Is(err, io.ErrUnexpectedEOF) || !strings.Contains(err.Error(), "before message_stop") {
					t.Fatalf("unterminated started stream returned %v", err)
				}
			default:
				if err != nil {
					t.Fatalf("existing terminal/prestart contract changed: %v", err)
				}
			}
			if got := strings.Join(events, ","); got != tc.wantEvents {
				t.Fatalf("relayed events=%q, want %q", got, tc.wantEvents)
			}
			if requests.Load() != 1 {
				t.Fatalf("upstream requests=%d, want one", requests.Load())
			}
		})
	}
}
