package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// prefillProgressServer is a fake llama-server whose prefill outlasts the content-progress
// window. While "prefilling" it emits one prompt_progress chunk every `every` for `prefill`,
// but ONLY when the request asked for return_progress (real llama-server is silent otherwise),
// then streams a short completion. repeat makes every report carry the same processed count,
// the shape of an upstream re-sending one report while doing no work. sawProgressFlag records
// whether the request body carried return_progress:true.
func prefillProgressServer(t *testing.T, prefill, every time.Duration, repeat bool, sawProgressFlag *atomic.Bool) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/props" {
			_, _ = io.WriteString(w, `{"total_slots":4}`)
			return
		}
		var body map[string]json.RawMessage
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		progress := string(body["return_progress"]) == "true"
		sawProgressFlag.Store(progress)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		flush := func() {
			if f != nil {
				f.Flush()
			}
		}
		flush()
		deadline := time.After(prefill)
		tick := time.NewTicker(every)
		defer tick.Stop()
		processed := 0
	prefilling:
		for {
			select {
			case <-r.Context().Done():
				return
			case <-deadline:
				break prefilling
			case <-tick.C:
				if !progress {
					continue
				}
				if !repeat {
					processed += 2048
				}
				_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null}}],\"prompt_progress\":{\"total\":102216,\"cache\":0,\"processed\":%d,\"time_ms\":1}}\n\n", processed)
				flush()
			}
		}
		for _, fr := range []string{
			"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
			"data: [DONE]\n\n",
		} {
			_, _ = io.WriteString(w, fr)
			flush()
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func prefillProgressPlanner(base string) *HTTPPlanner {
	p := NewHTTPPlanner(base, "m", "")
	p.LlamaSlotAffinity = true
	p.StreamProgressTimeout = 5 * time.Second
	return p
}

// TestStreamPrefillProgressKeepsLongColdPrefillAlive (#13781): the upstream produces no content
// for 9s against a 5s content-progress window, but reports rising prefill progress every 1s.
// That is a working upstream, so the turn completes on its first attempt. The planner learned
// the upstream is llama-server from RecordLlamaServerSlots (the gateway's startup /props probe),
// so even this first request carries return_progress.
//
// fak-test:runtime slow est=10s lane=default
func TestStreamPrefillProgressKeepsLongColdPrefillAlive(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	var flagged atomic.Bool
	srv := prefillProgressServer(t, 9*time.Second, time.Second, false, &flagged)
	p := prefillProgressPlanner(srv.URL)
	RecordLlamaServerSlots(p.BaseURL, 4)

	comp, err := p.CompleteStream(context.Background(), func(string) error { return nil },
		[]Message{{Role: RoleUser, Content: "long prompt"}}, nil)
	if !flagged.Load() {
		t.Fatal("streamed request to a discovered llama-server did not carry return_progress:true")
	}
	var stalled *UpstreamStalledError
	if errors.As(err, &stalled) {
		t.Fatalf("prefill reporting progress was cut as a stall: kind=%q idle=%s", stalled.Kind, stalled.Idle)
	}
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if comp.Message.Content != "ok" || comp.FinishReason != "stop" {
		t.Fatalf("content=%q finish=%q, want ok/stop", comp.Message.Content, comp.FinishReason)
	}
}

// TestStreamPrefillRepeatedProgressStillTrips: a report whose processed count never rises is not
// progress, so an upstream re-sending one report is still cut at the content-progress window
// (the #5486 property the deadline exists for).
//
// fak-test:runtime slow est=7s lane=default
func TestStreamPrefillRepeatedProgressStillTrips(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	var flagged atomic.Bool
	srv := prefillProgressServer(t, 30*time.Second, time.Second, true, &flagged)
	p := prefillProgressPlanner(srv.URL)
	RecordLlamaServerSlots(p.BaseURL, 4)

	began := time.Now()
	_, err := p.CompleteStream(context.Background(), func(string) error { return nil },
		[]Message{{Role: RoleUser, Content: "long prompt"}}, nil)
	var stalled *UpstreamStalledError
	if !errors.As(err, &stalled) || stalled.Kind != stallKindNoProgress {
		t.Fatalf("err = %v, want a no-progress stall", err)
	}
	if elapsed := time.Since(began); elapsed >= 20*time.Second {
		t.Fatalf("repeated progress report kept the stream alive for %s, want the 5s window", elapsed)
	}
}

// TestLlamaPromptProgressFlagGate: return_progress is llama-server-only and stream-only. It is
// absent before discovery, on a buffered request, and for a planner that did not opt in, and an
// operator-set value wins.
//
// fak-test:runtime fast est=50ms
func TestLlamaPromptProgressFlagGate(t *testing.T) {
	base := "http://llama-progress-gate.invalid/v1"
	p := &HTTPPlanner{Provider: ProviderOpenAI, BaseURL: base, ModelID: "m", LlamaSlotAffinity: true}
	has := func(extra json.RawMessage) bool { return strings.Contains(string(extra), `"return_progress"`) }

	if has(p.withLlamaPromptProgress(nil, true)) {
		t.Fatal("return_progress sent before the upstream was discovered as llama-server")
	}
	RecordLlamaServerSlots(base, 4)
	if got := p.withLlamaPromptProgress(nil, true); string(got) != `{"return_progress":true}` {
		t.Fatalf("discovered streamed request extra = %s, want return_progress:true", got)
	}
	if has(p.withLlamaPromptProgress(nil, false)) {
		t.Fatal("return_progress sent on a buffered request")
	}
	off := &HTTPPlanner{Provider: ProviderOpenAI, BaseURL: base, ModelID: "m"}
	if has(off.withLlamaPromptProgress(nil, true)) {
		t.Fatal("return_progress sent for a planner that did not opt in")
	}
	if got := p.withLlamaPromptProgress(json.RawMessage(`{"return_progress":false}`), true); string(got) != `{"return_progress":false}` {
		t.Fatalf("operator return_progress overridden: %s", got)
	}
}
