package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// pacedToolCallUpstream streams a tool-call-only turn: the role frame, then each
// argument fragment after gap, then blocks on release before the finish frame.
func pacedToolCallUpstream(t *testing.T, gap time.Duration, fragments []string, release <-chan struct{}) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl := w.(http.Flusher)
		send := func(frame string) {
			_, _ = io.WriteString(w, "data: "+frame+"\n\n")
			fl.Flush()
		}
		send(`{"model":"served-x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`)
		send(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"allow_x","arguments":""}}]}}]}`)
		for _, frag := range fragments {
			time.Sleep(gap)
			raw, _ := json.Marshal(frag)
			send(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":` + string(raw) + `}}]}}]}`)
		}
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		send(`{"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}`)
		send("[DONE]")
	}))
}

func startToolStream(t *testing.T, url string) <-chan *http.Response {
	t.Helper()
	raw, _ := json.Marshal(map[string]any{
		"model":    "x:model",
		"messages": []map[string]string{{"role": "user", "content": "do it"}},
		"tools":    []map[string]any{{"type": "function", "function": map[string]any{"name": "allow_x"}}},
		"stream":   true,
	})
	got := make(chan *http.Response, 1)
	go func() {
		resp, err := http.Post(url+"/v1/chat/completions", "application/json", bytes.NewReader(raw))
		if err != nil {
			t.Errorf("post: %v", err)
			close(got)
			return
		}
		got <- resp
	}()
	return got
}

// TestToolCallOnlyStreamCommitsBeforeTurnEnds: a turn whose only output is a tool call
// (the common agent step) must put the 200 and the opening role chunk on the wire as
// soon as the model starts decoding, not when the whole call has been generated.
func TestToolCallOnlyStreamCommitsBeforeTurnEnds(t *testing.T) {
	release := make(chan struct{})
	up := pacedToolCallUpstream(t, 0, []string{`{"a":1}`}, release)
	defer up.Close()
	ts := liveStreamServer(t, up.URL)
	defer ts.Close()
	defer close(release)

	var resp *http.Response
	select {
	case resp = <-startToolStream(t, ts.URL):
	case <-time.After(10 * time.Second):
		t.Fatal("no response headers while the upstream was still decoding the tool call")
	}
	if resp == nil {
		t.FailNow()
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil {
		t.Fatalf("read first frame: %v", err)
	}
	var first ChatStreamResponse
	if err := json.Unmarshal([]byte(strings.TrimPrefix(strings.TrimSpace(line), "data: ")), &first); err != nil {
		t.Fatalf("first frame %q: %v", line, err)
	}
	if len(first.Choices) != 1 || first.Choices[0].Delta.Role != "assistant" || len(first.Choices[0].Delta.ToolCalls) != 0 {
		t.Fatalf("first frame = %+v, want the bare opening role chunk", first)
	}
}

// TestHeldToolCallStreamEmitsKeepalive: while a long tool call is held for adjudication
// the committed stream must not go silent, or a downstream stall watchdog cuts a healthy
// turn. The held call itself still arrives whole, after the keepalives.
func TestHeldToolCallStreamEmitsKeepalive(t *testing.T) {
	release := make(chan struct{})
	close(release)
	frags := []string{`{"a":`, `1`, `,"b":`, `2`, `}`}
	up := pacedToolCallUpstream(t, 600*time.Millisecond, frags, release)
	defer up.Close()
	ts := liveStreamServer(t, up.URL)
	defer ts.Close()

	resp := <-startToolStream(t, ts.URL)
	if resp == nil {
		t.FailNow()
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	text := string(raw)
	callAt := strings.Index(text, `"tool_calls"`)
	if callAt < 0 {
		t.Fatalf("no tool call delivered: %s", raw)
	}
	if !strings.Contains(text[:callAt], "\n: ") && !strings.HasPrefix(text, ": ") {
		t.Fatalf("no SSE comment keepalive before the held tool call: %s", raw)
	}
	var data []string
	for _, line := range strings.Split(text, "\n") {
		if !strings.HasPrefix(line, ":") {
			data = append(data, line)
		}
	}
	chunks, sawDone := parseSSEChunks(t, []byte(strings.Join(data, "\n")))
	if !sawDone {
		t.Fatalf("missing [DONE]: %s", raw)
	}
	var args string
	for _, c := range chunks {
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			args += tc.Function.Arguments
		}
	}
	if args != `{"a":1,"b":2}` {
		t.Fatalf("tool call arguments = %q, want the whole held call", args)
	}
}

// liveChatFailWriter returns errors without canceling the request itself.
// Counters distinguish suppressed wrapper operations from physical retries.
type liveChatFailWriter struct {
	mu                            sync.Mutex
	header                        http.Header
	body                          strings.Builder
	failPrefix                    string
	cause                         error
	short                         bool
	failed                        bool
	writes, flushes, afterFailure int
	role                          chan struct{}
	roleOnce                      sync.Once
}

func (w *liveChatFailWriter) Header() http.Header { return w.header }
func (w *liveChatFailWriter) WriteHeader(int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.failed {
		w.afterFailure++
	}
}
func (w *liveChatFailWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.writes++
	if w.failed {
		w.afterFailure++
		return 0, w.cause
	}
	if w.failPrefix == "" || strings.HasPrefix(string(p), w.failPrefix) {
		w.failed = true
		if w.short {
			return len(p) - 1, nil
		}
		return 0, w.cause
	}
	w.body.Write(p)
	if strings.Contains(string(p), `"role":"assistant"`) && w.role != nil {
		w.roleOnce.Do(func() { close(w.role) })
	}
	return len(p), nil
}
func (w *liveChatFailWriter) Flush() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.failed {
		w.afterFailure++
	}
	w.flushes++
}

// fak-test:runtime fast est=20ms lane=default
func TestLiveChatWriterLatchesBeforeCancelAndSuppresses(t *testing.T) {
	for _, short := range []bool{false, true} {
		t.Run(fmt.Sprint(short), func(t *testing.T) {
			cause := errors.New("downstream write failed")
			want := error(cause)
			if short {
				want = io.ErrShortWrite
			}
			raw := &liveChatFailWriter{header: make(http.Header), cause: cause, short: short}
			var w *liveChatResponseWriter
			cancelled := make(chan struct{})
			w = &liveChatResponseWriter{syncResponseWriter: newSyncResponseWriter(raw), cancel: func() {
				if !w.mu.TryLock() {
					t.Error("cancel called under writer lock")
				} else {
					if w.firstError != want {
						t.Error("cancel ran before latch")
					}
					w.mu.Unlock()
				}
				close(cancelled)
			}}
			done := make(chan struct{})
			go func() {
				defer close(done)
				_, err := w.Write([]byte("frame"))
				if err != want {
					t.Errorf("Write error=%v", err)
				}
			}()
			select {
			case <-done:
			case <-time.After(3 * time.Second):
				t.Fatal("cancel called under writer lock")
			}
			select {
			case <-cancelled:
			default:
				t.Fatal("writer did not cancel")
			}
			if _, err := w.Write([]byte("later")); err != want {
				t.Fatal("lost first error")
			}
			w.Flush()
			w.WriteHeader(500)
			raw.mu.Lock()
			defer raw.mu.Unlock()
			if raw.writes != 1 || raw.flushes != 0 || raw.afterFailure != 0 {
				t.Fatalf("physical retries: writes=%d flushes=%d after=%d", raw.writes, raw.flushes, raw.afterFailure)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestLiveChatCompletionErrorPrecedence(t *testing.T) {
	writeErr := errors.New("writer cause")
	upstream := errors.New("independent upstream")
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	deadline, stop := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer stop()
	for _, tc := range []struct {
		name          string
		parent        context.Context
		planner, want error
	}{
		{"successful planner", context.Background(), nil, writeErr},
		{"writer child canceled", context.Background(), context.Canceled, writeErr},
		{"independent upstream", context.Background(), upstream, upstream},
		{"watchdog deadline", context.Background(), context.DeadlineExceeded, context.DeadlineExceeded},
		{"parent canceled", canceled, context.Canceled, context.Canceled},
		{"parent deadline", deadline, context.DeadlineExceeded, context.DeadlineExceeded},
		{"parent deadline swallowed", deadline, nil, context.DeadlineExceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := liveChatCompletionError(tc.parent, tc.planner, writeErr); got != tc.want {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
	if got := liveChatCompletionError(context.Background(), upstream, nil); got != upstream {
		t.Fatal("no-write-error path changed")
	}
}

// A real HTTPPlanner receives only tool-call deltas. No content sink invocation
// is needed for failed opening role, held keepalive, or heartbeat cancellation.
// Timers enforce the existing keepalive interval and bound waits; not a benchmark.
// fak-test:runtime medium est=8000ms lane=default
func TestLiveChatHeldAndHeartbeatWriteFailureCancelsHTTPPlanner(t *testing.T) {
	for _, mode := range []string{"role-error", "role-short", "held-error", "heartbeat-error"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv("FAK_STREAM_HEARTBEAT_S", "")
			t.Setenv("FAK_EP_FANOUT_ADDRS", "")
			if mode == "heartbeat-error" {
				t.Setenv("FAK_STREAM_HEARTBEAT_S", "1")
			}
			parent, cancel := context.WithCancel(context.Background())
			defer cancel()
			role := make(chan struct{})
			canceledUpstream := make(chan struct{})
			entered := make(chan struct{})
			var upstreamCalls atomic.Int32
			var canceledOnce sync.Once
			signalCanceled := func() { canceledOnce.Do(func() { close(canceledUpstream) }) }
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.ReadAll(r.Body)
				if upstreamCalls.Add(1) == 1 {
					close(entered)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				send := func(frame string) { _, _ = fmt.Fprintf(w, "data: %s\n\n", frame); w.(http.Flusher).Flush() }
				send(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"allow_x","arguments":""}}]}}]}`)
				if mode == "held-error" {
					select {
					case <-role:
					case <-r.Context().Done():
						signalCanceled()
						return
					}
					timer := time.NewTimer(heldKeepaliveEvery + 50*time.Millisecond)
					defer timer.Stop()
					select {
					case <-timer.C:
						send(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`)
					case <-r.Context().Done():
						signalCanceled()
						return
					}
				}
				<-r.Context().Done()
				signalCanceled()
			}))
			defer up.Close()
			// Cancel and join before closing the upstream even when a negative control
			// hits its timeout. The parent stays live throughout the positive witness.
			defer cancel()
			srv := newTestServer(t)
			srv.planner = agent.NewHTTPPlanner(up.URL, "test-model", "")
			var logMu sync.Mutex
			var logs []string
			srv.logf = func(format string, args ...any) {
				logMu.Lock()
				logs = append(logs, fmt.Sprintf(format, args...))
				logMu.Unlock()
			}
			cause := errors.New("held-client-write-cause")
			raw := &liveChatFailWriter{header: make(http.Header), cause: cause, role: role}
			switch mode {
			case "role-short":
				raw.short = true
			case "held-error":
				raw.failPrefix = ": fak-held"
			case "heartbeat-error":
				raw.failPrefix = ": fak-heartbeat"
			}
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test-model","messages":[{"role":"user","content":"do it"}],"tools":[{"type":"function","function":{"name":"allow_x"}}],"stream":true}`)).WithContext(parent)
			req.Header.Set("Content-Type", "application/json")
			done := make(chan struct{})
			go func() { defer close(done); srv.Handler().ServeHTTP(raw, req) }()
			defer func() {
				cancel()
				select {
				case <-done:
				case <-time.After(3 * time.Second):
					t.Error("handler did not join after cleanup cancellation")
				}
			}()
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("real upstream not called")
			}
			select {
			case <-done:
			case <-time.After(6 * time.Second):
				cancel()
				t.Fatal("failed held/heartbeat write did not stop planner")
			}
			select {
			case <-canceledUpstream:
			case <-time.After(3 * time.Second):
				t.Fatal("upstream request not canceled")
			}
			if parent.Err() != nil {
				t.Fatal("test parent cancellation caused the result")
			}
			raw.mu.Lock()
			failed, after, text, writes := raw.failed, raw.afterFailure, raw.body.String(), raw.writes
			raw.mu.Unlock()
			if !failed || after != 0 || strings.Contains(text, "[DONE]") || strings.Contains(text, `"tool_calls"`) {
				t.Fatalf("failure/later output mismatch: failed=%v after=%d body=%q", failed, after, text)
			}
			wantWrites := 1
			if mode == "held-error" || mode == "heartbeat-error" {
				wantWrites = 2
			}
			if writes != wantWrites {
				t.Fatalf("physical writes=%d want%d", writes, wantWrites)
			}
			want := cause.Error()
			if mode == "role-short" {
				want = io.ErrShortWrite.Error()
			}
			logMu.Lock()
			found := strings.Contains(strings.Join(logs, "\n"), want)
			logMu.Unlock()
			if upstreamCalls.Load() != 1 {
				t.Fatalf("upstream attempts=%d", upstreamCalls.Load())
			}
			if !found {
				t.Fatalf("gateway did not report original write cause %q", want)
			}
		})
	}
}

type liveChatCompletedAfterWriteFailure struct {
	calls      int
	sinkError  error
	childError error
}

func (*liveChatCompletedAfterWriteFailure) Model() string            { return "test-model" }
func (*liveChatCompletedAfterWriteFailure) StreamingSupported() bool { return true }
func (*liveChatCompletedAfterWriteFailure) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, errors.New("unexpected buffered fallback")
}
func (p *liveChatCompletedAfterWriteFailure) CompleteStream(ctx context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	p.sinkError = sink("generated text")
	p.childError = ctx.Err()
	// A completed generation may return successfully despite its client sink error.
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "generated text"}, FinishReason: "stop", Usage: agent.Usage{PromptTokens: 9, CompletionTokens: 4, TotalTokens: 13}}, nil
}

// fak-test:runtime fast est=20ms lane=default
func TestLiveChatCompletedGenerationSettlesOnceAfterWriteFailure(t *testing.T) {
	t.Setenv("FAK_STREAM_HEARTBEAT_S", "")
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	srv := newTestServer(t)
	planner := &liveChatCompletedAfterWriteFailure{}
	srv.planner = planner
	srv.admissionCtl = NewAdmissionController(AdmissionPolicy{TokenBudget: 10000, MaxNumSeqs: 1})
	srv.tokenRateGate = NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 10000}})
	cause := errors.New("completed client write failure")
	raw := &liveChatFailWriter{header: make(http.Header), cause: cause}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test-model","messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":16}`))
	req.Header.Set("Content-Type", "application/json")
	srv.Handler().ServeHTTP(raw, req)
	if planner.calls != 1 || planner.sinkError != cause || planner.childError != context.Canceled || req.Context().Err() != nil {
		t.Fatalf("planner failure closure: calls=%d sink=%v child=%v parent=%v", planner.calls, planner.sinkError, planner.childError, req.Context().Err())
	}
	snap := srv.tokenRateGate.Snapshot()
	if snap.Settled.InputTokens != 9 || snap.Settled.OutputTokens != 4 {
		t.Fatalf("completed usage not settled exactly once: %+v", snap)
	}
	if srv.admissionCtl.Stats().TokensInUse != 0 {
		t.Fatal("admission lease retained")
	}
	records, _, _ := srv.metrics.perfRecordsSnapshot()
	if len(records) != 1 || records[0].Error == "" || records[0].FinishReason != "error" {
		t.Fatalf("completion not recorded as one failure: %+v", records)
	}
	srv.metrics.inferenceMu.Lock()
	served := len(srv.metrics.inferReqs)
	srv.metrics.inferenceMu.Unlock()
	if served != 0 {
		t.Fatal("broken downstream counted as successful serving")
	}
	raw.mu.Lock()
	defer raw.mu.Unlock()
	if raw.writes != 1 || raw.afterFailure != 0 || raw.flushes != 0 {
		t.Fatalf("successful tail after failure: writes=%d after=%d flushes=%d", raw.writes, raw.afterFailure, raw.flushes)
	}
}
