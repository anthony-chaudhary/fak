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

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestAnthropicMessagesPlannerStreamEmitsTextBeforeToolGate(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	contentSent := make(chan struct{})
	releaseTools := make(chan struct{})
	var upstreamBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		var req struct {
			Stream bool `json:"stream"`
		}
		_ = json.Unmarshal(upstreamBody, &req)
		if !req.Stream {
			t.Errorf("gateway did not ask the upstream planner to stream: %s", upstreamBody)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		_, _ = io.WriteString(w,
			`data: {"model":"served-openai","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"content":"checking"}}]}`+"\n\n")
		flusher.Flush()
		close(contentSent)
		select {
		case <-releaseTools:
		case <-time.After(2 * time.Second):
			t.Error("test timed out waiting to release tool-call deltas")
			return
		}
		_, _ = io.WriteString(w,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a1","type":"function","function":{"name":"allow_a","arguments":"{\"x\":1}"}}]}}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"d1","type":"function","function":{"name":"deny_b","arguments":"{\"secret\":\"nope\"}"}}]}}]}`+"\n\n"+
				`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":8,"completion_tokens":5,"total_tokens":13}}`+"\n\n"+
				"data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "x:model", BaseURL: upstream.URL + "/compat", Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	inbound := []byte(`{"model":"claude-client","max_tokens":256,"stream":true,` +
		`"tools":[{"name":"allow_a","input_schema":{"type":"object"}},{"name":"deny_b","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"call tools"}]}`)
	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, raw)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "event-stream") {
		t.Fatalf("content-type = %q, want event-stream", ct)
	}

	lines := make(chan string, 128)
	go func() {
		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for sc.Scan() {
			lines <- sc.Text()
		}
		close(lines)
	}()

	select {
	case <-contentSent:
	case <-time.After(2 * time.Second):
		t.Fatal("upstream never emitted the first content delta")
	}

	var beforeRelease []string
	sawText := false
	for !sawText {
		select {
		case line, ok := <-lines:
			if !ok {
				t.Fatalf("stream ended before live text_delta; lines=%q", strings.Join(beforeRelease, "\n"))
			}
			beforeRelease = append(beforeRelease, line)
			if strings.Contains(line, `"text_delta"`) && strings.Contains(line, "checking") {
				sawText = true
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("client did not receive text_delta before tool release; lines=%q", strings.Join(beforeRelease, "\n"))
		}
	}
	if early := strings.Join(beforeRelease, "\n"); strings.Contains(early, "allow_a") || strings.Contains(early, "input_json_delta") {
		t.Fatalf("tool-call bytes reached the Anthropic client before adjudication:\n%s", early)
	}

	close(releaseTools)
	allLines := append([]string{}, beforeRelease...)
	for line := range lines {
		allLines = append(allLines, line)
	}
	body := strings.Join(allLines, "\n")
	if !strings.Contains(body, `"allow_a"`) || !strings.Contains(body, `"input_json_delta"`) {
		t.Fatalf("adjudicated allowed tool call did not reach the Anthropic stream:\n%s", body)
	}
	for _, leak := range []string{`"name":"deny_b"`, "nope", `"secret"`} {
		if strings.Contains(body, leak) {
			t.Fatalf("denied tool-call bytes leaked into the Anthropic stream (%q):\n%s", leak, body)
		}
	}
	if !strings.Contains(body, `"stop_reason":"tool_use"`) {
		t.Fatalf("surviving tool call should keep stop_reason tool_use:\n%s", body)
	}
	if !strings.Contains(body, `"input_tokens":8`) || !strings.Contains(body, `"output_tokens":5`) {
		t.Fatalf("terminal usage from the streamed planner was not forwarded:\n%s", body)
	}
}

// This recorder witnesses the public Messages handler's selected planner options.
// Its content callbacks are software controls, not native latency measurements.
type messagesPerTokenPlanner struct {
	mu                           sync.Mutex
	supported                    bool
	streamParams, bufferedParams []agent.SampleParams
}

func (p *messagesPerTokenPlanner) Model() string            { return "test-model" }
func (p *messagesPerTokenPlanner) StreamingSupported() bool { return p.supported }
func (p *messagesPerTokenPlanner) record(stream bool, opts []agent.SampleOpt) {
	var sp agent.SampleParams
	for _, opt := range opts {
		opt(&sp)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if stream {
		p.streamParams = append(p.streamParams, sp)
	} else {
		p.bufferedParams = append(p.bufferedParams, sp)
	}
}
func messagesPerTokenCompletion() *agent.Completion {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "Hello world"}, FinishReason: "stop", Usage: agent.Usage{PromptTokens: 2, CompletionTokens: 2, TotalTokens: 4}}
}
func (p *messagesPerTokenPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.record(false, opts)
	return messagesPerTokenCompletion(), nil
}
func (p *messagesPerTokenPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.record(true, opts)
	for _, text := range []string{"Hello ", "world"} {
		if err := sink(text); err != nil {
			return nil, err
		}
	}
	return messagesPerTokenCompletion(), nil
}

// fak-test:runtime medium est=1s lane=default
// Unmeasured estimate. Loopback HTTP/SSE and a scripted planner; no native model.
func TestAnthropicMessagesNativePerTokenOptionIsLiveOnly(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, tc := range []struct {
		name              string
		stream, supported bool
	}{
		{"live", true, true}, {"unsupported-stream-fallback", true, false}, {"buffered", false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			p := &messagesPerTokenPlanner{supported: tc.supported}
			srv.planner = p
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			body := fmt.Sprintf(`{"model":"test-model","max_tokens":64,"stream":%t,"messages":[{"role":"user","content":"hi"}]}`, tc.stream)
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/messages", strings.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			raw, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status=%d body=%s", resp.StatusCode, raw)
			}
			p.mu.Lock()
			streams, buffered := append([]agent.SampleParams(nil), p.streamParams...), append([]agent.SampleParams(nil), p.bufferedParams...)
			p.mu.Unlock()
			if tc.stream && tc.supported {
				if len(streams) != 1 || len(buffered) != 0 || streams[0].PerTokenStream == nil || !*streams[0].PerTokenStream {
					t.Fatalf("missing selected live override: stream=%+v buffered=%+v", streams, buffered)
				}
			} else if len(streams) != 0 || len(buffered) != 1 || buffered[0].PerTokenStream != nil {
				t.Fatalf("buffered/fallback options changed: stream=%+v buffered=%+v", streams, buffered)
			}
			if tc.stream {
				var text string
				deltas, terminal := 0, 0
				for _, line := range strings.Split(string(raw), "\n") {
					if !strings.HasPrefix(line, "data: ") {
						continue
					}
					var event struct {
						Type  string                                  `json:"type"`
						Delta struct{ Type, Text, StopReason string } `json:"delta"`
					}
					if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
						t.Fatal(err)
					}
					if event.Type == "content_block_delta" && event.Delta.Type == "text_delta" {
						text += event.Delta.Text
						deltas++
					}
					if event.Type == "message_stop" {
						terminal++
					}
				}
				if text != "Hello world" || terminal != 1 || (tc.supported && deltas < 2) {
					t.Fatalf("SSE content/terminal changed: text=%q deltas=%d stops=%d body=%s", text, deltas, terminal, raw)
				}
			} else {
				var result struct {
					Content    []struct{ Type, Text string } `json:"content"`
					StopReason string                        `json:"stop_reason"`
				}
				if err := json.Unmarshal(raw, &result); err != nil {
					t.Fatal(err)
				}
				if len(result.Content) != 1 || result.Content[0].Text != "Hello world" || result.StopReason != "end_turn" {
					t.Fatalf("buffered response changed: %s", raw)
				}
			}
		})
	}
}

// messagesCheckedWriter records actual serialized writes. The atomic entry guard
// detects overlap without hiding it behind the recorder mutex.
type messagesCheckedWriter struct {
	mu              sync.Mutex
	active          atomic.Int32
	overlap         atomic.Bool
	header          http.Header
	body            bytes.Buffer
	writes, flushes int
	failAt          int
	cause           error
	short           bool
	before          func([]byte)
}

func (w *messagesCheckedWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}
func (w *messagesCheckedWriter) WriteHeader(int) {}
func (w *messagesCheckedWriter) Write(b []byte) (int, error) {
	if w.active.Add(1) != 1 {
		w.overlap.Store(true)
	}
	defer w.active.Add(-1)
	if w.before != nil {
		w.before(b)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.writes++
	if w.writes == w.failAt {
		if w.short {
			return len(b) - 1, nil
		}
		return 0, w.cause
	}
	return w.body.Write(b)
}
func (w *messagesCheckedWriter) Flush() { w.mu.Lock(); w.flushes++; w.mu.Unlock() }

type messagesCheckedPlanner struct {
	run             func(context.Context, agent.StreamSink) (*agent.Completion, error)
	calls, fallback int
}

func (*messagesCheckedPlanner) Model() string            { return "test-model" }
func (*messagesCheckedPlanner) StreamingSupported() bool { return true }
func (p *messagesCheckedPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	p.fallback++
	return messagesPerTokenCompletion(), nil
}
func (p *messagesCheckedPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	return p.run(ctx, sink)
}
func messagesCheckedRequest(ctx context.Context) *http.Request {
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"test-model","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)).WithContext(ctx)
	r.Header.Set("Content-Type", "application/json")
	return r
}

// fak-test:runtime fast est=10ms lane=default
// Unmeasured estimate. Single-buffer wire, marshal and short-write source controls.
func TestAnthropicCheckedSenderPreservesWireAndFailure(t *testing.T) {
	cause := errors.New("writer failed")
	for _, short := range []bool{false, true} {
		w := &messagesCheckedWriter{failAt: 1, cause: cause, short: short}
		err := anthropicSSECheckedSender(w, w)("ping", map[string]string{"type": "ping"})
		want := cause
		if short {
			want = io.ErrShortWrite
		}
		if err != want || w.writes != 1 || w.flushes != 0 {
			t.Fatalf("err=%v writes=%d flushes=%d", err, w.writes, w.flushes)
		}
	}
	w := &messagesCheckedWriter{}
	if err := anthropicSSECheckedSender(w, w)("ping", map[string]string{"type": "ping"}); err != nil {
		t.Fatal(err)
	}
	if w.body.String() != "event: ping\ndata: {\"type\":\"ping\"}\n\n" || w.flushes != 1 {
		t.Fatal("wire changed")
	}
	if err := anthropicSSECheckedSender(w, w)("bad", make(chan int)); err == nil || w.writes != 1 {
		t.Fatal("marshal error wrote bytes")
	}
}

// fak-test:runtime medium est=1s lane=default
// Unmeasured estimate. Real handler, live parent context, software writer faults.
func TestAnthropicLiveWriterFailuresAbortSelectedPlanner(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, short := range []bool{false, true} {
		for failAt := 0; failAt <= 7; failAt++ {
			t.Run(fmt.Sprintf("short=%v/write=%d", short, failAt), func(t *testing.T) {
				srv := newTestServer(t)
				cause := errors.New("client write failed")
				var sinkErr error
				p := &messagesCheckedPlanner{run: func(ctx context.Context, sink agent.StreamSink) (*agent.Completion, error) {
					for _, part := range []string{"Hello ", "world"} {
						if err := sink(part); err != nil {
							sinkErr = err
							if ctx.Err() != context.Canceled {
								t.Error("writer did not cancel child")
							}
							return nil, err
						}
					}
					return messagesPerTokenCompletion(), nil
				}}
				srv.planner = p
				w := &messagesCheckedWriter{failAt: failAt, cause: cause, short: short}
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				srv.Handler().ServeHTTP(w, messagesCheckedRequest(ctx))
				wantWrites := failAt
				if failAt == 0 {
					wantWrites = 7
				}
				wantCalls := 1
				if failAt == 1 {
					wantCalls = 0
				}
				if w.writes != wantWrites || p.calls != wantCalls || p.fallback != 0 || w.overlap.Load() || ctx.Err() != nil {
					t.Fatalf("writes=%d calls=%d/%d context=%v overlap=%v", w.writes, p.calls, p.fallback, ctx.Err(), w.overlap.Load())
				}
				if failAt >= 2 && failAt <= 4 {
					want := cause
					if short {
						want = io.ErrShortWrite
					}
					if sinkErr != want {
						t.Fatalf("sink=%v want=%v", sinkErr, want)
					}
				}
				if failAt != 0 && strings.Contains(w.body.String(), "event: message_stop") {
					t.Fatal("failed writer received success terminal")
				}
				if failAt == 0 && !strings.Contains(w.body.String(), "event: message_stop") {
					t.Fatal("clean terminal missing")
				}
			})
		}
	}
}

// fak-test:runtime medium est=1s lane=default
// Unmeasured estimate. Non-parallel ticker override follows existing gateway tests;
// channels, rather than elapsed sleeps, prove ping/child/writer ownership.
func TestAnthropicPingWriteFailureCancelsAndJoins(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	old := anthropicStreamPingInterval
	anthropicStreamPingInterval = time.Millisecond
	defer func() { anthropicStreamPingInterval = old }()
	for _, mode := range []string{"canceled", "swallowed", "independent-error", "deadline"} {
		t.Run(mode, func(t *testing.T) {
			srv := newTestServer(t)
			cause := errors.New("ping writer failed")
			upstream := errors.New("independent upstream failure")
			var reported error
			srv.logf = func(format string, args ...any) {
				if strings.HasPrefix(format, "gateway: upstream model error mid-stream (messages):") && len(args) == 1 {
					reported, _ = args[0].(error)
				}
			}
			parent, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			entered := make(chan struct{})
			pingEntered := make(chan struct{})
			sinkAttempt := make(chan struct{})
			releasePing := make(chan struct{})
			var releaseOnce sync.Once
			defer releaseOnce.Do(func() { close(releasePing) })
			var once sync.Once
			var childCause error
			p := &messagesCheckedPlanner{run: func(ctx context.Context, sink agent.StreamSink) (*agent.Completion, error) {
				close(entered)
				select {
				case <-pingEntered:
				case <-ctx.Done():
					return nil, ctx.Err()
				}
				close(sinkAttempt)
				if err := sink("concurrent token"); err != cause {
					t.Errorf("concurrent sink=%v", err)
				}
				<-ctx.Done()
				childCause = context.Cause(ctx)
				if err := sink("must not write"); err != cause {
					t.Errorf("sink after ping=%v", err)
				}
				switch mode {
				case "swallowed":
					return messagesPerTokenCompletion(), nil
				case "independent-error":
					return nil, upstream
				case "deadline":
					return nil, context.DeadlineExceeded
				default:
					return nil, ctx.Err()
				}
			}}
			srv.planner = p
			w := &messagesCheckedWriter{failAt: 2, cause: cause, before: func(b []byte) {
				if bytes.HasPrefix(b, []byte("event: ping\n")) {
					select {
					case <-entered:
					case <-parent.Done():
						return
					}
					once.Do(func() { close(pingEntered) })
					select {
					case <-releasePing:
					case <-parent.Done():
					}
				}
			}}
			done := make(chan struct{})
			go func() { defer close(done); srv.Handler().ServeHTTP(w, messagesCheckedRequest(parent)) }()
			defer func() {
				cancel()
				releaseOnce.Do(func() { close(releasePing) })
				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Error("handler did not stop during cleanup")
				}
			}()
			select {
			case <-sinkAttempt:
				releaseOnce.Do(func() { close(releasePing) })
			case <-parent.Done():
				t.Fatal("ping never reached writer")
			}
			select {
			case <-done:
			case <-parent.Done():
				t.Fatal("ping cancellation/join did not finish")
			}
			wantReported := cause
			if mode == "independent-error" {
				wantReported = upstream
			}
			if mode == "deadline" {
				wantReported = context.DeadlineExceeded
			}
			if reported != wantReported {
				t.Fatalf("reported=%v want exact %v", reported, wantReported)
			}
			if childCause != cause || parent.Err() != nil || w.writes != 2 || w.flushes != 1 || w.overlap.Load() || p.calls != 1 || p.fallback != 0 {
				t.Fatalf("cause=%v parent=%v writes=%d flushes=%d calls=%d/%d overlap=%v", childCause, parent.Err(), w.writes, w.flushes, p.calls, p.fallback, w.overlap.Load())
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
// Unmeasured estimate. A planner swallowing a content sink error cannot succeed.
func TestAnthropicLiveSwallowedWriterFailureDoesNotComplete(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	srv := newTestServer(t)
	cause := errors.New("delta failed")
	var got error
	p := &messagesCheckedPlanner{run: func(_ context.Context, sink agent.StreamSink) (*agent.Completion, error) {
		got = sink("Hello world")
		return messagesPerTokenCompletion(), nil
	}}
	srv.planner = p
	w := &messagesCheckedWriter{failAt: 3, cause: cause}
	srv.Handler().ServeHTTP(w, messagesCheckedRequest(context.Background()))
	if got != cause || w.writes != 3 || p.calls != 1 || p.fallback != 0 || strings.Contains(w.body.String(), "message_stop") {
		t.Fatalf("sink=%v writes=%d calls=%d/%d", got, w.writes, p.calls, p.fallback)
	}
}

// fak-test:runtime medium est=1s lane=default
// Unmeasured estimate. Upstream error tails retain their existing cause and dialect.
func TestAnthropicLiveUpstreamErrorAndFinalRemainder(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, tc := range []struct {
		name     string
		upstream bool
		failAt   int
	}{
		{"upstream-clean", true, 0}, {"upstream-close-failure", true, 4}, {"upstream-terminal-failure", true, 5}, {"remainder-failure", false, 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			upstream := errors.New("upstream failed")
			var reported error
			srv.logf = func(format string, args ...any) {
				if strings.HasPrefix(format, "gateway: upstream model error mid-stream (messages):") && len(args) == 1 {
					reported, _ = args[0].(error)
				}
			}
			p := &messagesCheckedPlanner{run: func(_ context.Context, sink agent.StreamSink) (*agent.Completion, error) {
				if err := sink("Hello "); err != nil {
					return nil, err
				}
				if tc.upstream {
					return nil, upstream
				}
				return messagesPerTokenCompletion(), nil
			}}
			srv.planner = p
			w := &messagesCheckedWriter{failAt: tc.failAt, cause: errors.New("tail failed")}
			srv.Handler().ServeHTTP(w, messagesCheckedRequest(context.Background()))
			want := tc.failAt
			if want == 0 {
				want = 5
			}
			if w.writes != want || p.calls != 1 || p.fallback != 0 || strings.Contains(w.body.String(), "message_stop") {
				t.Fatalf("writes=%d calls=%d/%d", w.writes, p.calls, p.fallback)
			}
			if tc.upstream && reported != upstream {
				t.Fatalf("reported=%v want original upstream", reported)
			}
			if tc.failAt == 0 && !strings.Contains(w.body.String(), "event: error\n") {
				t.Fatal("healthy writer lost upstream error event")
			}
		})
	}
}
