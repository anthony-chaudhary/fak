package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// This is a real loopback HTTP handler/stream witness with a scripted planner,
// not GPU scheduling or throughput evidence. The writer barrier makes a stalled
// downstream deterministic; no network-speed or sleep-based speedup threshold.
type chatWriterBarrier struct {
	http.ResponseWriter
	mode                    string
	entered, release, wake  chan struct{}
	enterOnce, releaseOnce  sync.Once
	wakeOnce                sync.Once
	content                 bool
	active, late, zeroCalls atomic.Int32
	zeroAfterCancel         atomic.Int32
	returned, interrupted   atomic.Bool
}

func (w *chatWriterBarrier) wait() error {
	w.active.Add(1)
	defer w.active.Add(-1)
	if w.returned.Load() {
		w.late.Add(1)
	}
	w.enterOnce.Do(func() { close(w.entered) })
	select {
	case <-w.release:
		return nil
	case <-w.wake:
		return os.ErrDeadlineExceeded
	}
}

func (w *chatWriterBarrier) SetWriteDeadline(d time.Time) error {
	if w.mode == "unsupported" {
		return http.ErrNotSupported
	}
	if d.IsZero() {
		w.zeroCalls.Add(1)
		if w.interrupted.Load() {
			w.zeroAfterCancel.Add(1)
		}
	} else if !d.After(time.Now()) {
		w.interrupted.Store(true)
		w.wakeOnce.Do(func() { close(w.wake) })
	}
	return http.NewResponseController(w.ResponseWriter).SetWriteDeadline(d)
}

func (w *chatWriterBarrier) WriteHeader(code int) {
	if w.mode == "header" || w.mode == "prefill" {
		if w.wait() != nil {
			return
		}
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *chatWriterBarrier) Write(p []byte) (int, error) {
	if strings.Contains(string(p), `"content":`) {
		w.content = true
		if w.mode != "flush" && w.mode != "flush-error" && w.mode != "unsupported" {
			if err := w.wait(); err != nil {
				return 0, err
			}
		}
		if w.mode == "panic" {
			panic("controlled writer panic must not escape async request ownership")
		}
	}
	return w.ResponseWriter.Write(p)
}

func (w *chatWriterBarrier) FlushError() error {
	if w.content && (w.mode == "flush" || w.mode == "flush-error") {
		if err := w.wait(); err != nil {
			return err
		}
		if w.mode == "flush-error" {
			return errors.New("controlled downstream flush failure")
		}
	}
	return http.NewResponseController(w.ResponseWriter).Flush()
}

func (w *chatWriterBarrier) Flush() { _ = w.FlushError() }

// A shared emission lock models the caller-side serialization the real native
// decode loop has around its sink call. It does not simulate model arithmetic.
type chatSharedSinkPlanner struct {
	mu       sync.Mutex
	barrier  *chatWriterBarrier
	mode     string
	produced chan struct{}
}

func (*chatSharedSinkPlanner) Model() string            { return "test-model" }
func (*chatSharedSinkPlanner) StreamingSupported() bool { return true }
func (*chatSharedSinkPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, agent.ErrStreamingUnsupported
}

func (p *chatSharedSinkPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if messages[len(messages)-1].Content == "peer" {
		if err := sink("peer"); err != nil {
			return nil, err
		}
		return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "peer"}, FinishReason: "stop", Model: "test-model"}, nil
	}
	defer close(p.produced)
	if p.mode == "prefill" {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	if err := sink("alpha"); err != nil {
		return nil, err
	}
	if p.mode != "unsupported" {
		select {
		case <-p.barrier.entered:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if p.mode == "cancel" || p.mode == "header" {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	if p.mode == "bytes" || p.mode == "events" {
		fragment, count := strings.Repeat("x", (1<<20)+1), 1
		if p.mode == "events" {
			fragment, count = "x", 1025
		}
		for i := 0; i < count; i++ {
			if err := sink(fragment); err != nil {
				return nil, err
			}
		}
		return nil, errors.New("bounded stream unexpectedly accepted excess output")
	}
	pieces := []string{" beta", " gamma"}
	if p.mode == "guard" {
		pieces = []string{" \xe2", "\x82\xac", `<tool_call>{"name":"deny_x","arguments":{"secret":"DO_NOT_LEAK"}}</tool_call>`}
	}
	text := "alpha"
	for _, piece := range pieces {
		if err := sink(piece); err != nil {
			return nil, err
		}
		text += piece
	}
	message := agent.LiftTextToolCalls(agent.Message{Role: agent.RoleAssistant, Content: text}, agent.NewOfferedTools(tools))
	return &agent.Completion{Message: message, Model: "test-model", FinishReason: "stop", Usage: agent.Usage{PromptTokens: 4, CompletionTokens: 3}}, nil
}

type chatBarrierResult struct {
	body []byte
	err  error
}

// fak-test:runtime fast est=1s lane=default
func TestLiveChatSlowConsumerIsolation(t *testing.T) {
	t.Setenv("FAK_STREAM_HEARTBEAT_S", "0")
	t.Setenv("FAK_STREAM_PREFILL_KEEPALIVE_S", "0")
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, mode := range []string{"write", "flush", "guard", "cancel", "header", "prefill", "bytes", "events", "flush-error", "panic", "unsupported"} {
		t.Run(mode, func(t *testing.T) {
			if mode == "prefill" {
				t.Setenv("FAK_STREAM_PREFILL_KEEPALIVE_S", "5")
			}
			barrier := &chatWriterBarrier{mode: mode, entered: make(chan struct{}), release: make(chan struct{}), wake: make(chan struct{})}
			planner := &chatSharedSinkPlanner{barrier: barrier, mode: mode, produced: make(chan struct{})}
			srv := newTestServer(t)
			srv.planner = planner
			handler := srv.Handler()
			returned := make(chan struct{})
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("X-Test-Slow-Stream") == "1" {
					barrier.ResponseWriter = w
					handler.ServeHTTP(barrier, r)
					barrier.returned.Store(true)
					close(returned)
					return
				}
				handler.ServeHTTP(w, r)
			}))
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(func() {
				cancel()
				barrier.releaseOnce.Do(func() { close(barrier.release) })
				ts.CloseClientConnections()
				ts.Close()
			})
			request := func(slow bool) <-chan chatBarrierResult {
				result := make(chan chatBarrierResult, 1)
				go func() {
					var output []byte
					message := "peer"
					if slow {
						message = "slow"
					}
					body, _ := json.Marshal(map[string]any{
						"model": "test-model", "messages": []agent.Message{{Role: agent.RoleUser, Content: message}},
						"tools": offeredWireDefs("deny_x"), "stream": true, "stream_options": map[string]bool{"include_usage": true},
					})
					req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(string(body)))
					if err == nil {
						req.Header.Set("Content-Type", "application/json")
						if slow {
							req.Header.Set("X-Test-Slow-Stream", "1")
						}
						var resp *http.Response
						resp, err = ts.Client().Do(req)
						if err == nil {
							output, err = io.ReadAll(resp.Body)
							_ = resp.Body.Close()
						}
					}
					result <- chatBarrierResult{body: output, err: err}
				}()
				return result
			}
			await := func(ch <-chan struct{}, reason string) {
				t.Helper()
				select {
				case <-ch:
				case <-time.After(2 * time.Second):
					t.Fatal(reason)
				}
			}
			read := func(ch <-chan chatBarrierResult) chatBarrierResult {
				t.Helper()
				select {
				case got := <-ch:
					return got
				case <-time.After(2 * time.Second):
					t.Fatal("HTTP result did not terminate")
					return chatBarrierResult{}
				}
			}
			slow := request(true)
			if mode != "unsupported" {
				await(barrier.entered, "selected real handler never reached controlled downstream I/O")
			}
			if mode == "header" || mode == "prefill" {
				if n := barrier.zeroCalls.Load(); n != 1 {
					t.Errorf("initial SSE commit cleared the live deadline: zero calls=%d, want capability probe only", n)
				}
			}
			success := mode == "write" || mode == "flush" || mode == "guard" || mode == "unsupported"
			if success {
				peer := read(request(false))
				if peer.err != nil || !strings.Contains(string(peer.body), `"content":"peer"`) {
					t.Fatalf("peer could not progress independently of blocked writer: %v %s", peer.err, peer.body)
				}
				await(planner.produced, "slow writer still owns the generator")
				barrier.releaseOnce.Do(func() { close(barrier.release) })
			} else if mode == "cancel" || mode == "header" || mode == "prefill" {
				cancel()
			} else if mode == "flush-error" || mode == "panic" {
				barrier.releaseOnce.Do(func() { close(barrier.release) })
			}
			await(returned, "stream did not cancel/join its writer before handler return")
			await(planner.produced, "stream failed to release its generator")
			got := read(slow)
			if barrier.active.Load() != 0 || barrier.late.Load() != 0 || barrier.zeroAfterCancel.Load() != 0 {
				t.Fatalf("writer escaped handler or rearmed cancellation: active=%d late=%d clears=%d", barrier.active.Load(), barrier.late.Load(), barrier.zeroAfterCancel.Load())
			}
			if !success {
				if strings.Contains(string(got.body), "data: [DONE]") {
					t.Fatalf("failed downstream emitted successful terminal frames: %s", got.body)
				}
				return
			}
			if got.err != nil {
				t.Fatal(got.err)
			}
			chunks, done := parseSSEChunks(t, got.body)
			if !done || len(chunks) == 0 || chunks[0].Choices[0].Delta.Role != "assistant" {
				t.Fatalf("invalid role/terminal order: %s", got.body)
			}
			var text strings.Builder
			var fragments []string
			usage, finish := 0, ""
			for _, chunk := range chunks {
				if chunk.Usage != nil {
					usage++
					if chunk.Usage.CompletionTokens != 3 || len(chunk.Choices) != 0 {
						t.Fatalf("usage shape changed: %+v", chunk)
					}
				}
				for _, choice := range chunk.Choices {
					if choice.Delta.Content != "" {
						fragments = append(fragments, choice.Delta.Content)
						text.WriteString(choice.Delta.Content)
					}
					if len(choice.Delta.ToolCalls) != 0 {
						t.Fatal("unadjudicated/denied tool call reached client")
					}
					if choice.FinishReason != nil {
						finish = *choice.FinishReason
					}
				}
			}
			if usage != 1 || finish != "stop" {
				t.Fatalf("terminal usage/finish changed: %d/%s", usage, finish)
			}
			if mode == "guard" {
				if !strings.HasPrefix(text.String(), "alpha €") || strings.Contains(string(got.body), "DO_NOT_LEAK") || strings.Contains(text.String(), "<tool_call>") {
					t.Fatalf("UTF8/tool quarantine changed: %s", got.body)
				}
			} else if text.String() != "alpha beta gamma" || len(fragments) != 3 || fragments[0] != "alpha" || fragments[1] != " beta" || fragments[2] != " gamma" {
				t.Fatalf("per-token fragments collapsed/dropped/replayed: %q", fragments)
			}
		})
	}
}
