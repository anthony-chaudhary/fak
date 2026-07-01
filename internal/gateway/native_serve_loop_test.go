package gateway

// native_serve_loop_test.go — the acceptance witness for the native-harness keystone
// (#1316): a multi-turn /v1/messages posted to a `serve --native` gateway is driven by
// fak's OWN agent loop (agent.RunArm), NOT the single-shot proxy turn. The test proves
// three things the program's definition-of-done names:
//
//  1. a LIVE, non-test RunArm caller on the serve path — serveNativeMessages →
//     runNativeArm → agent.RunArm (grep-able; this is the first such caller);
//  2. the response carries RunArm-only ArmMetrics (Turns > 1, a real final answer) on
//     the `fak.native_arm` extension; and
//  3. the session gate fired at EACH turn boundary on the served trace — the
//     WithSessionGate twin of WithSessionTable, counted against the same session.Table
//     the proxy path uses via the injected DecideSession hook.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

type nativeStreamingPlanner struct {
	turns         int32
	firstDelta    chan struct{}
	releaseSecond chan struct{}
}

func (p *nativeStreamingPlanner) Model() string { return "native-stream" }

func (p *nativeStreamingPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return p.complete(ctx, nil, messages, tools, opts...)
}

func (p *nativeStreamingPlanner) StreamingSupported() bool { return true }

func (p *nativeStreamingPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return p.complete(ctx, sink, messages, tools, opts...)
}

func (p *nativeStreamingPlanner) complete(ctx context.Context, sink agent.StreamSink, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	_ = ctx
	turn := atomic.AddInt32(&p.turns, 1)
	switch turn {
	case 1:
		return &agent.Completion{
			Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{
				ID: "call_user", Type: "function", Function: agent.Func{Name: "get_user_details", Arguments: `{"user_id":"mia_li_3668"}`},
			}}},
			FinishReason: "tool_calls",
			Usage:        agent.Usage{PromptTokens: len(messages) * 4, CompletionTokens: 2},
		}, nil
	default:
		if sink != nil {
			if err := sink("Booked"); err != nil {
				return nil, err
			}
			if p.firstDelta != nil {
				close(p.firstDelta)
			}
			if p.releaseSecond != nil {
				select {
				case <-p.releaseSecond:
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}
			if err := sink(" direct."); err != nil {
				return nil, err
			}
		}
		return &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "Booked direct."},
			FinishReason: "stop",
			Usage:        agent.Usage{PromptTokens: len(messages) * 4, CompletionTokens: 3},
		}, nil
	}
}

func readAnthropicSSEUntil(t *testing.T, r *bufio.Reader, stop func(sseFrame) bool) []sseFrame {
	t.Helper()
	var frames []sseFrame
	var ev, data string
	flush := func() bool {
		if data == "" {
			ev, data = "", ""
			return false
		}
		frame := sseFrame{event: ev, data: data}
		frames = append(frames, frame)
		ev, data = "", ""
		return stop(frame)
	}
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			t.Fatalf("read SSE before stop condition: %v", err)
		}
		line = strings.TrimRight(line, "\r\n")
		switch {
		case line == "":
			if flush() {
				return frames
			}
		case strings.HasPrefix(line, "event:"):
			ev = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			data += strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
	}
}

func TestNativeServeLoopDrivesRunArmWithPerTurnSessionGate(t *testing.T) {
	// RunArm(fak=true) creates its own kernel.New("localtools") and calls Configure(),
	// which registers the localtools engine + the agent policy. Configure here too so the
	// gateway's New(EngineID:"localtools") validates against a registered engine, and make
	// sure a region backend is present for the syscall Ref resolver regardless of any
	// prior test's abi.ResetForTest (blob's init registers the default, but a reset wipes
	// it). inlineBackend is the package test backend newTestServer uses.
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})

	// The SAME process-local drive-state table the host (cmd/fak serveSessions) wires; the
	// native loop gates each turn boundary on it through the injected DecideSession hook,
	// exactly as the proxy request boundary does. decideCount/debitCount count the gate
	// firings so we can assert one Decide per turn.
	tbl := session.NewTable()
	var decideCount, debitCount int32
	const trace = "native-serve-trace"

	srv, err := New(Config{
		EngineID:       "localtools",
		Model:          "test-model",
		VDSO:           true,
		Native:         true,
		NativeMaxTurns: 24,
		DecideSession: func(_ context.Context, tr string) SessionVerdict {
			atomic.AddInt32(&decideCount, 1)
			v := tbl.Decide(tr)
			return SessionVerdict{Proceed: v.Proceed, MaxTokens: v.MaxTokens, MinGapMs: v.MinGapMs, Stop: v.Stop, Reason: v.Reason}
		},
		DebitSession: func(_ context.Context, tr string, u SessionUsage) SessionState {
			atomic.AddInt32(&debitCount, 1)
			st := tbl.DebitUsage(tr, session.Usage{OutputTokens: u.CompletionTokens, ContextTokens: u.ContextTokens})
			return SessionState{TraceID: st.TraceID, Rev: st.Rev}
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	// Pin the deterministic AgentDojo planner so the owned loop runs the multi-turn
	// airline flow (get_user → … → book_flight → final answer) under test, independent of
	// whatever default New picked.
	srv.planner = agent.NewMockPlanner("test-model")

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{
		"model":      "test-model",
		"max_tokens": 256,
		"messages":   []map[string]string{{"role": "user", "content": "Book me a direct flight."}},
	})
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", trace)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/messages: %v", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, raw)
	}

	var got anthropicMessageResponse
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode response: %v; body=%s", err, raw)
	}

	// (2) RunArm-only ArmMetrics rode back on the response.
	if got.Fak == nil || got.Fak.NativeArm == nil {
		t.Fatalf("response carried no fak.native_arm — the owned loop's ArmMetrics is the witness it drove the turn; body=%s", raw)
	}
	arm := got.Fak.NativeArm
	if arm.Turns < 2 {
		t.Fatalf("native_arm.turns = %d, want >= 2 (a multi-turn owned loop, not a single-shot proxy turn)", arm.Turns)
	}
	if arm.FinalAnswer == "" {
		t.Fatalf("native_arm carried no final answer — the loop must run to a model final answer")
	}
	if arm.Arm != "fak" {
		t.Fatalf("native_arm.arm = %q, want \"fak\" (the kernel-mediated arm)", arm.Arm)
	}

	// (3) the session gate fired at EACH turn boundary on the served trace. RunArm calls
	// gateTurn (→ DecideSession → tbl.Decide) once per turn and debitTurn (→ DebitSession)
	// once per turn, so both counts equal the turn count exactly.
	if dc := atomic.LoadInt32(&decideCount); int(dc) != arm.Turns {
		t.Fatalf("DecideSession fired %d times, want %d (once per owned-loop turn boundary)", dc, arm.Turns)
	}
	if bc := atomic.LoadInt32(&debitCount); int(bc) != arm.Turns {
		t.Fatalf("DebitSession fired %d times, want %d (once per owned-loop turn)", bc, arm.Turns)
	}
	// The live table recorded the activity: an unseen trace's Rev advanced past zero, so
	// the gate ran against real drive state, not a no-op.
	if st := tbl.Get(trace); st.Rev == 0 {
		t.Fatalf("session table Rev for %q is 0 — the per-turn gate never touched the live drive state", trace)
	}
}

// TestNativeServeLoopOffByDefault is the negative witness: with Native unset the
// /v1/messages path stays the single-shot proxy turn and carries no native_arm, so the
// keystone is strictly opt-in and the proxy path is unchanged.
func TestNativeServeLoopOffByDefault(t *testing.T) {
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})

	srv, err := New(Config{EngineID: "localtools", Model: "test-model", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	srv.planner = agent.NewMockPlanner("test-model")

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{
		"model":      "test-model",
		"max_tokens": 256,
		"messages":   []map[string]string{{"role": "user", "content": "Book me a direct flight."}},
	})
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/messages: %v", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, raw)
	}
	var got anthropicMessageResponse
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode response: %v; body=%s", err, raw)
	}
	if got.Fak != nil && got.Fak.NativeArm != nil {
		t.Fatalf("proxy path (Native unset) must not carry native_arm; got %+v", got.Fak.NativeArm)
	}
}

func TestNativeServeLoopStreamsRunArmDeltasAndMetrics(t *testing.T) {
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})

	tbl := session.NewTable()
	var decideCount int32
	const trace = "native-stream-trace"

	srv, err := New(Config{
		EngineID:       "localtools",
		Model:          "test-model",
		VDSO:           true,
		Native:         true,
		NativeMaxTurns: 8,
		DecideSession: func(_ context.Context, tr string) SessionVerdict {
			atomic.AddInt32(&decideCount, 1)
			v := tbl.Decide(tr)
			return SessionVerdict{Proceed: v.Proceed, MaxTokens: v.MaxTokens, MinGapMs: v.MinGapMs, Stop: v.Stop, Reason: v.Reason}
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	planner := &nativeStreamingPlanner{firstDelta: make(chan struct{}), releaseSecond: make(chan struct{})}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{
		"model":      "test-model",
		"max_tokens": 256,
		"stream":     true,
		"messages":   []map[string]string{{"role": "user", "content": "Book me a direct flight."}},
	})
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", trace)
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/messages stream: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, raw)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}

	reader := bufio.NewReader(resp.Body)
	frames := readAnthropicSSEUntil(t, reader, func(frame sseFrame) bool {
		var obj map[string]any
		if err := json.Unmarshal([]byte(frame.data), &obj); err != nil {
			t.Fatalf("decode frame %q: %v", frame.data, err)
		}
		if obj["type"] != "content_block_delta" {
			return false
		}
		delta, _ := obj["delta"].(map[string]any)
		return delta["type"] == "text_delta" && delta["text"] == "Booked"
	})
	select {
	case <-planner.firstDelta:
	default:
		t.Fatal("test planner had not reached its first delta before the client observed it")
	}
	close(planner.releaseSecond)
	frames = append(frames, readAnthropicSSE(t, reader)...)
	var text string
	var stopHasArm bool
	for _, frame := range frames {
		var obj map[string]any
		if err := json.Unmarshal([]byte(frame.data), &obj); err != nil {
			t.Fatalf("decode frame %q: %v", frame.data, err)
		}
		switch obj["type"] {
		case "content_block_delta":
			delta, _ := obj["delta"].(map[string]any)
			if delta["type"] == "text_delta" {
				text += delta["text"].(string)
			}
		case "message_stop":
			fak, _ := obj["fak"].(map[string]any)
			arm, _ := fak["native_arm"].(map[string]any)
			stopHasArm = arm["arm"] == "fak" && arm["final_answer"] == "Booked direct."
		}
	}
	if text != "Booked direct." {
		t.Fatalf("streamed text = %q, want final answer deltas", text)
	}
	if !stopHasArm {
		t.Fatalf("message_stop did not carry fak.native_arm with final answer; frames=%+v", frames)
	}
	if got := atomic.LoadInt32(&decideCount); got < 2 {
		t.Fatalf("DecideSession fired %d times, want at least 2 owned-loop turn boundaries", got)
	}
}
