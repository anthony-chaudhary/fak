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
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

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
