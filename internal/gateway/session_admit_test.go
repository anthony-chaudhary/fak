package gateway

// session_admit_test.go — the PROXY-path enforcement of session control: a paused /
// draining / stopped session's next /v1/chat/completions request is refused with 409
// (operator "cancel a request in flight"), while running / unknown / no-route-wired
// fail OPEN (the request proceeds, byte-for-byte the pre-control path).

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

type plannerFunc func(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error)

func (f plannerFunc) Complete(ctx context.Context, msgs []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return f(ctx, msgs, tools, opts...)
}

func (f plannerFunc) Model() string { return "test-model" }

func chatPostWithTrace(t *testing.T, url, trace string) *http.Response {
	t.Helper()
	body, _ := json.Marshal(map[string]any{
		"model":    "test-model",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	req, err := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if trace != "" {
		req.Header.Set("X-Trace-Id", trace)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestSessionAdmitRefusesHeldSessionsOnProxy(t *testing.T) {
	srv := newTestServer(t)
	// The operator-set DRIVE state for this trace; the test flips it per case.
	var run, reason string
	srv.observeSession = func(_ context.Context, trace string) SessionState {
		return SessionState{TraceID: trace, Run: run, Reason: reason}
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Non-advancing states refuse the next request with 409 + a session_<state> code.
	for _, st := range []string{"paused", "draining", "stopped"} {
		run, reason = st, "operator-"+st
		resp := chatPostWithTrace(t, ts.URL, "held-"+st)
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusConflict {
			t.Fatalf("run=%s: status = %d, want 409; body=%s", st, resp.StatusCode, raw)
		}
		if !strings.Contains(string(raw), "session_"+st) || !strings.Contains(string(raw), "operator-"+st) {
			t.Fatalf("run=%s: refusal body missing code/reason: %s", st, raw)
		}
	}
}

func TestSessionAdmitFailsOpenWhenRunningOrUnwired(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// (1) No observeSession wired ⇒ fail OPEN: the guard must not produce a 409.
	resp := chatPostWithTrace(t, ts.URL, "trace-unwired")
	resp.Body.Close()
	if resp.StatusCode == http.StatusConflict {
		t.Fatalf("unwired session route must not refuse; got 409")
	}

	// (2) An advancing state (running) is admitted — the guard does not 409 it.
	srv.observeSession = func(_ context.Context, trace string) SessionState {
		return SessionState{TraceID: trace, Run: "running"}
	}
	resp = chatPostWithTrace(t, ts.URL, "trace-running")
	resp.Body.Close()
	if resp.StatusCode == http.StatusConflict {
		t.Fatalf("running session must be admitted; got 409")
	}

	// (3) throttled is admitted too (pace shapes fak's own loop, not proxy admission).
	srv.observeSession = func(_ context.Context, trace string) SessionState {
		return SessionState{TraceID: trace, Run: "throttled"}
	}
	resp = chatPostWithTrace(t, ts.URL, "trace-throttled")
	resp.Body.Close()
	if resp.StatusCode == http.StatusConflict {
		t.Fatalf("throttled session must be admitted; got 409")
	}
}

func TestTraceForUsesConfiguredDefaultTrace(t *testing.T) {
	srv := &Server{defaultTraceID: "stable-session"}
	if got := srv.traceFor(""); got != "stable-session" {
		t.Fatalf("traceFor empty = %q, want configured default", got)
	}
	if got := srv.traceFor(" caller "); got != "caller" {
		t.Fatalf("traceFor explicit = %q, want trimmed caller trace", got)
	}
}

func TestSessionDecideDebitsTurnsOnServedChat(t *testing.T) {
	srv := newTestServer(t)
	tbl := session.NewTable()
	tbl.SetBudget("sess-turns", session.Budget{TurnsLeft: 1, TokensLeft: session.Unbounded})
	wireSessionTableForTest(srv, tbl)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := chatPostWithTrace(t, ts.URL, "sess-turns")
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first turn status = %d, want 200", resp.StatusCode)
	}
	if got := tbl.Get("sess-turns").Budget.TurnsLeft; got != 0 {
		t.Fatalf("TurnsLeft after first served turn = %d, want 0", got)
	}

	resp = chatPostWithTrace(t, ts.URL, "sess-turns")
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("second turn status = %d, want 409; body=%s", resp.StatusCode, raw)
	}
	if !strings.Contains(string(raw), session.ReasonBudgetTurns) {
		t.Fatalf("turn-budget refusal missing reason %s: %s", session.ReasonBudgetTurns, raw)
	}
}

func TestSessionDebitTokensAndPaceOnServedChat(t *testing.T) {
	srv := newTestServer(t)
	tbl := session.NewTable()
	tbl.SetBudget("sess-tokens", session.Budget{TurnsLeft: session.Unbounded, TokensLeft: 5})
	tbl.SetPace("sess-tokens", session.Pace{MaxTokensPerTurn: 3})
	wireSessionTableForTest(srv, tbl)
	rp := &recordingPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 1, CompletionTokens: 5, TotalTokens: 6},
	}}
	srv.planner = rp
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"model":"test-model","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`
	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", "sess-tokens")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first token-budget turn status = %d, want 200", resp.StatusCode)
	}
	if rp.got.MaxTokens == nil || *rp.got.MaxTokens != 3 {
		t.Fatalf("planner max_tokens = %v, want session pace cap 3", rp.got.MaxTokens)
	}
	if got := tbl.Get("sess-tokens").Budget.TokensLeft; got != 0 {
		t.Fatalf("TokensLeft after completion debit = %d, want 0", got)
	}

	resp = chatPostWithTrace(t, ts.URL, "sess-tokens")
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("second token-budget turn status = %d, want 409; body=%s", resp.StatusCode, raw)
	}
	if !strings.Contains(string(raw), session.ReasonBudgetTokens) {
		t.Fatalf("token-budget refusal missing reason %s: %s", session.ReasonBudgetTokens, raw)
	}
}

func TestSessionContextBudgetRefusalCarriesResetDirective(t *testing.T) {
	srv := newTestServer(t)
	tbl := session.NewTable()
	tbl.SetBudget("sess-context", session.Budget{
		TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 10,
	})
	wireSessionTableForTest(srv, tbl)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage: agent.Usage{
			PromptTokens:             8,
			CompletionTokens:         1,
			CacheReadInputTokens:     3,
			CacheCreationInputTokens: 2,
			TotalTokens:              14,
		},
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := chatPostWithTrace(t, ts.URL, "sess-context")
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first context-budget turn status = %d, want 200", resp.StatusCode)
	}
	st := tbl.Get("sess-context")
	if st.Run != session.Draining || st.Reason != session.ReasonBudgetContext || st.ContinuationID == "" {
		t.Fatalf("post-debit state = %+v, want draining context exhaustion with continuation", st)
	}

	resp = chatPostWithTrace(t, ts.URL, "sess-context")
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("second context-budget turn status = %d, want 409; body=%s", resp.StatusCode, raw)
	}
	body := string(raw)
	for _, want := range []string{
		session.ReasonBudgetContext,
		"restart_fresh_session",
		"continuation_id",
		st.ContinuationID,
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("context-budget refusal missing %q: %s", want, body)
		}
	}
}

// TestResetOnBudgetContinuesTransparently proves the opt-in human-like reset: once a
// session's context budget drains (the case that 409s in the test above), a wired
// ResetOnBudget hook re-arms a fresh session and the client's NEXT request succeeds
// (200) on the new trace with the carryover seed spliced, instead of being refused.
func TestResetOnBudgetContinuesTransparently(t *testing.T) {
	srv := newTestServer(t)
	tbl := session.NewTable()
	tbl.SetBudget("sess-reset", session.Budget{
		TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 8,
	})
	wireSessionTableForTest(srv, tbl)

	// The host's reset action: re-arm a fresh session under the continuation id and
	// hand back a one-line carryover seed. Records that it fired and what it saw.
	var fired bool
	var sawMessages int
	srv.resetOnBudget = func(_ context.Context, trace string, msgs []agent.Message) (string, []agent.Message, bool) {
		fired = true
		sawMessages = len(msgs)
		st := tbl.Get(trace)
		child := st.ContinuationID
		tbl.Recontinue(trace, child, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 10})
		return child, []agent.Message{{Role: agent.RoleSystem, Content: "[carryover] durable: user prefers concise answers"}}, true
	}

	cp := &seedCapturePlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 8, CompletionTokens: 1, TotalTokens: 9},
	}}
	srv.planner = cp
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Turn 1 drains the context budget to Draining/Stopped with a continuation id.
	resp := chatPostWithTrace(t, ts.URL, "sess-reset")
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first turn status = %d, want 200", resp.StatusCode)
	}
	if st := tbl.Get("sess-reset"); st.ContinuationID == "" {
		t.Fatalf("expected a continuation id after context drain, got %+v", st)
	}

	// Turn 2 would 409 — but with ResetOnBudget wired it transparently continues (200).
	resp = chatPostWithTrace(t, ts.URL, "sess-reset")
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("reset turn status = %d, want 200 (transparent continue); body=%s", resp.StatusCode, raw)
	}
	if !fired {
		t.Fatal("ResetOnBudget hook never fired on the budget-drained boundary")
	}
	if sawMessages == 0 {
		t.Fatal("ResetOnBudget hook got no transcript to distill")
	}
	// The seed was spliced ahead of the live messages the planner saw (1 user + 1 seed).
	if cp.seen < 2 {
		t.Fatalf("planner saw %d messages, want >=2 (seed spliced ahead of live turn)", cp.seen)
	}
}

// seedCapturePlanner records how many messages reached the planner, so a test can
// assert the carryover seed was spliced ahead of the live turn.
type seedCapturePlanner struct {
	comp *agent.Completion
	seen int
}

func (p *seedCapturePlanner) Complete(_ context.Context, m []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.seen = len(m)
	return p.comp, nil
}
func (*seedCapturePlanner) Model() string { return "seed-capture" }

// TestResetOnBudgetUnwiredStillRefuses is the no-regression guard: with NO reset hook
// wired, a budget drain still produces the historical 409 + directive — the reset is
// strictly opt-in.
func TestResetOnBudgetUnwiredStillRefuses(t *testing.T) {
	srv := newTestServer(t)
	tbl := session.NewTable()
	tbl.SetBudget("sess-noreset", session.Budget{
		TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 8,
	})
	wireSessionTableForTest(srv, tbl) // note: resetOnBudget intentionally NOT wired
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 8, CompletionTokens: 1, TotalTokens: 9},
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := chatPostWithTrace(t, ts.URL, "sess-noreset")
	resp.Body.Close()
	resp = chatPostWithTrace(t, ts.URL, "sess-noreset")
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("unwired reset must still 409 on budget drain; got %d: %s", resp.StatusCode, raw)
	}
}

func wireSessionTableForTest(srv *Server, tbl *session.Table) {
	srv.decideSession = func(_ context.Context, trace string) SessionVerdict {
		return testSessionVerdict(tbl.Decide(trace))
	}
	srv.debitSession = func(_ context.Context, trace string, usage SessionUsage) SessionState {
		return testSessionState(tbl.DebitUsage(trace, session.Usage{
			OutputTokens:  usage.CompletionTokens,
			ContextTokens: usage.ContextTokens,
		}))
	}
}

func testSessionVerdict(v session.Verdict) SessionVerdict {
	return SessionVerdict{
		Proceed:   v.Proceed,
		MaxTokens: v.MaxTokens,
		MinGapMs:  v.MinGapMs,
		State:     testSessionState(v.State),
		Stop:      v.Stop,
		Reason:    v.Reason,
	}
}

func testSessionState(st session.State) SessionState {
	return SessionState{
		TraceID:        st.TraceID,
		Run:            st.Run.String(),
		Budget:         SessionBudget{TurnsLeft: st.Budget.TurnsLeft, TokensLeft: st.Budget.TokensLeft, ContextTokensLeft: st.Budget.ContextTokensLeft},
		Priority:       st.Priority,
		Pace:           SessionPace{MaxTokensPerTurn: st.Pace.MaxTokensPerTurn, MinTurnGapMs: st.Pace.MinTurnGapMs},
		Reason:         st.Reason,
		ContinuationID: st.ContinuationID,
		ParentTrace:    st.ParentTrace,
		Generation:     st.Generation,
		Rev:            st.Rev,
	}
}
