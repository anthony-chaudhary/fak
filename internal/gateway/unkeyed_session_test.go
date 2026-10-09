package gateway

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// unkeyedPerConnectionTrace is the reserved DefaultTraceID that asks the gateway
// for one session per header-less connection (fak-private#930). Spelled as a
// literal so this test also compiles against the pre-fix gateway.
const unkeyedPerConnectionTrace = "unkeyed:per-connection"

func headerlessRequest(remoteAddr string) *http.Request {
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
	r.RemoteAddr = remoteAddr
	return r
}

func TestUnkeyedPerConnectionTraceSplitsHeaderlessCallers(t *testing.T) {
	srv := &Server{defaultTraceID: unkeyedPerConnectionTrace}

	recA := httptest.NewRecorder()
	a := srv.useHTTPTrace(recA, headerlessRequest("10.0.0.5:40001"), "")
	a2 := srv.useHTTPTrace(httptest.NewRecorder(), headerlessRequest("10.0.0.5:40001"), "")
	b := srv.useHTTPTrace(httptest.NewRecorder(), headerlessRequest("10.0.0.6:40002"), "")

	if a == unkeyedPerConnectionTrace || b == unkeyedPerConnectionTrace {
		t.Fatalf("header-less callers collapsed onto the shared default trace: a=%q b=%q", a, b)
	}
	if !strings.HasPrefix(a, "unkeyed-") || !strings.HasPrefix(b, "unkeyed-") {
		t.Fatalf("per-connection traces = %q, %q; want unkeyed-* identities", a, b)
	}
	if a == b {
		t.Fatalf("two connections share trace %q; want distinct identities", a)
	}
	if a != a2 {
		t.Fatalf("one connection got traces %q then %q; want a stable identity", a, a2)
	}
	if got := recA.Header().Get(traceHeader); got != a {
		t.Fatalf("response %s = %q, want the derived identity %q so the client can address it", traceHeader, got, a)
	}

	keyed := headerlessRequest("10.0.0.5:40001")
	keyed.Header.Set(traceHeader, "agent-7")
	if got := srv.useHTTPTrace(httptest.NewRecorder(), keyed, ""); got != "agent-7" {
		t.Fatalf("explicit %s = %q, want the caller's agent-7", traceHeader, got)
	}
	if got := srv.traceFor(""); got == unkeyedPerConnectionTrace {
		t.Fatalf("traceFor(\"\") = %q; the reserved value must never become a shared session", got)
	}
}

// The witnessed defect: one header-less agent exhausting its context budget 409'd
// every other header-less client on the endpoint. With per-connection identity the
// exhaustion stays on the connection that spent it.
func TestUnkeyedBudgetExhaustionDoesNotPoisonOtherConnections(t *testing.T) {
	const capTokens = 1000
	tbl := session.NewTable()
	seeded := map[string]bool{}
	srv := &Server{
		defaultTraceID: unkeyedPerConnectionTrace,
		decideSession: func(_ context.Context, trace string) SessionVerdict {
			if !seeded[trace] {
				seeded[trace] = true
				tbl.SetBudget(trace, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: capTokens})
			}
			return toGatewaySessionVerdict(tbl.Decide(trace))
		},
		debitSession: func(_ context.Context, trace string, u SessionUsage) SessionState {
			return toGatewaySessionState(tbl.DebitUsage(trace, session.Usage{OutputTokens: u.CompletionTokens, ContextTokens: u.ContextTokens}))
		},
	}
	ctx := context.Background()
	admit := func(remote string) (servedSessionTurn, bool) {
		trace := srv.useHTTPTrace(httptest.NewRecorder(), headerlessRequest(remote), "")
		turn, ok, canceled := srv.beginServedSessionTurn(ctx, trace)
		if canceled {
			t.Fatalf("turn for %s canceled", remote)
		}
		turn.traceID = trace
		return turn, ok
	}

	turn, ok := admit("10.0.0.5:40001")
	if !ok {
		t.Fatalf("first header-less turn refused: %+v", turn.state)
	}
	srv.debitServedSessionTurn(ctx, turn, agent.Usage{PromptTokens: capTokens + 500}, 0, nil)

	if again, ok := admit("10.0.0.5:40001"); ok {
		t.Fatalf("exhausted connection still admitted (trace %q); budget admission must stay fail-closed", again.traceID)
	} else if again.state.Reason != sessionReasonBudgetContext {
		t.Fatalf("exhausted connection refusal reason = %q, want %s", again.state.Reason, sessionReasonBudgetContext)
	}

	other, ok := admit("10.0.0.6:40002")
	if !ok {
		t.Fatalf("a distinct header-less connection was refused (trace %q reason %q): one agent's budget poisoned the endpoint", other.traceID, other.state.Reason)
	}
}

// End to end through Handler(): the trace middleware runs before the served
// wire, so the per-connection identity must hold on the real request path.
func TestUnkeyedHandlerIsolatesBudgetPerConnection(t *testing.T) {
	const capTokens = 1000
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: capTokens + 500, CompletionTokens: 1, TotalTokens: capTokens + 501},
	}}
	srv.SetDefaultTraceID(unkeyedPerConnectionTrace)
	tbl := session.NewTable()
	var mu sync.Mutex
	seeded := map[string]bool{}
	srv.decideSession = func(_ context.Context, trace string) SessionVerdict {
		mu.Lock()
		if !seeded[trace] {
			seeded[trace] = true
			tbl.SetBudget(trace, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: capTokens})
		}
		mu.Unlock()
		return toGatewaySessionVerdict(tbl.Decide(trace))
	}
	srv.debitSession = func(_ context.Context, trace string, u SessionUsage) SessionState {
		return toGatewaySessionState(tbl.DebitUsage(trace, session.Usage{OutputTokens: u.CompletionTokens, ContextTokens: u.ContextTokens}))
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	post := func(c *http.Client) (int, string, string) {
		t.Helper()
		resp, err := c.Post(ts.URL+"/v1/chat/completions", "application/json",
			strings.NewReader(`{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, resp.Header.Get(traceHeader), string(raw)
	}
	clientA := &http.Client{Transport: &http.Transport{}}
	clientB := &http.Client{Transport: &http.Transport{}}

	code, traceA, body := post(clientA)
	if code != http.StatusOK {
		t.Fatalf("first header-less request status = %d, want 200; body=%s", code, body)
	}
	if !strings.HasPrefix(traceA, "unkeyed-") {
		t.Fatalf("response %s = %q, want a per-connection unkeyed-* identity", traceHeader, traceA)
	}
	code, traceA2, body := post(clientA)
	if code != http.StatusConflict || traceA2 != traceA || !strings.Contains(body, sessionReasonBudgetContext) {
		t.Fatalf("exhausted connection: status=%d trace=%q (first %q) body=%s; want 409 %s on the same identity", code, traceA2, traceA, body, sessionReasonBudgetContext)
	}
	code, traceB, body := post(clientB)
	if code != http.StatusOK {
		t.Fatalf("a distinct header-less connection got %d (trace %q): one agent's budget poisoned the endpoint; body=%s", code, traceB, body)
	}
	if traceB == traceA {
		t.Fatalf("two connections share trace %q", traceA)
	}
}
