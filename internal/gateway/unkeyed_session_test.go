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

// Ref #13777: unchanged conversation accounting survives reconnects; distinct
// conversations sharing one connection do not share their exhausted budget.
// fak-test:runtime fast est=100ms lane=default
func TestUnkeyedHandlerConversationContinuity(t *testing.T) {
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
	var peers []string
	handler := srv.Handler()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		peers = append(peers, r.RemoteAddr)
		mu.Unlock()
		handler.ServeHTTP(w, r)
	}))
	defer ts.Close()

	post := func(c *http.Client, body string) (int, string, string) {
		t.Helper()
		resp, err := c.Post(ts.URL+"/v1/chat/completions", "application/json",
			strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, resp.Header.Get(traceHeader), string(raw)
	}
	clientA := &http.Client{Transport: &http.Transport{}}
	clientB := &http.Client{Transport: &http.Transport{}}
	defer clientA.CloseIdleConnections()
	defer clientB.CloseIdleConnections()
	const rootA = `{"model":"test-model","messages":[{"role":"user","content":"conversation A"}]}`
	const rootB = `{"model":"test-model","messages":[{"role":"user","content":"conversation B"}]}`

	code, traceA, body := post(clientA, rootA)
	if code != http.StatusOK {
		t.Fatalf("first header-less request status = %d, want 200; body=%s", code, body)
	}
	if !strings.HasPrefix(traceA, "unkeyed-") {
		t.Fatalf("response %s = %q, want an unkeyed-* accounting identity", traceHeader, traceA)
	}
	code, traceA2, body := post(clientA, rootA)
	if code != http.StatusConflict || traceA2 != traceA || !strings.Contains(body, sessionReasonBudgetContext) {
		t.Fatalf("exhausted connection: status=%d trace=%q (first %q) body=%s; want 409 %s on the same identity", code, traceA2, traceA, body, sessionReasonBudgetContext)
	}
	code, traceA3, body := post(clientB, rootA)
	if code != http.StatusConflict || traceA3 != traceA || !strings.Contains(body, sessionReasonBudgetContext) {
		t.Fatalf("reconnected conversation: status=%d trace=%q first=%q body=%s; want same exhausted budget", code, traceA3, traceA, body)
	}
	code, traceB, body := post(clientA, rootB)
	if code != http.StatusOK {
		t.Fatalf("a distinct conversation on the original connection got %d (trace %q); body=%s", code, traceB, body)
	}
	if traceB == traceA {
		t.Fatalf("two distinct conversation roots share trace %q", traceA)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(peers) != 4 || peers[0] == peers[2] || peers[0] != peers[3] {
		t.Fatalf("fixture did not exercise two connections and original connection reuse: %v", peers)
	}
}

func unkeyedContinuityServer(t *testing.T, defaultTrace string, limit int) *Server {
	t.Helper()
	srv := newTestServer(t)
	srv.SetDefaultTraceID(defaultTrace)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop",
		Usage: agent.Usage{PromptTokens: 1500, CompletionTokens: 1, TotalTokens: 1501},
	}}
	tbl := session.NewTableWithLimit(limit)
	srv.decideSession = func(_ context.Context, trace string) SessionVerdict {
		tbl.SeedBudget(trace, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 1000})
		return toGatewaySessionVerdict(tbl.Decide(trace))
	}
	srv.debitSession = func(_ context.Context, trace string, u SessionUsage) SessionState {
		return toGatewaySessionState(tbl.DebitUsage(trace, session.Usage{OutputTokens: u.CompletionTokens, ContextTokens: u.ContextTokens}))
	}
	return srv
}

func unkeyedContinuityPost(t *testing.T, handler http.Handler, body, remote string, headers map[string]string) (int, string) {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	r.RemoteAddr = remote
	r.Header.Set("Content-Type", "application/json")
	for name, value := range headers {
		r.Header.Set(name, value)
	}
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, r)
	if w.Code != http.StatusOK && (w.Code != http.StatusConflict || !strings.Contains(w.Body.String(), sessionReasonBudgetContext)) {
		t.Fatalf("unexpected handler outcome %d: %s", w.Code, w.Body.String())
	}
	return w.Code, w.Header().Get(traceHeader)
}

// All requests cross Handler(), including real keyset authentication. Inputs use
// existing APIs so the same test can be overlaid on the unmodified parent.
// fak-test:runtime medium est=1s lane=default
func TestUnkeyedHandlerConversationBoundaries(t *testing.T) {
	const root = `{"model":"test-model","messages":[{"role":"user","content":"root"}]}`
	const changed = `{"model":"test-model","messages":[{"role":"user","content":"changed root"}]}`
	const continuation = `{"model":"test-model","messages":[{"role":"user","content":"root"},{"role":"assistant","content":"ok"},{"role":"user","content":"next"}]}`
	for _, tc := range []struct {
		name, defaultTrace, secondBody, exactTrace string
		firstHeaders, secondHeaders                map[string]string
		wantSecond                                 int
	}{
		{name: "unchanged root continuation", defaultTrace: unkeyedPerConnectionTrace, secondBody: continuation, wantSecond: http.StatusConflict},
		{name: "changed root resets", defaultTrace: unkeyedPerConnectionTrace, secondBody: changed, wantSecond: http.StatusOK},
		{name: "changed model resets", defaultTrace: unkeyedPerConnectionTrace, secondBody: strings.Replace(root, "test-model", "other-model", 1), wantSecond: http.StatusOK},
		{name: "changed tools reset", defaultTrace: unkeyedPerConnectionTrace, secondBody: `{"model":"test-model","tools":[{"type":"function","function":{"name":"example","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"root"}]}`, wantSecond: http.StatusOK},
		{name: "default disabled", secondBody: root, wantSecond: http.StatusOK},
		{name: "configured default preserved", defaultTrace: "operator-session", secondBody: changed, exactTrace: "operator-session", wantSecond: http.StatusConflict},
		{name: "explicit trace cannot spoof generated marker", defaultTrace: unkeyedPerConnectionTrace, secondBody: changed, exactTrace: "unkeyed-caller-chosen", firstHeaders: map[string]string{"X-Trace-Id": "unkeyed-caller-chosen", "X-Fak-Unkeyed-Generated": "true"}, secondHeaders: map[string]string{"X-Trace-Id": "unkeyed-caller-chosen", "X-Fak-Unkeyed-Generated": "true"}, wantSecond: http.StatusConflict},
		{name: "request id preserved", defaultTrace: unkeyedPerConnectionTrace, secondBody: changed, exactTrace: "caller-request", firstHeaders: map[string]string{"X-Request-ID": "caller-request"}, secondHeaders: map[string]string{"X-Request-ID": "caller-request"}, wantSecond: http.StatusConflict},
		{name: "traceparent preserved", defaultTrace: unkeyedPerConnectionTrace, secondBody: changed, exactTrace: "0123456789abcdef0123456789abcdef", firstHeaders: map[string]string{"traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"}, secondHeaders: map[string]string{"traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"}, wantSecond: http.StatusConflict},
		{name: "anonymous header changes do not create identity", defaultTrace: unkeyedPerConnectionTrace, secondBody: root, firstHeaders: map[string]string{"X-Fak-Principal": "claimed-a", "X-Forwarded-For": "192.0.2.8", "User-Agent": "one"}, secondHeaders: map[string]string{"X-Fak-Principal": "claimed-b", "X-Forwarded-For": "192.0.2.9", "User-Agent": "two"}, wantSecond: http.StatusConflict},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handler := unkeyedContinuityServer(t, tc.defaultTrace, 0).Handler()
			firstCode, firstTrace := unkeyedContinuityPost(t, handler, root, "192.0.2.1:40001", tc.firstHeaders)
			secondCode, secondTrace := unkeyedContinuityPost(t, handler, tc.secondBody, "192.0.2.1:40002", tc.secondHeaders)
			if firstCode != http.StatusOK || secondCode != tc.wantSecond || (firstTrace == secondTrace) != (tc.wantSecond == http.StatusConflict) {
				t.Fatalf("statuses=%d/%d traces=%q/%q; want second=%d with matching accounting iff exhausted", firstCode, secondCode, firstTrace, secondTrace, tc.wantSecond)
			}
			if tc.exactTrace != "" && (firstTrace != tc.exactTrace || secondTrace != tc.exactTrace) {
				t.Fatalf("explicit/default trace was rewritten: %q/%q want %q", firstTrace, secondTrace, tc.exactTrace)
			}
		})
	}
	t.Run("authenticated principals remain isolated", func(t *testing.T) {
		srv := unkeyedContinuityServer(t, unkeyedPerConnectionTrace, 0)
		srv.keyset = newKeyset(map[string]string{"fixture-key-a": "tenant-a", "fixture-key-b": "tenant-b"})
		handler := srv.Handler()
		codeA, traceA := unkeyedContinuityPost(t, handler, root, "192.0.2.1:40001", map[string]string{"Authorization": "Bearer fixture-key-a", "X-Fak-Principal": "tenant-b"})
		codeB, traceB := unkeyedContinuityPost(t, handler, root, "192.0.2.1:40001", map[string]string{"Authorization": "Bearer fixture-key-b", "X-Fak-Principal": "tenant-a"})
		codeAgain, traceAgain := unkeyedContinuityPost(t, handler, root, "192.0.2.1:40002", map[string]string{"Authorization": "Bearer fixture-key-a", "X-Fak-Principal": "claimed-new"})
		if codeA != http.StatusOK || codeB != http.StatusOK || traceA == traceB || codeAgain != http.StatusConflict || traceAgain != traceA {
			t.Fatalf("authenticated accounting lost isolation or continuity: %d/%d/%d traces=%q/%q/%q", codeA, codeB, codeAgain, traceA, traceB, traceAgain)
		}
	})
	t.Run("no user keeps connection fallback", func(t *testing.T) {
		handler := unkeyedContinuityServer(t, unkeyedPerConnectionTrace, 0).Handler()
		const noUser = `{"model":"test-model","messages":[{"role":"system","content":"setup only"}]}`
		firstCode, firstTrace := unkeyedContinuityPost(t, handler, noUser, "192.0.2.1:40001", nil)
		secondCode, secondTrace := unkeyedContinuityPost(t, handler, noUser, "192.0.2.1:40002", nil)
		if firstCode != http.StatusOK || secondCode != http.StatusOK || firstTrace == secondTrace {
			t.Fatalf("no-user fallback changed: statuses=%d/%d traces=%q/%q", firstCode, secondCode, firstTrace, secondTrace)
		}
	})
	t.Run("eviction remains a reset boundary", func(t *testing.T) {
		handler := unkeyedContinuityServer(t, unkeyedPerConnectionTrace, 1).Handler()
		_, firstTrace := unkeyedContinuityPost(t, handler, root, "192.0.2.1:40001", nil)
		otherCode, _ := unkeyedContinuityPost(t, handler, changed, "192.0.2.1:40001", nil)
		retryCode, retryTrace := unkeyedContinuityPost(t, handler, root, "192.0.2.1:40002", nil)
		if otherCode != http.StatusOK || retryCode != http.StatusOK || retryTrace != firstTrace {
			t.Fatalf("bounded-table reset contract changed: other=%d retry=%d traces=%q/%q", otherCode, retryCode, firstTrace, retryTrace)
		}
	})
}
