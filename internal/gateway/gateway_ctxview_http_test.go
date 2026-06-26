package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// gateway_ctxview_http_test.go — the END-TO-END live-loop witness for the ctxplan context
// PLANNER (issue #555). The sibling gateway_ctxview_test.go exercises the planner by calling
// s.maybePlanMessages directly; THIS file drives a real HTTP request through srv.Handler() to
// a mock upstream and asserts on the bytes that actually reach the wire — closing the gap
// between "the hook is correct" and "the hook is on the live request path."
//
// Three properties, each read off the forwarded upstream body, not an internal call:
//   - ON, non-passthrough: with --ctx-view-budget > 0 the upstream's re-marshaled messages[]
//     is the PLANNED view — strictly fewer messages than the full transcript, bounded under the
//     budget. The planner reaches the wire.
//   - OFF (the default, budget 0): the upstream sees the FULL transcript, message-for-message.
//     A deploy that leaves the flag at 0 is unchanged on the live path.
//   - The Anthropic PASSTHROUGH boundary: with the budget OFF (the default) the flagship
//     `fak guard -- claude` route forwards req.Raw byte-for-byte — the planner is inert
//     until an operator opts in. With the budget ON the planned view NOW reaches this wire
//     too (#927 — the deferred #555 req.Raw transform): the off-topic middle turn is stubbed
//     in place beyond the cached system prefix, the prefix bytes stay byte-identical, and the
//     elided span stays demand-pageable. This file pins BOTH fences as live-path witnesses.

// ctxviewBudgetSession is a multi-turn transcript whose full residency exceeds a tight token
// budget, so an enabled planner must elide at least one older turn. The last user turn's intents
// predict "auth"/"token"/"refund", so the off-topic weather turn is the miss the forecast sheds
// first.
func ctxviewBudgetSession() []agent.Message {
	return []agent.Message{
		{Role: agent.RoleSystem, Content: "You are a support agent. Use the tools to help the user."},
		{Role: agent.RoleUser, Content: "rotate the auth token and then check the refund policy"},
		{Role: agent.RoleAssistant, Content: "weather sunny 22C light wind from the west, unrelated chatter to pad the history well beyond the resident budget so a planned view must elide it"},
		{Role: agent.RoleUser, Content: "what is the auth token rotation and refund window"},
	}
}

// captureUpstreamMessages stands up a mock OpenAI-compatible upstream that records the messages[]
// array of the LAST request body it received, and answers with a trivial no-tool completion. The
// recorded slice is what the gateway actually forwarded — the planned view when ctxview is on.
func captureUpstreamMessages(t *testing.T, got *[]capturedMessage) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read upstream body: %v", err)
		}
		var req struct {
			Messages []capturedMessage `json:"messages"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("decode upstream request: %v\n%s", err, raw)
		}
		*got = req.Messages
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":1,"total_tokens":12}}`))
	}))
}

type capturedMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// postCtxviewChat sends an OpenAI /v1/chat/completions request with the given messages through
// the gateway handler and fails on any non-200. It is the ctxview test's own minimal poster
// (the package's other postChat helper takes a trace + a ChatRequest; this one only needs a
// message list and the default trace).
func postCtxviewChat(t *testing.T, gatewayURL string, messages []agent.Message) {
	t.Helper()
	body, err := json.Marshal(ChatRequest{Model: "test-model", Messages: messages})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(gatewayURL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, b)
	}
}

// TestCtxViewHTTPOffForwardsFullHistory is the OFF live-path guard: with CtxViewBudget == 0 (the
// default) the upstream receives the FULL transcript over the real HTTP path — message-for-message
// — so a deploy that does not opt in sees the unplanned history on the wire, unchanged.
func TestCtxViewHTTPOffForwardsFullHistory(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var upstreamMsgs []capturedMessage
	upstream := captureUpstreamMessages(t, &upstreamMsgs)
	defer upstream.Close()

	srv, err := New(Config{
		EngineID: "test", Model: "test-model", BaseURL: upstream.URL, Provider: "openai",
		CtxViewBudget: 0, // OFF — the default
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	session := ctxviewBudgetSession()
	postCtxviewChat(t, ts.URL, session)

	if len(upstreamMsgs) != len(session) {
		t.Fatalf("OFF: upstream got %d messages, want the full %d (no planning)", len(upstreamMsgs), len(session))
	}
	for i, m := range upstreamMsgs {
		if m.Content != session[i].Content {
			t.Errorf("OFF: message %d content rewritten: got %q want %q", i, m.Content, session[i].Content)
		}
	}
}

// sessionTokens is the bytes/4 token estimate the planner charges and the render realizes — the
// same proxy ctxplan.TokenCost uses — so a forwarded view's sessionTokens is what the budget bounds.
func sessionTokens[T any](items []T, content func(T) string) int {
	n := 0
	for _, it := range items {
		n += (len(content(it)) + 3) / 4
	}
	return n
}

// TestCtxViewHTTPOnPlansHistoryOnTheWire is the ON live-path witness: with a CtxViewBudget set the
// upstream's re-marshaled messages[] is the PLANNED view — strictly fewer messages than the full
// transcript and bounded by the budget (modulo the documented pin floor) — proving the planner
// reaches the actual wire, not just an internal hook. The off-topic span is the one the forecast
// sheds, but every shed span stays demand-pageable in the lossless store (the recall half is
// witnessed at the seam in internal/agent), so this asserts the bounded-residency property the
// live loop must show.
//
// The budget is set just above the mandatory pin floor (system + first user + last user turns, which
// are FORCED resident and charged first — a pin is never elided, since that would drop the session's
// anchor/goal). A budget BELOW that floor is the "documented pin-overrun" case CLAIMS.md:133 names;
// here we bound the view by max(budget, pin-floor) so the assertion is honest about which term binds.
func TestCtxViewHTTPOnPlansHistoryOnTheWire(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var upstreamMsgs []capturedMessage
	upstream := captureUpstreamMessages(t, &upstreamMsgs)
	defer upstream.Close()

	session := ctxviewBudgetSession()
	fullTokens := sessionTokens(session, func(m agent.Message) string { return m.Content })
	// The pin floor: system (msg 0) + first user (msg 1) + last user (msg 3) are forced resident.
	// The off-topic assistant span (msg 2) is the only non-pin, so it is the one a plan can shed.
	pinFloor := sessionTokens([]agent.Message{session[0], session[1], session[3]}, func(m agent.Message) string { return m.Content })
	// Budget at the pin floor: tight enough that the off-topic span MUST be elided (full > budget),
	// loose enough that the bound binds at the budget, not the pin overrun.
	budget := pinFloor
	if budget >= fullTokens {
		t.Fatalf("test fixture: pin floor %d must be < full %d so a plan genuinely elides", pinFloor, fullTokens)
	}

	srv, err := New(Config{
		EngineID: "test", Model: "test-model", BaseURL: upstream.URL, Provider: "openai",
		CtxViewBudget: budget,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	postCtxviewChat(t, ts.URL, session)

	// The planner reached the wire: the upstream saw FEWER messages than the full transcript.
	if len(upstreamMsgs) == 0 {
		t.Fatal("ON: upstream got an empty history — the planner must never empty a turn")
	}
	if len(upstreamMsgs) >= len(session) {
		t.Fatalf("ON: upstream got %d messages, want strictly fewer than the full %d (the planned view must elide)", len(upstreamMsgs), len(session))
	}
	// BOUNDED: the forwarded view is at or under the budget (which equals the pin floor here), and
	// strictly below the full transcript's residency — the O(1) resident property on the live wire.
	tokens := sessionTokens(upstreamMsgs, func(m capturedMessage) string { return m.Content })
	for _, m := range upstreamMsgs {
		if m.Content == "" {
			t.Errorf("ON: a forwarded resident span must carry its bytes, got empty content role=%q", m.Role)
		}
	}
	if tokens > budget {
		t.Errorf("ON: forwarded view %d tokens must be <= budget %d (= pin floor)", tokens, budget)
	}
	if tokens >= fullTokens {
		t.Errorf("ON: forwarded view %d tokens must be strictly below the full transcript %d (it must have elided)", tokens, fullTokens)
	}
	// The off-topic weather span (the forecast miss) is the one shed: the planned view that reached
	// the wire must not still carry it verbatim.
	for _, m := range upstreamMsgs {
		if strings.Contains(m.Content, "weather sunny 22C") {
			t.Error("ON: the off-topic span should have been elided from the planned view that reached the wire")
		}
	}
}

// TestCtxViewHTTPAnthropicPassthroughOffForwardsRaw is the OFF live-path guard for the
// flagship wire: with CtxViewBudget == 0 (the default) the Anthropic passthrough forwards
// req.Raw byte-for-byte, so the planner is INERT and the upstream sees the caller's ORIGINAL
// body, unmodified. A deploy that does not opt in sees no change on the flagship wire.
func TestCtxViewHTTPAnthropicPassthroughOffForwardsRaw(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	// A real Claude-Code-shaped body with a cache_control prefix, which the passthrough must forward
	// verbatim so the upstream cache hit survives.
	inbound := []byte(`{"model":"claude-test","max_tokens":4096,` +
		`"system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[` +
		`{"role":"user","content":"rotate the auth token and check the refund policy"},` +
		`{"role":"assistant","content":"weather sunny 22C light wind from the west, unrelated padding to exceed any small resident budget"},` +
		`{"role":"user","content":"what is the auth token rotation and refund window"}]}`)

	var upstreamBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	// Provider anthropic + an anthropic upstream == the passthrough route. Budget 0 == OFF.
	srv, err := New(Config{
		EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic",
		APIKey: "configured-key", CtxViewBudget: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "caller-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, b)
	}

	// OFF: the passthrough forwarded the ORIGINAL bytes, byte-for-byte.
	if !bytes.Equal(upstreamBody, inbound) {
		t.Errorf("OFF: passthrough must forward req.Raw byte-for-byte when CtxViewBudget==0:\n got %q\nwant %q", upstreamBody, inbound)
	}
}

// TestCtxViewHTTPAnthropicPassthroughPlansView is the #927 live-path witness: with
// CtxViewBudget set, the ctxplan planned view NOW reaches the Anthropic passthrough — the
// off-topic assistant turn beyond the cached system prefix is stubbed out of the FORWARDED
// body (the bounded view), while (a) the cached prefix bytes stay byte-identical so the
// upstream cache hit survives, and (b) the elided span's bytes stay recoverable under a
// permissive re-plan (the lossless store the planner views preserved them — exact recall).
func TestCtxViewHTTPAnthropicPassthroughPlansView(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	inbound := []byte(`{"model":"claude-test","max_tokens":4096,` +
		`"system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[` +
		`{"role":"user","content":"rotate the auth token and check the refund policy"},` +
		`{"role":"assistant","content":"weather sunny 22C light wind from the west, unrelated padding to exceed any small resident budget"},` +
		`{"role":"user","content":"what is the auth token rotation and refund window"}]}`)

	var upstreamBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	// Provider anthropic == the passthrough route. A tight budget that forces the off-topic
	// assistant turn (the forecast MISS) to be elided beyond the cached system prefix.
	srv, err := New(Config{
		EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic",
		APIKey: "configured-key", CtxViewBudget: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "caller-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, b)
	}

	forwarded := string(upstreamBody)

	// (a) BOUNDED VIEW: the off-topic weather span was elided from the forwarded body —
	// replaced by a same-role [fak] ctxview-elided stub, not forwarded verbatim. The two
	// resident turns (the auth/refund user turns) are still present, byte-for-byte.
	if strings.Contains(forwarded, "weather sunny 22C") {
		t.Errorf("(a): the off-topic span must be elided from the forwarded passthrough body, still present:\n%s", forwarded)
	}
	if !strings.Contains(forwarded, "[fak] ctxview-elided") {
		t.Errorf("(a): the elided turn must be replaced by a [fak] ctxview-elided stub:\n%s", forwarded)
	}
	if !strings.Contains(forwarded, "rotate the auth token") || !strings.Contains(forwarded, "what is the auth token rotation") {
		t.Errorf("(a): the resident turns must still be present verbatim in the forwarded body:\n%s", forwarded)
	}

	// (b) PREFIX BYTE-IDENTITY: the cached system prefix bytes are unchanged so the upstream
	// cache hit (cache_read_input_tokens) survives. The system field is outside messages[],
	// so it rides through untouched; assert it byte-for-byte.
	wantSys := `"system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}]`
	if !strings.Contains(forwarded, wantSys) {
		t.Errorf("(b): the cached system prefix must be byte-identical in the forwarded body:\nwant %s\ngot  %s", wantSys, forwarded)
	}
	// The message COUNT and role alternation are preserved (same-role stubs), so Anthropic
	// accepts the body — verify three messages survive (one stubbed, two resident).
	if c := strings.Count(forwarded, `"role":`); c != 3 {
		t.Errorf("(b): the forwarded body must keep all 3 messages (one stubbed in place), got %d role keys:\n%s", c, forwarded)
	}

	// (c) EXACT RECALL: the elided span is not lost. Re-planning the SAME transcript under a
	// permissive budget selects the weather span resident again — the lossless store the
	// planner views preserved its verbatim bytes (the agent-level seam witnesses the demand-
	// page mechanism; here we witness the OUTCOME: nothing was destroyed).
	permissive := &agent.CtxViewPlanner{Enabled: true, Budget: 1 << 20}
	decoded, err := agent.DecodeAnthropicMessagesRequest(inbound)
	if err != nil {
		t.Fatalf("decode inbound: %v", err)
	}
	recovered, err := permissive.RenderTurn(context.Background(), decoded.Messages)
	if err != nil {
		t.Fatalf("permissive re-plan: %v", err)
	}
	foundWeather := false
	for _, m := range recovered {
		if strings.Contains(m.Content, "weather sunny 22C") {
			foundWeather = true
		}
	}
	if !foundWeather {
		t.Error("(c): the elided weather span must be recoverable under a permissive re-plan (exact recall — the store preserved it)")
	}
}
