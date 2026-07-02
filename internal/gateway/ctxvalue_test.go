package gateway

// ctxvalue_test.go — the managed-context value report (ctxvalue.go): the pure
// step-advice policy rung by rung, the rolling per-session accumulator (growth
// slope, context-event era reset), and the two query surfaces — GET
// /v1/fak/ctxvalue over the live wire and the fak_context_value MCP tool.

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestAdviseCtxStepRungs walks the closed decision ladder: every step class is
// reachable, each from the rung that owns it, and the fail-closed rungs answer
// unknown instead of guessing.
func TestAdviseCtxStepRungs(t *testing.T) {
	cases := []struct {
		name  string
		st    ctxValueState
		class StepClass
		basis string
	}{
		{"no turns", ctxValueState{}, StepClassUnknown, "none"},
		{"event last turn", ctxValueState{Turns: 10, ContextEvents: 1, TurnsSinceEvent: 0, LastTurnEvent: true, Budget: 10000, Resident: 100}, StepClassRebuild, "context_event"},
		{"one turn after event", ctxValueState{Turns: 11, ContextEvents: 1, TurnsSinceEvent: 1, Budget: 10000, Resident: 100}, StepClassRebuild, "context_event"},
		{"crowded window", ctxValueState{Turns: 5, TurnsSinceEvent: 5, Budget: 10000, Resident: 8500}, StepClassCheckpoint, "token_headroom"},
		{"forecast tightens below the percent rung", ctxValueState{Turns: 5, TurnsSinceEvent: 5, Budget: 10000, Resident: 3000, GrowthPerTurn: 2500}, StepClassCheckpoint, "token_headroom"},
		{"half-full window", ctxValueState{Turns: 5, TurnsSinceEvent: 5, Budget: 10000, Resident: 5200}, StepClassBounded, "token_headroom"},
		{"slow growth forecast bounds an uncrowded window", ctxValueState{Turns: 5, TurnsSinceEvent: 5, Budget: 10000, Resident: 3000, GrowthPerTurn: 700}, StepClassBounded, "token_headroom"},
		{"wide headroom", ctxValueState{Turns: 5, TurnsSinceEvent: 5, Budget: 10000, Resident: 2000, GrowthPerTurn: 100}, StepClassAny, "token_headroom"},
		{"cadence checkpoint", ctxValueState{Turns: 20, ContextEvents: 2, TurnsSinceEvent: 9}, StepClassCheckpoint, "event_cadence"},
		{"cadence bounded", ctxValueState{Turns: 20, ContextEvents: 2, TurnsSinceEvent: 6}, StepClassBounded, "event_cadence"},
		{"cadence any", ctxValueState{Turns: 20, ContextEvents: 2, TurnsSinceEvent: 3}, StepClassAny, "event_cadence"},
		{"no budget no events", ctxValueState{Turns: 5, TurnsSinceEvent: 5}, StepClassUnknown, "none"},
	}
	for _, tc := range cases {
		a := adviseCtxStep(tc.st)
		if a.StepClass != tc.class || a.Basis != tc.basis {
			t.Errorf("%s: adviseCtxStep = %s/%s, want %s/%s (reason: %s)", tc.name, a.StepClass, a.Basis, tc.class, tc.basis, a.Reason)
		}
		if a.Provenance != "DECISION" {
			t.Errorf("%s: advice provenance = %q, want DECISION", tc.name, a.Provenance)
		}
		if a.Reason == "" {
			t.Errorf("%s: advice carries no reason", tc.name)
		}
	}
}

// TestObserveCtxValueRollsLevels proves the accumulator's three levels: the turn
// counters (WITNESSED), the token ring's growth slope (OBSERVED axes), and the
// context-event era reset that keeps the slope from spanning a window rewrite.
func TestObserveCtxValueRollsLevels(t *testing.T) {
	s := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model", CompactHistoryBudget: 10000})

	// Three growing turns: resident 1000 -> 2000 -> 3000, output 10 each.
	for i := 1; i <= 3; i++ {
		s.observeCtxValue("t1", 1000*i-200, 100, 100, 10, false)
	}
	r := s.CtxValueReportFor("t1")
	if r.Turns.TurnsObserved != 3 || r.Turns.ContextEvents != 0 || r.Turns.TurnsSinceContextEvent != 3 {
		t.Fatalf("turns level = %+v, want 3 turns, 0 events, 3 since", r.Turns)
	}
	if r.Tokens.ResidentTokens != 3000 || r.Tokens.PeakResidentTokens != 3000 {
		t.Fatalf("tokens level = %+v, want resident/peak 3000", r.Tokens)
	}
	if r.Tokens.GrowthPerTurn != 1000 {
		t.Fatalf("growth = %v, want 1000 tokens/turn over the ring", r.Tokens.GrowthPerTurn)
	}
	if r.Tokens.Headroom == nil || r.Tokens.Headroom.Tokens != 7000 {
		t.Fatalf("headroom = %+v, want 7000 tokens", r.Tokens.Headroom)
	}
	// est = 7000/1000 = 7 turns -> FORECAST published, and the advice tightens to
	// checkpoint (est < 4 is false, est < 12 -> bounded... 7 < 12) => bounded.
	if r.Turns.EstTurnsToContextEvent != 7 || r.Turns.EstProvenance != "FORECAST" {
		t.Fatalf("est = %v/%q, want 7/FORECAST", r.Turns.EstTurnsToContextEvent, r.Turns.EstProvenance)
	}
	if r.StepAdvice.StepClass != StepClassBounded || r.StepAdvice.Basis != "token_headroom" {
		t.Fatalf("advice = %+v, want bounded/token_headroom", r.StepAdvice)
	}
	if r.Session.TotalOutputTokens != 30 || r.Session.TotalResidentTokenTurns != 6000 {
		t.Fatalf("session level = %+v, want 30 output, 6000 resident-token-turns", r.Session)
	}

	// A context event: the era resets, the advice flips to rebuild, the phase to
	// post_event, and the growth slope no longer spans the rewrite.
	s.observeCtxValue("t1", 500, 100, 100, 10, true)
	r = s.CtxValueReportFor("t1")
	if r.Turns.ContextEvents != 1 || r.Turns.TurnsSinceContextEvent != 0 {
		t.Fatalf("after event: turns level = %+v, want 1 event, 0 since", r.Turns)
	}
	if r.Tokens.GrowthPerTurn != 0 {
		t.Fatalf("after event: growth = %v, want 0 (fresh era, one sample)", r.Tokens.GrowthPerTurn)
	}
	if r.StepAdvice.StepClass != StepClassRebuild {
		t.Fatalf("after event: advice = %+v, want rebuild", r.StepAdvice)
	}
	if r.Session.Phase != "post_event" {
		t.Fatalf("after event: phase = %q, want post_event", r.Session.Phase)
	}
	// Peak survives the era reset (it is a session-arc fact, not a window fact).
	if r.Tokens.PeakResidentTokens != 3000 {
		t.Fatalf("after event: peak = %d, want 3000", r.Tokens.PeakResidentTokens)
	}
}

// TestCtxValueUnknownTraceIsDecidable proves the MCP single-session read never
// errors: an unseen trace gets a zero report whose advice says unknown and why.
func TestCtxValueUnknownTraceIsDecidable(t *testing.T) {
	s := newTestServer(t)
	r := s.CtxValueReportFor("never-served")
	if r.TraceID != "never-served" || r.Schema != ctxValueSchema {
		t.Fatalf("zero report header = %+v", r)
	}
	if r.StepAdvice.StepClass != StepClassUnknown || r.StepAdvice.Reason == "" {
		t.Fatalf("zero report advice = %+v, want unknown with a reason", r.StepAdvice)
	}
	if r.Turns.TurnsObserved != 0 {
		t.Fatalf("zero report claims %d turns", r.Turns.TurnsObserved)
	}
}

// TestCtxValueHTTPEndpoint proves the live-wire acceptance: a served chat turn
// makes GET /v1/fak/ctxvalue carry that session's multi-level report with the
// Law-A2 provenance labels intact, and ?trace= filters the snapshot.
func TestCtxValueHTTPEndpoint(t *testing.T) {
	s := newTestServer(t)
	s.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage: agent.Usage{
			PromptTokens:             100,
			CompletionTokens:         4,
			CacheReadInputTokens:     40000,
			CacheCreationInputTokens: 500,
		},
	}}

	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Before traffic: an empty snapshot, never a phantom session.
	var empty CtxValueSnapshot
	getJSON(t, ts.URL+"/v1/fak/ctxvalue", &empty)
	if len(empty.Sessions) != 0 {
		t.Fatalf("pre-traffic snapshot carries %d sessions, want 0", len(empty.Sessions))
	}

	var chat ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
		Model:    "test-model",
		Messages: []agent.Message{{Role: "user", Content: "hello"}},
	}, &chat)
	if code != http.StatusOK {
		t.Fatalf("chat status = %d, want 200", code)
	}

	var snap CtxValueSnapshot
	getJSON(t, ts.URL+"/v1/fak/ctxvalue", &snap)
	if snap.Schema != ctxValueSchema || len(snap.Sessions) != 1 {
		t.Fatalf("snapshot = %+v, want schema %s with exactly one session", snap, ctxValueSchema)
	}
	r := snap.Sessions[0]
	if r.Turns.TurnsObserved != 1 {
		t.Fatalf("turns_observed = %d, want 1", r.Turns.TurnsObserved)
	}
	// resident = uncached prompt + normalized cache read + creation = 100+40000+500.
	if r.Tokens.ResidentTokens != 40600 {
		t.Fatalf("resident_tokens = %d, want 40600", r.Tokens.ResidentTokens)
	}
	if r.Tokens.Provenance != "OBSERVED" || r.Turns.Provenance != "WITNESSED" || r.Session.Provenance != "WITNESSED" || r.StepAdvice.Provenance != "DECISION" {
		t.Fatalf("provenance labels lost on the wire: %+v", r)
	}
	if r.StepAdvice.StepClass == "" || r.Session.Phase == "" {
		t.Fatalf("report missing advice/phase: %+v", r)
	}

	// ?trace= filters: the served session's trace matches itself, a bogus one is empty.
	var one CtxValueSnapshot
	getJSON(t, ts.URL+"/v1/fak/ctxvalue?trace="+r.TraceID, &one)
	if len(one.Sessions) != 1 || one.Sessions[0].TraceID != r.TraceID {
		t.Fatalf("trace filter returned %+v, want the %s session", one, r.TraceID)
	}
	var none CtxValueSnapshot
	getJSON(t, ts.URL+"/v1/fak/ctxvalue?trace=no-such-trace", &none)
	if len(none.Sessions) != 0 {
		t.Fatalf("bogus trace filter returned %d sessions, want 0", len(none.Sessions))
	}

	// Method guard: POST is refused.
	resp, err := http.Post(ts.URL+"/v1/fak/ctxvalue", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("POST /v1/fak/ctxvalue = %d, want 405", resp.StatusCode)
	}
}

// TestCtxValueMCPTool proves the agent-facing seam: fak_context_value is listed,
// and a call with a trace id returns that session's report through the MCP wire.
func TestCtxValueMCPTool(t *testing.T) {
	srv := newTestServer(t)
	list := resultMap(t, rpcRoundTrip(t, srv, "tools/list", ""))
	tools, ok := list["tools"].([]any)
	if !ok {
		t.Fatalf("tools/list malformed: %v", list)
	}
	found := false
	for _, raw := range tools {
		if raw.(map[string]any)["name"] == "fak_context_value" {
			found = true
		}
	}
	if !found {
		t.Fatal("tools/list missing fak_context_value")
	}

	// Two served turns for a known trace, then query it over MCP.
	srv.logInferenceTurn("t-mcp", "anthropic_messages", false,
		agent.Usage{PromptTokens: 200, CompletionTokens: 5, CacheReadInputTokens: 1000}, "end_turn", time.Millisecond, false)
	srv.logInferenceTurn("t-mcp", "anthropic_messages", false,
		agent.Usage{PromptTokens: 300, CompletionTokens: 5, CacheReadInputTokens: 1500}, "end_turn", time.Millisecond, false)

	r := callMCPTool[CtxValueReport](t, srv, "fak_context_value", map[string]any{"trace_id": "t-mcp"})
	if r.TraceID != "t-mcp" || r.Turns.TurnsObserved != 2 {
		t.Fatalf("MCP report = %+v, want trace t-mcp with 2 turns", r)
	}
	if r.StepAdvice.StepClass == "" || r.StepAdvice.Reason == "" {
		t.Fatalf("MCP report advice = %+v, want a decidable class + reason", r.StepAdvice)
	}
}
