package gateway

// ctxvalue_test.go — the managed-context value report (ctxvalue.go): the pure
// step-advice policy rung by rung, the rolling per-session accumulator (growth
// slope, context-event era reset), and the two query surfaces — GET
// /v1/fak/ctxvalue over the live wire and the fak_context_value MCP tool.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestAdviseCtxStepRungs walks the closed decision ladder: every step class is
// reachable, each from the rung that owns it, and the fail-closed rungs answer
// unknown instead of guessing.

func TestStepAdviceTokenStableWithAffordance(t *testing.T) {
	cases := []struct {
		name  string
		state ctxValueState
		want  StepClass
	}{
		{"any", ctxValueState{Turns: 5, Resident: 100, Budget: 1000}, StepClassAny},
		{"bounded", ctxValueState{Turns: 5, Resident: 600, Budget: 1000}, StepClassBounded},
		{"checkpoint", ctxValueState{Turns: 5, Resident: 900, Budget: 1000}, StepClassCheckpoint},
		{"rebuild", ctxValueState{Turns: 5, ContextEvents: 1, LastTurnEvent: true}, StepClassRebuild},
		{"unknown", ctxValueState{}, StepClassUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := adviseCtxStep(tc.state)
			if got.StepClass != tc.want {
				t.Fatalf("step_class=%q want %q", got.StepClass, tc.want)
			}
			if got.Affordance == "" {
				t.Fatalf("affordance empty: %+v", got)
			}
			b, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(b), `"step_class":"`+string(tc.want)+`"`) || !strings.Contains(string(b), `"affordance":`) {
				t.Fatalf("payload lost token or affordance: %s", b)
			}
		})
	}
}

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

// TestCtxValue_WakeupHorizonPresent proves the #2446 acceptance: a served turn
// makes the report carry the advice-only cache-window horizon, defaulting to the
// 5m tier, with the priced re-prefill cost of committing past the window; an
// unseen trace still answers with a decidable horizon rather than an error.
func TestCtxValue_WakeupHorizonPresent(t *testing.T) {
	s := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model", CompactHistoryBudget: 10000})
	s.observeCtxValue("t1", 3000, 100, 100, 10, false)
	r := s.CtxValueReportFor("t1")
	if r.Wakeup.TTLTier != "5m" {
		t.Fatalf("default tier = %q, want 5m", r.Wakeup.TTLTier)
	}
	if r.Wakeup.WindowMs != cacheTTLWindow5mMs || r.Wakeup.WakeupByMs != cacheTTLWindow5mMs {
		t.Fatalf("horizon = %+v, want the 5m window %d ms", r.Wakeup, cacheTTLWindow5mMs)
	}
	// resident = 3000 + 100 + 100 = 3200 re-prefills uncached if the window lapses.
	if r.Wakeup.PastWindowReprefillTokens != 3200 {
		t.Fatalf("re-prefill cost = %d, want 3200", r.Wakeup.PastWindowReprefillTokens)
	}
	if z := s.CtxValueReportFor("never-served"); z.Wakeup.TTLTier != "5m" || z.Wakeup.WindowProvenance != "OBSERVED" {
		t.Fatalf("zero-report horizon = %+v, want a decidable 5m/OBSERVED", z.Wakeup)
	}
}

// TestCtxValue_HorizonTracksTTLUpgrade proves the horizon shifts from the 5m to
// the 1h window once the 1h TTL-upgrade rung fires for the session.
func TestCtxValue_HorizonTracksTTLUpgrade(t *testing.T) {
	s := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model", CompactHistoryBudget: 10000})
	s.observeCtxValue("t1", 2000, 0, 0, 10, false)
	if r := s.CtxValueReportFor("t1"); r.Wakeup.TTLTier != "5m" || r.Wakeup.WindowMs != cacheTTLWindow5mMs {
		t.Fatalf("pre-upgrade horizon = %+v, want 5m", r.Wakeup)
	}
	// The 1h TTL-upgrade rung fires for this session (as maybeUpgradeAnthropicCacheTTL1H does).
	s.noteCtxValueTTL1h("t1")
	r := s.CtxValueReportFor("t1")
	if r.Wakeup.TTLTier != "1h" || r.Wakeup.WindowMs != cacheTTLWindow1hMs || r.Wakeup.WakeupByMs != cacheTTLWindow1hMs {
		t.Fatalf("post-upgrade horizon = %+v, want the 1h window %d ms", r.Wakeup, cacheTTLWindow1hMs)
	}
	// The note-minted flag alone never conjures a phantom into the multi-session snapshot.
	s.noteCtxValueTTL1h("note-only")
	if snap := s.ctxValueSnapshot(""); len(snap.Sessions) != 1 || snap.Sessions[0].TraceID != "t1" {
		t.Fatalf("snapshot = %+v, want only the served t1 session", snap.Sessions)
	}
}

// TestCtxValue_ProvenanceLabels proves the Law-A2 split: the provider-owned TTL
// window is OBSERVED, the tier decision and horizon arithmetic are WITNESSED, and
// the pure fold agrees with the served-report path.
func TestCtxValue_ProvenanceLabels(t *testing.T) {
	s := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model", CompactHistoryBudget: 10000})
	s.observeCtxValue("t1", 1000, 0, 0, 10, false)
	r := s.CtxValueReportFor("t1")
	if r.Wakeup.WindowProvenance != "OBSERVED" {
		t.Fatalf("window provenance = %q, want OBSERVED (provider-owned TTL)", r.Wakeup.WindowProvenance)
	}
	if r.Wakeup.HorizonProvenance != "WITNESSED" {
		t.Fatalf("horizon provenance = %q, want WITNESSED (fak arithmetic)", r.Wakeup.HorizonProvenance)
	}
	if got := ctxWakeupHorizon(false, 1000); got != r.Wakeup {
		t.Fatalf("pure horizon %+v != report horizon %+v", got, r.Wakeup)
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

// TestCtxValueMCPTool proves the cold value schema remains discoverable through
// fak_tools_search and callable with a trace id through the MCP wire.
func TestCtxValueMCPTool(t *testing.T) {
	srv := newTestServer(t)
	search, err := srv.toolsSearch(ToolsSearchRequest{Query: "managed context pressure", DetailLevel: "name"})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, descriptor := range search.Tools {
		if descriptor["name"] == "fak_context_value" {
			found = true
		}
	}
	if !found {
		t.Fatal("fak_tools_search missing fak_context_value")
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

func TestObserveCtxValueCompactionFiresAutoCheckpoint(t *testing.T) {
	fired := make(chan string, 1)
	s := &Server{compactHistoryBudget: 1000, autoCheckpoint: func(session, reason string) { fired <- session + ":" + reason }}
	s.observeCtxValue("session-ctx", 500, 100, 100, 10, true)
	select {
	case got := <-fired:
		if got != "session-ctx:compaction" {
			t.Fatalf("callback=%q", got)
		}
	case <-time.After(time.Second):
		t.Fatal("compaction rebuild advice did not auto-checkpoint")
	}
}
