package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// agent_sessions_test.go — the spine endpoint's offline witness (#3258): a POSTed
// goal drives one REAL kernel-governed loop (agent.RunGovernedArm over the
// deterministic MockPlanner — no live model, no network) and the streamed NDJSON
// events carry the witnessed decision trace: the kernel's quarantine of the
// poisoned policy fetch and the terminal ArmMetrics with the booking completed.

func newAgentSessionsTestServer(t *testing.T) *Server {
	t.Helper()
	srv, err := New(Config{EngineID: "mock", Model: "mock", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return srv
}

func postAgentSession(t *testing.T, srv *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/fak/agent/sessions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	return rec
}

func decodeAgentSessionEvents(t *testing.T, body string) []AgentSessionEvent {
	t.Helper()
	var events []AgentSessionEvent
	for i, line := range strings.Split(strings.TrimSpace(body), "\n") {
		var ev AgentSessionEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("line %d is not a JSON event: %v (%q)", i, err, line)
		}
		events = append(events, ev)
	}
	return events
}

// TestAgentSessionsStreamsGovernedLoop is the endpoint's core witness: a POSTed
// goal streams session.start, per-call adjudicated rows from the fak arm, and a
// terminal session.end whose ArmMetrics show the governed loop ran to the goal.
func TestAgentSessionsStreamsGovernedLoop(t *testing.T) {
	srv := newAgentSessionsTestServer(t)
	rec := postAgentSession(t, srv, `{"goal":"Book the cheapest direct SFO-JFK flight for mia_li_3668."}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/x-ndjson" {
		t.Fatalf("Content-Type = %q, want application/x-ndjson", ct)
	}
	events := decodeAgentSessionEvents(t, rec.Body.String())
	if len(events) < 3 {
		t.Fatalf("want >= 3 events (start, calls, end), got %d: %+v", len(events), events)
	}

	start := events[0]
	if start.Event != "session.start" || start.Goal == "" || start.MaxTurns <= 0 || start.TraceID == "" {
		t.Fatalf("first event is not a well-formed session.start: %+v", start)
	}

	end := events[len(events)-1]
	if end.Event != "session.end" || end.Metrics == nil {
		t.Fatalf("last event is not a session.end with metrics: %+v", end)
	}
	if end.Metrics.Arm != "fak" {
		t.Fatalf("metrics arm = %q, want the kernel-governed fak arm", end.Metrics.Arm)
	}
	if end.Metrics.Turns == 0 || !end.Metrics.TaskCompleted {
		t.Fatalf("governed loop did not run to the goal: %+v", *end.Metrics)
	}

	var calls, governed int
	for _, ev := range events[1 : len(events)-1] {
		if ev.Event != "call" || ev.Call == nil {
			t.Fatalf("middle event is not a call row: %+v", ev)
		}
		if ev.Call.Arm != "fak" {
			t.Fatalf("call row from arm %q, want fak: %+v", ev.Call.Arm, *ev.Call)
		}
		if ev.Call.Verdict == "" || ev.Call.Tool == "" || ev.Call.Turn == 0 {
			t.Fatalf("call row missing turn/tool/verdict: %+v", *ev.Call)
		}
		calls++
		if ev.Call.Verdict != "ALLOW" {
			governed++ // DENY / TRANSFORM / QUARANTINE — the kernel visibly intervened
		}
	}
	if calls == 0 {
		t.Fatal("no per-call verdict rows were streamed")
	}
	// The deterministic scenario carries an alias-prone call the kernel repairs
	// in-syscall and a poisoned policy fetch it contains (denied pre-execution or
	// result-quarantined, depending on which registered rung catches it) — so at
	// least one streamed row must show a non-ALLOW kernel verdict: the kernel
	// governing, not narrating.
	if governed == 0 {
		t.Fatalf("no non-ALLOW kernel verdict in the streamed trace (%d calls): %s", calls, rec.Body.String())
	}
	// The terminal metrics must agree: the in-syscall grammar repair fired and the
	// unsafe policy fetch was contained (a deny or a quarantine), and the duplicate
	// read-only lookup was vDSO-served without a second dispatch.
	if end.Metrics.Repairs < 1 {
		t.Fatalf("metrics repairs = %d, want >= 1: %+v", end.Metrics.Repairs, *end.Metrics)
	}
	if end.Metrics.Denies+end.Metrics.Quarantines < 1 {
		t.Fatalf("unsafe fetch neither denied nor quarantined: %+v", *end.Metrics)
	}
	if end.Metrics.VDSOHits < 1 {
		t.Fatalf("metrics vdso_hits = %d, want >= 1: %+v", end.Metrics.VDSOHits, *end.Metrics)
	}
}

// TestAgentSessionsClampsMaxTurns verifies the client can narrow but never widen
// the server's turn budget.
func TestAgentSessionsClampsMaxTurns(t *testing.T) {
	srv := newAgentSessionsTestServer(t)
	if got := srv.agentSessionMaxTurns(0); got != DefaultNativeMaxTurns {
		t.Fatalf("maxTurns(0) = %d, want the default cap %d", got, DefaultNativeMaxTurns)
	}
	if got := srv.agentSessionMaxTurns(3); got != 3 {
		t.Fatalf("maxTurns(3) = %d, want the narrowed 3", got)
	}
	if got := srv.agentSessionMaxTurns(DefaultNativeMaxTurns + 50); got != DefaultNativeMaxTurns {
		t.Fatalf("maxTurns(cap+50) = %d, want clamped to %d", got, DefaultNativeMaxTurns)
	}
}

// TestAgentSessionsRequestShape verifies the pre-stream request-shape floor:
// wrong method 405, malformed body 400, and a missing goal 400 — all as plain
// JSON errors, never a half-open stream.
func TestAgentSessionsRequestShape(t *testing.T) {
	srv := newAgentSessionsTestServer(t)

	req := httptest.NewRequest(http.MethodGet, "/v1/fak/agent/sessions", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET status = %d, want 405", rec.Code)
	}

	if rec := postAgentSession(t, srv, `{not json`); rec.Code != http.StatusBadRequest {
		t.Fatalf("malformed body status = %d, want 400", rec.Code)
	}
	if rec := postAgentSession(t, srv, `{"max_turns":4}`); rec.Code != http.StatusBadRequest {
		t.Fatalf("missing goal status = %d, want 400", rec.Code)
	}
}
