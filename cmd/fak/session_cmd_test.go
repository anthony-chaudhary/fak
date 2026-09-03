package main

// session_cmd_test.go — exercises the `fak session` operator CLI (runSession)
// against a stub gateway that speaks the /v1/fak/session(s) wire: the read verbs
// (ls/status), the run-state family (stop/pause/resume/throttle/run), the
// partial budget/pace merge (read-modify-write fenced with the observed rev), the
// priority verb, and the error/usage exit codes. The real route↔table wiring is
// covered by session_control_test.go and the gateway route tests; this proves the
// CLI builds the right requests and renders the results.

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// stubGateway records the last control request and serves canned drive state.
type stubGateway struct {
	lastMethod, lastPath, lastVerb string
	lastRawQuery                   string
	lastBody                       gateway.SessionControlRequest
	verbs                          []string
	bodies                         []gateway.SessionControlRequest
	curBudget                      gateway.SessionBudget
	curPace                        gateway.SessionPace
	curRev                         uint64
	conflictID                     string // an id whose control POST returns 409
	ctxValue                       gateway.CtxValueSnapshot
	sessions                       []gateway.SessionState
}

func (g *stubGateway) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/fak/sessions", func(w http.ResponseWriter, r *http.Request) {
		g.lastMethod, g.lastPath = r.Method, r.URL.Path
		g.lastRawQuery = r.URL.RawQuery
		respSessions := g.sessions
		if respSessions == nil {
			respSessions = []gateway.SessionState{
				{TraceID: "urgent", Run: "running", Priority: 0, Rev: 1,
					Budget: gateway.SessionBudget{TurnsLeft: -1, TokensLeft: -1}},
				{TraceID: "bg", Run: "throttled", Priority: 5, Reason: "operator-throttle", Rev: 4,
					Budget: gateway.SessionBudget{TurnsLeft: 3, TokensLeft: -1}},
			}
		}
		writeTestJSON(w, 200, gateway.SessionListResponse{
			Count:    len(respSessions),
			Sessions: respSessions,
		})
	})
	mux.HandleFunc("/v1/fak/ctxvalue", func(w http.ResponseWriter, r *http.Request) {
		g.lastMethod, g.lastPath = r.Method, r.URL.Path
		g.lastRawQuery = r.URL.RawQuery
		writeTestJSON(w, 200, g.ctxValue)
	})
	mux.HandleFunc("/v1/fak/session/", func(w http.ResponseWriter, r *http.Request) {
		g.lastMethod, g.lastPath = r.Method, r.URL.Path
		g.lastRawQuery = r.URL.RawQuery
		rest := strings.TrimPrefix(r.URL.Path, "/v1/fak/session/")
		parts := strings.Split(rest, "/")
		id := parts[0]
		if r.Method == http.MethodGet {
			writeTestJSON(w, 200, gateway.SessionState{
				TraceID: id, Run: "running", Budget: g.curBudget, Pace: g.curPace, Rev: g.curRev,
			})
			return
		}
		// POST {id}/{verb}
		g.lastVerb = parts[1]
		_ = json.NewDecoder(r.Body).Decode(&g.lastBody)
		g.verbs = append(g.verbs, g.lastVerb)
		g.bodies = append(g.bodies, g.lastBody)
		if id == g.conflictID {
			writeTestJSON(w, http.StatusConflict, map[string]any{
				"error": map[string]any{"message": "session control refused (terminal or stale rev)"},
			})
			return
		}
		st := gateway.SessionState{TraceID: id, Run: "running", Rev: g.curRev + 1}
		if g.lastBody.Run != "" {
			st.Run = g.lastBody.Run
			st.Reason = g.lastBody.Reason
		}
		if g.lastBody.Budget != nil {
			st.Budget = *g.lastBody.Budget
		}
		if g.lastBody.Pace != nil {
			st.Pace = *g.lastBody.Pace
		}
		if g.lastBody.Priority != nil {
			st.Priority = *g.lastBody.Priority
		}
		writeTestJSON(w, 200, st)
	})
	return mux
}

func writeTestJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func runSessionAt(t *testing.T, addr string, args ...string) (string, string, int) {
	t.Helper()
	var out, errb bytes.Buffer
	argv := append([]string{}, args...)
	argv = append(argv, "--addr", addr)
	code := runSession(&out, &errb, argv)
	return out.String(), errb.String(), code
}

func TestSessionCLIStatusAndList(t *testing.T) {
	g := &stubGateway{curBudget: gateway.SessionBudget{TurnsLeft: -1, TokensLeft: -1}, curRev: 2}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	out, errb, code := runSessionAt(t, ts.URL, "status", "sess-1")
	if code != 0 {
		t.Fatalf("status exit = %d (%s)", code, errb)
	}
	if !strings.Contains(out, "sess-1") || !strings.Contains(out, "running") || !strings.Contains(out, "budget(turns=inf") {
		t.Fatalf("status output missing fields: %q", out)
	}
	if g.lastMethod != http.MethodGet || g.lastPath != "/v1/fak/session/sess-1" {
		t.Fatalf("status hit %s %s, want GET /v1/fak/session/sess-1", g.lastMethod, g.lastPath)
	}

	out, errb, code = runSessionAt(t, ts.URL, "ls")
	if code != 0 {
		t.Fatalf("ls exit = %d (%s)", code, errb)
	}
	if !strings.Contains(out, "urgent") || !strings.Contains(out, "bg") || !strings.Contains(out, "2 session(s)") {
		t.Fatalf("ls output missing fields: %q", out)
	}
}

func TestSessionCLIStopAndRunVerbs(t *testing.T) {
	g := &stubGateway{curRev: 1}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	_, errb, code := runSessionAt(t, ts.URL, "stop", "sess-9", "--reason", "operator-cancel")
	if code != 0 {
		t.Fatalf("stop exit = %d (%s)", code, errb)
	}
	if g.lastVerb != "run" || g.lastBody.Run != "stopped" || g.lastBody.Reason != "operator-cancel" {
		t.Fatalf("stop sent verb=%q body=%+v, want run/stopped/operator-cancel", g.lastVerb, g.lastBody)
	}

	if _, errb, code := runSessionAt(t, ts.URL, "pause", "sess-9"); code != 0 || g.lastBody.Run != "paused" {
		t.Fatalf("pause: code=%d run=%q (%s)", code, g.lastBody.Run, errb)
	}
	if _, _, code := runSessionAt(t, ts.URL, "resume", "sess-9"); code != 0 || g.lastBody.Run != "running" {
		t.Fatalf("resume: code=%d run=%q", code, g.lastBody.Run)
	}
}

func TestSessionCLITerminateMapsToTerminating(t *testing.T) {
	g := &stubGateway{curRev: 1}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	_, errb, code := runSessionAt(t, ts.URL, "terminate", "sess-9", "--reason", "operator-force-stop")
	if code != 0 {
		t.Fatalf("terminate exit = %d (%s)", code, errb)
	}
	if g.lastVerb != "run" || g.lastBody.Run != "terminating" || g.lastBody.Reason != "operator-force-stop" {
		t.Fatalf("terminate sent verb=%q body=%+v, want run/terminating/operator-force-stop", g.lastVerb, g.lastBody)
	}
}
func TestSessionCLIBudgetPartialMergeFencesRev(t *testing.T) {
	// Current state: turns=7 tokens=-1 context=150 rev=5. A `--turns 3` partial
	// update must preserve the other axes and fence the write with the observed rev.
	g := &stubGateway{curBudget: gateway.SessionBudget{TurnsLeft: 7, TokensLeft: -1, ContextTokensLeft: 150}, curRev: 5}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	_, errb, code := runSessionAt(t, ts.URL, "budget", "sess-2", "--turns", "3")
	if code != 0 {
		t.Fatalf("budget exit = %d (%s)", code, errb)
	}
	if g.lastVerb != "budget" || g.lastBody.Budget == nil {
		t.Fatalf("budget verb missing body: verb=%q body=%+v", g.lastVerb, g.lastBody)
	}
	if g.lastBody.Budget.TurnsLeft != 3 || g.lastBody.Budget.TokensLeft != -1 || g.lastBody.Budget.ContextTokensLeft != 150 {
		t.Fatalf("budget merge = %+v, want turns=3 tokens=-1 context=150 (preserved)", *g.lastBody.Budget)
	}
	if g.lastBody.IfRev != 5 {
		t.Fatalf("budget if_rev = %d, want 5 (the observed rev fences the read-modify-write)", g.lastBody.IfRev)
	}
}

func TestSessionCLIContextBudget(t *testing.T) {
	g := &stubGateway{curBudget: gateway.SessionBudget{TurnsLeft: -1, TokensLeft: -1}, curRev: 6}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	out, errb, code := runSessionAt(t, ts.URL, "budget", "sess-context", "--context-tokens", "150000")
	if code != 0 {
		t.Fatalf("context budget exit = %d (%s)", code, errb)
	}
	if g.lastVerb != "budget" || g.lastBody.Budget == nil {
		t.Fatalf("context budget verb missing body: verb=%q body=%+v", g.lastVerb, g.lastBody)
	}
	if g.lastBody.Budget.TurnsLeft != -1 || g.lastBody.Budget.TokensLeft != -1 || g.lastBody.Budget.ContextTokensLeft != 150000 {
		t.Fatalf("context budget merge = %+v, want turns=-1 tokens=-1 context=150000", *g.lastBody.Budget)
	}
	if !strings.Contains(out, "context=150000") {
		t.Fatalf("context budget output missing axis: %q", out)
	}
}

func TestSessionCLIContextValue(t *testing.T) {
	g := &stubGateway{ctxValue: gateway.CtxValueSnapshot{
		Schema:       "fak-ctxvalue-report/1",
		BudgetTokens: 8000,
		Sessions: []gateway.CtxValueReport{{
			Schema:  "fak-ctxvalue-report/1",
			TraceID: "sess-context",
			Tokens: gateway.CtxValueTokens{
				ResidentTokens:     6500,
				PeakResidentTokens: 7000,
				BudgetTokens:       8000,
				Headroom:           &gateway.CtxValueHeadroom{Tokens: 1500, Pct: 18.75},
				GrowthPerTurn:      300,
				Provenance:         "OBSERVED",
			},
			Turns: gateway.CtxValueTurns{
				TurnsObserved:          9,
				ContextEvents:          1,
				TurnsSinceContextEvent: 5,
				Provenance:             "WITNESSED",
			},
			Session: gateway.CtxValueSession{
				Phase:      "crowding",
				Provenance: "WITNESSED",
			},
			StepAdvice: gateway.CtxStepAdvice{
				StepClass:  gateway.StepClassCheckpoint,
				Basis:      "token_headroom",
				Reason:     "resident 6.5K of 8K budget",
				Provenance: "DECISION",
			},
		}},
	}}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	out, errb, code := runSessionAt(t, ts.URL, "context", "sess-context")
	if code != 0 {
		t.Fatalf("context exit = %d (%s)", code, errb)
	}
	if g.lastMethod != http.MethodGet || g.lastPath != "/v1/fak/ctxvalue" || g.lastRawQuery != "trace=sess-context" {
		t.Fatalf("context hit %s %s?%s, want GET /v1/fak/ctxvalue?trace=sess-context",
			g.lastMethod, g.lastPath, g.lastRawQuery)
	}
	for _, want := range []string{"sess-context context", "step=checkpoint", "basis=token_headroom", "phase=crowding", "resident=6500", "headroom=1500(18.8%)", "events=1"} {
		if !strings.Contains(out, want) {
			t.Fatalf("context output missing %q:\n%s", want, out)
		}
	}

	out, errb, code = runSessionAt(t, ts.URL, "context", "sess-context", "--json")
	if code != 0 {
		t.Fatalf("context json exit = %d (%s)", code, errb)
	}
	if !strings.Contains(out, `"schema": "fak-ctxvalue-report/1"`) || !strings.Contains(out, `"step_class": "checkpoint"`) {
		t.Fatalf("context json missing ctxvalue report:\n%s", out)
	}
}

func TestSessionCLIPriorityAndPace(t *testing.T) {
	g := &stubGateway{curPace: gateway.SessionPace{MaxTokensPerTurn: 0, MinTurnGapMs: 0}, curRev: 1}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	if _, errb, code := runSessionAt(t, ts.URL, "priority", "sess-3", "7"); code != 0 {
		t.Fatalf("priority exit = %d (%s)", code, errb)
	}
	if g.lastBody.Priority == nil || *g.lastBody.Priority != 7 {
		t.Fatalf("priority body = %+v, want 7", g.lastBody)
	}

	if _, errb, code := runSessionAt(t, ts.URL, "pace", "sess-3", "--max-tokens", "256"); code != 0 {
		t.Fatalf("pace exit = %d (%s)", code, errb)
	}
	if g.lastBody.Pace == nil || g.lastBody.Pace.MaxTokensPerTurn != 256 {
		t.Fatalf("pace body = %+v, want max=256", g.lastBody)
	}
}

func TestSessionCLIBudgetEnvelopeInspectOnly(t *testing.T) {
	out, errb, code := runSessionAt(t, "http://127.0.0.1:1", "envelope", "sess-env", "turns=5,tokens=1000,context=64000,wall=30m,spend=$2.50,throughput=20/s,max-tokens=256,gap=100ms", "--inspect-only")
	if code != 0 {
		t.Fatalf("envelope inspect exit = %d (%s)", code, errb)
	}
	for _, want := range []string{"budget-envelope", "turns=5", "tokens=1000", "context=64000", "wall=30m0s", "spend=USD 2.50", "throughput=20/s", "pace(max=256 gap=100ms)"} {
		if !strings.Contains(out, want) {
			t.Fatalf("inspect output missing %q:\n%s", want, out)
		}
	}
}

func TestSessionCLIBudgetEnvelopeAppliesBudgetAndPace(t *testing.T) {
	g := &stubGateway{curBudget: gateway.SessionBudget{TurnsLeft: -1, TokensLeft: -1}, curRev: 8}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	out, errb, code := runSessionAt(t, ts.URL, "envelope", "sess-env", "turns=4,tokens=900,context=1200,max-tokens=128,gap=75ms")
	if code != 0 {
		t.Fatalf("envelope apply exit = %d (%s)", code, errb)
	}
	if g.lastVerb != "pace" || g.lastBody.Pace == nil {
		t.Fatalf("final envelope verb = %q body=%+v, want pace body", g.lastVerb, g.lastBody)
	}
	if len(g.verbs) != 2 || g.verbs[0] != "budget" || g.verbs[1] != "pace" {
		t.Fatalf("envelope verbs = %v, want budget then pace", g.verbs)
	}
	if got := g.bodies[0].Budget; got == nil || got.TurnsLeft != 4 || got.TokensLeft != 900 || got.ContextTokensLeft != 1200 {
		t.Fatalf("budget body = %+v, want turns=4 tokens=900 context=1200", got)
	}
	if g.lastBody.Pace.MaxTokensPerTurn != 128 || g.lastBody.Pace.MinTurnGapMs != 75 {
		t.Fatalf("pace body = %+v, want max=128 gap=75", *g.lastBody.Pace)
	}
	if g.lastBody.IfRev != 9 {
		t.Fatalf("pace if_rev = %d, want first write's returned rev 9", g.lastBody.IfRev)
	}
	for _, want := range []string{"applied: budget,pace", "context=1200", "pace(max=128 gap=75ms)"} {
		if !strings.Contains(out, want) {
			t.Fatalf("envelope output missing %q:\n%s", want, out)
		}
	}
}

// TestSessionCLIBudgetEnvelopeAppliesAllAxes is the #2762 control-route witness:
// an envelope stating every axis issues one control verb per axis — budget
// (with the spend ceiling riding the budget wire), pace, wall, throughput — so
// no stated axis is parsed-and-dropped anymore.
func TestSessionCLIBudgetEnvelopeAppliesAllAxes(t *testing.T) {
	g := &stubGateway{curBudget: gateway.SessionBudget{TurnsLeft: -1, TokensLeft: -1}, curRev: 8}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	out, errb, code := runSessionAt(t, ts.URL, "envelope", "sess-env",
		"turns=4,tokens=900,wall=30m,spend=$2.50,throughput=20/s,min-tps=5/s,max-tokens=128,gap=75ms")
	if code != 0 {
		t.Fatalf("envelope apply exit = %d (%s)", code, errb)
	}
	if len(g.verbs) != 4 || g.verbs[0] != "budget" || g.verbs[1] != "pace" || g.verbs[2] != "wall" || g.verbs[3] != "throughput" {
		t.Fatalf("envelope verbs = %v, want budget,pace,wall,throughput", g.verbs)
	}
	if b := g.bodies[0].Budget; b == nil || b.SpendMicroCentsLeft != 250_000_000 || b.SpendMicroCentsCap != 250_000_000 {
		t.Fatalf("budget body = %+v, want spend=$2.50 as 250,000,000 micro-cents left+cap", b)
	}
	if w := g.bodies[2].Wall; w == nil || w.LimitNanos != int64(30*time.Minute) {
		t.Fatalf("wall body = %+v, want limit_nanos=30m", w)
	}
	if tp := g.bodies[3].Throughput; tp == nil || tp.ExpectedTokensPerSec != 20 || tp.MinTokensPerSec != 5 {
		t.Fatalf("throughput body = %+v, want expected=20 min=5", tp)
	}
	if !strings.Contains(out, "applied: budget,pace,wall,throughput") {
		t.Fatalf("envelope output missing full applied list:\n%s", out)
	}
}

// TestSessionCLIBudgetMergePreservesSpendCeiling: the partial budget
// read-modify-write must carry a live spend ceiling forward (#2762) — before the
// spend axis rode the budget wire, `fak session budget --turns N` silently
// cleared a launch-set dollar cap.
func TestSessionCLIBudgetMergePreservesSpendCeiling(t *testing.T) {
	g := &stubGateway{curBudget: gateway.SessionBudget{
		TurnsLeft: 7, TokensLeft: -1, SpendMicroCentsLeft: 900, SpendMicroCentsCap: 1000,
	}, curRev: 5}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	_, errb, code := runSessionAt(t, ts.URL, "budget", "sess-2", "--turns", "3")
	if code != 0 {
		t.Fatalf("budget exit = %d (%s)", code, errb)
	}
	b := g.lastBody.Budget
	if b == nil || b.TurnsLeft != 3 || b.SpendMicroCentsLeft != 900 || b.SpendMicroCentsCap != 1000 {
		t.Fatalf("budget merge = %+v, want turns=3 with spend 900/1000 preserved", b)
	}
}

func TestSessionCLIConflictExit1(t *testing.T) {
	g := &stubGateway{curRev: 1, conflictID: "term-1"}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	_, errb, code := runSessionAt(t, ts.URL, "stop", "term-1")
	if code != 1 {
		t.Fatalf("conflict exit = %d, want 1", code)
	}
	if !strings.Contains(errb, "409") {
		t.Fatalf("conflict stderr should mention 409: %q", errb)
	}
}

func TestSessionCLIUsageErrors(t *testing.T) {
	// No args at all ⇒ usage (exit 2).
	var out, errb bytes.Buffer
	if code := runSession(&out, &errb, nil); code != 2 {
		t.Fatalf("no-arg exit = %d, want 2", code)
	}
	// Unknown verb ⇒ exit 2.
	if _, _, code := runSessionAt(t, "http://127.0.0.1:0", "frobnicate", "x"); code != 2 {
		t.Fatalf("unknown-verb exit = %d, want 2", code)
	}
	// Missing positional id ⇒ exit 2 (call runSession directly: runSessionAt would
	// append --addr and that token would be misread as the id).
	var o2, e2 bytes.Buffer
	if code := runSession(&o2, &e2, []string{"status"}); code != 2 {
		t.Fatalf("missing-id exit = %d, want 2", code)
	}
	// budget with no axis flags ⇒ exit 2 (nothing to set).
	g := &stubGateway{}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()
	if _, _, code := runSessionAt(t, ts.URL, "budget", "sess-1"); code != 2 {
		t.Fatalf("empty-budget exit = %d, want 2", code)
	}
}

func TestSessionCLIRejectsLeftoverArgs(t *testing.T) {
	g := &stubGateway{curRev: 1}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	// A stray extra positional ⇒ exit 2 (not a silent drop).
	if _, _, code := runSessionAt(t, ts.URL, "priority", "sess-1", "7", "8"); code != 2 {
		t.Fatalf("extra-positional exit = %d, want 2", code)
	}
	// A flag placed BEFORE the id ⇒ exit 2 (it would otherwise be misread as the id).
	if _, _, code := runSessionAt(t, ts.URL, "status", "--json", "sess-1"); code != 2 {
		t.Fatalf("flag-before-id exit = %d, want 2", code)
	}
}

func TestSessionCLIEscapesIDInPath(t *testing.T) {
	// An id with a query char must reach the gateway WHOLE (escaped), not be split so a
	// DIFFERENT session is read. Without url.PathEscape, "sess?x" would target "sess".
	g := &stubGateway{curRev: 1, curBudget: gateway.SessionBudget{TurnsLeft: -1, TokensLeft: -1}}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	out, errb, code := runSessionAt(t, ts.URL, "status", "sess?danger")
	if code != 0 {
		t.Fatalf("status exit = %d (%s)", code, errb)
	}
	if !strings.Contains(out, "sess?danger") {
		t.Fatalf("escaped id did not round-trip whole; output=%q path=%q", out, g.lastPath)
	}
}

func TestSessionCLIResumeAll(t *testing.T) {
	g := &stubGateway{
		curRev: 1,
		sessions: []gateway.SessionState{
			{TraceID: "p-1", Run: "paused", Reason: "operator-hold", Rev: 2},
			{TraceID: "p-2", Run: "paused", Reason: "operator-hold", Rev: 3},
			{TraceID: "live", Run: "running", Rev: 1},
		},
	}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	// 1. Resume all paused sessions.
	out, errb, code := runSessionAt(t, ts.URL, "resume", "--all")
	if code != 0 {
		t.Fatalf("resume --all exit = %d (%s)", code, errb)
	}
	if !strings.Contains(out, "p-1") || !strings.Contains(out, "p-2") || !strings.Contains(out, "2 session(s) resumed") {
		t.Fatalf("resume --all output unexpected: %q", out)
	}
	if len(g.bodies) != 2 {
		t.Fatalf("control calls = %d, want 2", len(g.bodies))
	}
	for _, b := range g.bodies {
		if b.Run != "running" {
			t.Errorf("control body run = %q, want running", b.Run)
		}
	}

	// 2. Resume all with --json when no sessions are paused.
	g.sessions = []gateway.SessionState{
		{TraceID: "live", Run: "running", Rev: 1},
	}
	g.bodies = nil
	out, errb, code = runSessionAt(t, ts.URL, "resume", "--all", "--json")
	if code != 0 {
		t.Fatalf("resume --all --json exit = %d (%s)", code, errb)
	}
	if !strings.Contains(out, `"resumed": 0`) {
		t.Fatalf("resume --all --json output unexpected: %q", out)
	}

	// 3. resume with missing args (neither id nor --all).
	var out3, errb3 bytes.Buffer
	code = runSession(&out3, &errb3, []string{"resume"})
	if code != 2 {
		t.Fatalf("resume with no args exit = %d, want 2", code)
	}
	if !strings.Contains(errb3.String(), "--all") {
		t.Fatalf("expected usage message to mention --all, got: %q", errb3.String())
	}
}
