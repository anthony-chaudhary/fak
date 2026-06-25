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

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// stubGateway records the last control request and serves canned drive state.
type stubGateway struct {
	lastMethod, lastPath, lastVerb string
	lastBody                       gateway.SessionControlRequest
	curBudget                      gateway.SessionBudget
	curPace                        gateway.SessionPace
	curRev                         uint64
	conflictID                     string // an id whose control POST returns 409
}

func (g *stubGateway) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/fak/sessions", func(w http.ResponseWriter, r *http.Request) {
		g.lastMethod, g.lastPath = r.Method, r.URL.Path
		writeTestJSON(w, 200, gateway.SessionListResponse{
			Count: 2,
			Sessions: []gateway.SessionState{
				{TraceID: "urgent", Run: "running", Priority: 0, Rev: 1,
					Budget: gateway.SessionBudget{TurnsLeft: -1, TokensLeft: -1}},
				{TraceID: "bg", Run: "throttled", Priority: 5, Reason: "operator-throttle", Rev: 4,
					Budget: gateway.SessionBudget{TurnsLeft: 3, TokensLeft: -1}},
			},
		})
	})
	mux.HandleFunc("/v1/fak/session/", func(w http.ResponseWriter, r *http.Request) {
		g.lastMethod, g.lastPath = r.Method, r.URL.Path
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

func TestSessionCLIBudgetPartialMergeFencesRev(t *testing.T) {
	// Current state: turns=7 tokens=-1 rev=5. A `--turns 3` partial update must
	// preserve tokens=-1 and fence the write with the observed rev (5).
	g := &stubGateway{curBudget: gateway.SessionBudget{TurnsLeft: 7, TokensLeft: -1}, curRev: 5}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	_, errb, code := runSessionAt(t, ts.URL, "budget", "sess-2", "--turns", "3")
	if code != 0 {
		t.Fatalf("budget exit = %d (%s)", code, errb)
	}
	if g.lastVerb != "budget" || g.lastBody.Budget == nil {
		t.Fatalf("budget verb missing body: verb=%q body=%+v", g.lastVerb, g.lastBody)
	}
	if g.lastBody.Budget.TurnsLeft != 3 || g.lastBody.Budget.TokensLeft != -1 {
		t.Fatalf("budget merge = %+v, want turns=3 tokens=-1 (preserved)", *g.lastBody.Budget)
	}
	if g.lastBody.IfRev != 5 {
		t.Fatalf("budget if_rev = %d, want 5 (the observed rev fences the read-modify-write)", g.lastBody.IfRev)
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
