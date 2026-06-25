package main

// signal_test.go - proves `fak signal` job control over a stub gateway: pause/resume/stop
// map onto the live /v1/fak/session/{id}/run control verb (the OS names over the shipped
// control plane), and steer POSTs /v1/fak/session/{id}/steer, surfacing a 422 refusal.

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// signalStub serves the run-control + steer routes and records what it saw.
type signalStub struct {
	lastPath    string
	lastRun     string
	lastReason  string
	lastSteer   gateway.SteerRequest
	steerStatus int // override the steer response status (0 -> 202 accepted)
}

func (g *signalStub) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/fak/session/", func(w http.ResponseWriter, r *http.Request) {
		g.lastPath = r.URL.Path
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/v1/fak/session/"), "/")
		id := parts[0]
		verb := ""
		if len(parts) >= 2 {
			verb = parts[1]
		}
		if verb == "steer" {
			_ = json.NewDecoder(r.Body).Decode(&g.lastSteer)
			if g.steerStatus != 0 {
				writeTestJSON(w, g.steerStatus, map[string]any{
					"error": map[string]any{"message": "steer refused: a2a floor refused (TRUST_VIOLATION)"},
				})
				return
			}
			writeTestJSON(w, http.StatusAccepted, map[string]any{"trace_id": id, "steered": true})
			return
		}
		// run verb: record the requested run-state + reason, echo a new state.
		var req gateway.SessionControlRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		g.lastRun, g.lastReason = req.Run, req.Reason
		writeTestJSON(w, 200, gateway.SessionState{TraceID: id, Run: req.Run, Reason: req.Reason, Rev: 7})
	})
	return mux
}

func runSignalAt(addr string, args ...string) (string, string, int) {
	var out, errb bytes.Buffer
	argv := append(append([]string{}, args...), "--addr", addr)
	code := runSignal(&out, &errb, argv)
	return out.String(), errb.String(), code
}

func TestSignalPauseResumeStopMapToRunVerb(t *testing.T) {
	g := &signalStub{}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	cases := []struct{ verb, wantRun string }{
		{"pause", "paused"}, {"resume", "running"}, {"stop", "stopped"},
	}
	for _, c := range cases {
		_, errb, code := runSignalAt(ts.URL, c.verb, "sess-1")
		if code != 0 {
			t.Fatalf("%s exit = %d (%s)", c.verb, code, errb)
		}
		if g.lastPath != "/v1/fak/session/sess-1/run" {
			t.Fatalf("%s hit %s, want /v1/fak/session/sess-1/run", c.verb, g.lastPath)
		}
		if g.lastRun != c.wantRun {
			t.Fatalf("%s set run=%q, want %q", c.verb, g.lastRun, c.wantRun)
		}
	}
}

func TestSignalStopCarriesReason(t *testing.T) {
	g := &signalStub{}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()
	if _, errb, code := runSignalAt(ts.URL, "stop", "sess-2", "--reason", "operator-cancel"); code != 0 {
		t.Fatalf("stop exit = %d (%s)", code, errb)
	}
	if g.lastReason != "operator-cancel" {
		t.Fatalf("stop reason = %q, want operator-cancel", g.lastReason)
	}
}

func TestSignalSteerPostsText(t *testing.T) {
	g := &signalStub{}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()
	out, errb, code := runSignalAt(ts.URL, "steer", "sess-3", "--text", "switch to plan B")
	if code != 0 {
		t.Fatalf("steer exit = %d (%s)", code, errb)
	}
	if g.lastPath != "/v1/fak/session/sess-3/steer" {
		t.Fatalf("steer hit %s, want .../sess-3/steer", g.lastPath)
	}
	if g.lastSteer.Text != "switch to plan B" {
		t.Fatalf("steer text = %q, want 'switch to plan B'", g.lastSteer.Text)
	}
	if !strings.Contains(out, "steered sess-3") {
		t.Fatalf("steer output = %q, want a 'steered sess-3' ack", out)
	}
}

func TestSignalSteerRefusedSurfacesError(t *testing.T) {
	g := &signalStub{steerStatus: http.StatusUnprocessableEntity}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()
	out, errb, code := runSignalAt(ts.URL, "steer", "sess-4", "--text", "do X")
	if code != 1 {
		t.Fatalf("refused steer exit = %d, want 1 (out=%q err=%q)", code, out, errb)
	}
	if !strings.Contains(errb, "refused") {
		t.Fatalf("refused steer stderr = %q, want it to mention 'refused'", errb)
	}
}

func TestSignalSteerReadsStdin(t *testing.T) {
	g := &signalStub{}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()
	// Swap os.Stdin for a pipe carrying the steer text, restoring it after.
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	io.WriteString(w, "  piped steer input\n")
	w.Close()
	old := os.Stdin
	os.Stdin = r
	defer func() { os.Stdin = old }()

	if _, errb, code := runSignalAt(ts.URL, "steer", "sess-5", "--stdin"); code != 0 {
		t.Fatalf("stdin steer exit = %d (%s)", code, errb)
	}
	if g.lastSteer.Text != "piped steer input" {
		t.Fatalf("stdin steer text = %q, want trimmed 'piped steer input'", g.lastSteer.Text)
	}
}

func TestSignalUsageAndArity(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runSignal(&out, &errb, nil); code != 2 {
		t.Fatalf("no args exit = %d, want 2", code)
	}
	if code := runSignal(&out, &errb, []string{"bogus", "sess-1"}); code != 2 {
		t.Fatalf("unknown verb exit = %d, want 2", code)
	}
	if code := runSignal(&out, &errb, []string{"pause"}); code != 2 {
		t.Fatalf("missing id exit = %d, want 2", code)
	}
}
