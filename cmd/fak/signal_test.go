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
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
)

// signalStub serves the run-control + steer routes and records what it saw.
type signalStub struct {
	lastPath    string
	lastRun     string
	lastReason  string
	lastSteer   gateway.SteerRequest
	steerStatus int    // override the steer response status (0 -> 202 accepted)
	steerCode   string // override the steer error envelope's OpenAI code (e.g. steer_no_owned_loop)
	steerMsg    string // override the steer error envelope's message
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
				msg := g.steerMsg
				if msg == "" {
					msg = "steer refused: a2a floor refused (TRUST_VIOLATION)"
				}
				writeTestJSON(w, g.steerStatus, map[string]any{
					"error": map[string]any{"message": msg, "code": g.steerCode},
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

// TestSignalSteerProxyRefusalIsHonest proves the CLI half of #3528: when the gateway
// refuses a steer with 409 STEER_NO_OWNED_LOOP (a proxy-served target), the operator sees a
// truthful, reason-specific line — NOT the misleading generic-409 "session is stopped /
// changed under you" text, and NEVER a false "delivered" claim.
func TestSignalSteerProxyRefusalIsHonest(t *testing.T) {
	g := &signalStub{
		steerStatus: http.StatusConflict,
		steerCode:   "steer_no_owned_loop",
		steerMsg:    "STEER_NO_OWNED_LOOP: this serve process forwards proxy turns and owns no agent loop; start the gateway with --native",
	}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()
	out, errb, code := runSignalAt(ts.URL, "steer", "sess-proxy", "--text", "switch to plan B")
	if code != 1 {
		t.Fatalf("proxy-refused steer exit = %d, want 1 (out=%q err=%q)", code, out, errb)
	}
	if strings.Contains(out, "delivered") || strings.Contains(out, "steered") {
		t.Fatalf("proxy-refused steer must not claim delivery; stdout=%q", out)
	}
	if !strings.Contains(errb, "STEER_NO_OWNED_LOOP") {
		t.Fatalf("refusal should name the reason; stderr=%q", errb)
	}
	if strings.Contains(errb, "stopped (terminal) or changed under you") {
		t.Fatalf("refusal must not use the misleading generic-409 text; stderr=%q", errb)
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

func TestSignalSteerResolvesDynamicGatewayFromIndex(t *testing.T) {
	t.Setenv("FAK_ADDR", "")
	t.Setenv("FAK_KEY", "")
	g := &signalStub{}
	var gotAuth string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		g.handler().ServeHTTP(w, r)
	}))
	defer ts.Close()

	dir := t.TempDir()
	row := guardsessions.NewRow("tr-dyn-1", "claude", os.Getpid(), `C:\work\x`, "", "", time.Now()).
		WithGateway(ts.URL, "read-only-bearer-token")
	if err := guardsessions.Record(dir, row); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runSignal(&out, &errb, []string{"steer", row.Handle, "--text", "steer to dynamic", "--reg-dir", dir})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, errb.String())
	}
	if g.lastPath != "/v1/fak/session/tr-dyn-1/steer" {
		t.Fatalf("path = %q, want /v1/fak/session/tr-dyn-1/steer", g.lastPath)
	}
	if g.lastSteer.Text != "steer to dynamic" {
		t.Fatalf("text = %q, want 'steer to dynamic'", g.lastSteer.Text)
	}
	if strings.Contains(gotAuth, "read-only-bearer-token") {
		t.Fatalf("control request must not reuse the read-scoped published bearer, got auth header: %q", gotAuth)
	}
	if !strings.Contains(out.String(), "steered tr-dyn-1") {
		t.Fatalf("stdout=%q, want 'steered tr-dyn-1'", out.String())
	}
}

func TestSignalSteerIndexAmbiguityReportsCandidates(t *testing.T) {
	t.Setenv("FAK_ADDR", "")
	dir := t.TempDir()
	row1 := guardsessions.NewRow("tr-ambig-1", "claude", os.Getpid(), `C:\work\x`, "", "", time.Now()).
		WithGateway("http://127.0.0.1:50001", "tok1")
	row1.Handle = "handle-ambig-1"
	row2 := guardsessions.NewRow("tr-ambig-2", "claude", os.Getpid(), `C:\work\x`, "", "", time.Now()).
		WithGateway("http://127.0.0.1:50002", "tok2")
	row2.Handle = "handle-ambig-2"
	if err := guardsessions.Record(dir, row1); err != nil {
		t.Fatal(err)
	}
	if err := guardsessions.Record(dir, row2); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runSignal(&out, &errb, []string{"steer", "handle-ambig", "--text", "hello", "--reg-dir", dir})
	if code != 3 {
		t.Fatalf("ambiguous exit = %d, want 3 (stderr=%s)", code, errb.String())
	}
	if !strings.Contains(errb.String(), "matches 2 guard sessions — narrow the prefix") {
		t.Fatalf("stderr=%q, want ambiguity message naming 2 sessions", errb.String())
	}
}

func TestSignalSteerMissingGatewayURLReportsActionableError(t *testing.T) {
	t.Setenv("FAK_ADDR", "")
	dir := t.TempDir()
	row := guardsessions.NewRow("tr-nogw-1", "claude", os.Getpid(), `C:\work\x`, "", "", time.Now())
	row.GatewayURL = ""
	if err := guardsessions.Record(dir, row); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runSignal(&out, &errb, []string{"steer", row.Handle, "--text", "hello", "--reg-dir", dir})
	if code != 1 {
		t.Fatalf("missing gateway exit = %d, want 1 (stderr=%s)", code, errb.String())
	}
	if !strings.Contains(errb.String(), "published no gateway_url") {
		t.Fatalf("stderr=%q, want missing gateway_url explanation", errb.String())
	}
}

func TestSignalSteerExplicitAddrPrecedence(t *testing.T) {
	g1 := &signalStub{}
	ts1 := httptest.NewServer(g1.handler())
	defer ts1.Close()

	g2 := &signalStub{}
	ts2 := httptest.NewServer(g2.handler())
	defer ts2.Close()

	dir := t.TempDir()
	row := guardsessions.NewRow("tr-prec-1", "claude", os.Getpid(), `C:\work\x`, "", "", time.Now()).
		WithGateway(ts1.URL, "tok1")
	if err := guardsessions.Record(dir, row); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runSignal(&out, &errb, []string{"steer", row.Handle, "--text", "explicit wins", "--addr", ts2.URL, "--reg-dir", dir})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, errb.String())
	}
	if g2.lastPath != "/v1/fak/session/"+row.Handle+"/steer" {
		t.Fatalf("ts2 path = %q, want hit on ts2", g2.lastPath)
	}
	if g1.lastPath != "" {
		t.Fatalf("ts1 was hit unexpectedly: %q", g1.lastPath)
	}
}

func TestSignalSteerTransportFailureActionable(t *testing.T) {
	t.Setenv("FAK_ADDR", "")
	dir := t.TempDir()
	row := guardsessions.NewRow("tr-dead-1", "claude", os.Getpid(), `C:\work\x`, "", "", time.Now()).
		WithGateway("http://127.0.0.1:59999", "tok")
	if err := guardsessions.Record(dir, row); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runSignal(&out, &errb, []string{"steer", row.Handle, "--text", "hello", "--reg-dir", dir})
	if code != 1 {
		t.Fatalf("transport failure exit = %d, want 1", code)
	}
	if !strings.Contains(errb.String(), "fak signal steer:") {
		t.Fatalf("stderr=%q, want actionable steer error", errb.String())
	}
}
