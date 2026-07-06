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
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// panicPlanner reproduces the #2336 failure class synthetically: a native
// served model whose forward pass panics on EVERY completion turn (the GLM-5.2
// q3 "glm_moe_dsa attention step failed" shape), with no private weights needed.
type panicPlanner struct{ msg string }

func (p panicPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	panic(p.msg)
}
func (panicPlanner) Model() string { return "glm-5.2-q3-native" }

// oaiErrEnvelope is the OpenAI-style error envelope both wires share.
type oaiErrEnvelope struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    any    `json:"code"`
	} `json:"error"`
}

func postServedTurn(t *testing.T, url string, body map[string]any) (*http.Response, []byte) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(url, "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("POST %s: connection-level failure (want a structured HTTP response, not a reset): %v", url, err)
	}
	defer resp.Body.Close()
	respRaw, _ := io.ReadAll(resp.Body)
	return resp, respRaw
}

// TestServedPanicCompletionReturnsStructuredError pins the #2336 request half:
// a served planner/model panic on the OpenAI-compatible completion routes must
// come back as a structured 5xx OpenAI-style error envelope — a parseable
// server_error a client/watchdog can classify — never a reset connection or a
// bare text body.
func TestServedPanicCompletionReturnsStructuredError(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = panicPlanner{msg: "glm_moe_dsa attention step failed"}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	cases := []struct {
		path string
		body map[string]any
	}{
		{"/v1/chat/completions", map[string]any{
			"model":    "test-model",
			"messages": []map[string]any{{"role": "user", "content": "hi"}},
		}},
		{"/v1/completions", map[string]any{
			"model":  "test-model",
			"prompt": "hi",
		}},
	}
	for _, c := range cases {
		resp, raw := postServedTurn(t, ts.URL+c.path, c.body)
		if resp.StatusCode != http.StatusInternalServerError {
			t.Fatalf("%s status = %d, want 500 (%s)", c.path, resp.StatusCode, raw)
		}
		if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
			t.Fatalf("%s Content-Type = %q, want application/json envelope, body=%q", c.path, ct, raw)
		}
		var env oaiErrEnvelope
		if err := json.Unmarshal(raw, &env); err != nil {
			t.Fatalf("%s body is not the structured error envelope: %v (%s)", c.path, err, raw)
		}
		if env.Error.Type != "server_error" {
			t.Fatalf("%s error.type = %q, want server_error (%s)", c.path, env.Error.Type, raw)
		}
		if !strings.Contains(env.Error.Message, "glm_moe_dsa attention step failed") {
			t.Fatalf("%s error.message = %q, want the recovered panic value visible", c.path, env.Error.Message)
		}
	}
}

// TestHealthzQualifiedAfterServedPanic pins the #2336 health half: /healthz must
// stop reporting an unqualified ok:true once a served completion turn has
// recently panicked (the operator trap: watchdogs keep routing to a broken
// native serve), and must return to the plain healthy report once the failure
// ages out of the window — the healthy path stays green.
func TestHealthzQualifiedAfterServedPanic(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = panicPlanner{msg: "glm_moe_dsa attention step failed"}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var before map[string]any
	getJSON(t, ts.URL+"/healthz", &before)
	if before["ok"] != true {
		t.Fatalf("pre-panic /healthz = %+v, want ok:true", before)
	}
	if _, present := before["recent_served_failure"]; present {
		t.Fatalf("pre-panic /healthz already qualified: %+v", before)
	}

	resp, raw := postServedTurn(t, ts.URL+"/v1/chat/completions", map[string]any{
		"model":    "test-model",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("served turn status = %d, want 500 (%s)", resp.StatusCode, raw)
	}

	var after map[string]any
	getJSON(t, ts.URL+"/healthz", &after)
	if after["ok"] != false {
		t.Fatalf("post-panic /healthz = %+v, want ok:false while the served failure is recent", after)
	}
	fail, ok := after["recent_served_failure"].(map[string]any)
	if !ok {
		t.Fatalf("post-panic /healthz missing recent_served_failure detail: %+v", after)
	}
	if fail["route"] != "/v1/chat/completions" {
		t.Fatalf("recent_served_failure.route = %v, want /v1/chat/completions", fail["route"])
	}
	if msg, _ := fail["error"].(string); !strings.Contains(msg, "glm_moe_dsa attention step failed") {
		t.Fatalf("recent_served_failure.error = %v, want the recovered panic value", fail["error"])
	}

	// Age the marker past the window: the qualification is time-bounded honesty,
	// not a permanent condemnation — an aged-out failure restores the plain
	// healthy report.
	srv.servedFailure.mu.Lock()
	srv.servedFailure.at = time.Now().Add(-servedFailureWindow - time.Second)
	srv.servedFailure.mu.Unlock()

	var aged map[string]any
	getJSON(t, ts.URL+"/healthz", &aged)
	if aged["ok"] != true {
		t.Fatalf("aged-out /healthz = %+v, want ok:true restored", aged)
	}
	if _, present := aged["recent_served_failure"]; present {
		t.Fatalf("aged-out /healthz still qualified: %+v", aged)
	}
}

// TestServedTurnPathScopesHealthMarker pins the marker's scope: only served
// model-turn routes qualify /healthz. A control-plane handler panic still gets
// the contained 500, but must not condemn serving health.
func TestServedTurnPathScopesHealthMarker(t *testing.T) {
	s := &Server{metrics: newGatewayMetrics(time.Now()), logf: func(string, ...any) {}}
	h := s.withMetrics(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		panic("boom")
	}))

	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/v1/fak/syscall", nil))
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("control-plane panic status = %d, want 500", rr.Code)
	}
	if _, _, _, recent := s.servedFailure.recent(time.Now()); recent {
		t.Fatal("control-plane panic marked the served-failure record; only served turns may")
	}

	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/v1/completions", nil))
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("served panic status = %d, want 500", rr.Code)
	}
	route, msg, _, recent := s.servedFailure.recent(time.Now())
	if !recent {
		t.Fatal("served-turn panic did not mark the served-failure record")
	}
	if route != "/v1/completions" || !strings.Contains(msg, "boom") {
		t.Fatalf("served-failure record = (%q, %q), want (/v1/completions, ...boom...)", route, msg)
	}
}
