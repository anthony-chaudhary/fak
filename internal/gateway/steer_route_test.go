package gateway

// steer_route_test.go - the HTTP contract for POST /v1/fak/session/{id}/steer (#760):
// operator input to a RUNNING session. A clean steer is enqueued (202); a refused one (the
// a2achan floor's deny-as-value, surfaced as a non-nil error) maps to 422; a nil injection
// is fail-closed (404); an empty body is 400. Mirrors session_routes_test.go.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSteerRouteEnqueuesCleanSteer(t *testing.T) {
	srv := newTestServer(t)
	srv.native = true // this serve owns a RunArm loop that drains the steer bus (#3528)
	gotTrace, gotPrincipal, gotText := "", "unset", ""
	srv.steerSession = func(_ context.Context, traceID, principal, text string) error {
		gotTrace, gotPrincipal, gotText = traceID, principal, text
		return nil
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(SteerRequest{Text: "switch to plan B"})
	r, err := http.Post(ts.URL+"/v1/fak/session/sess-7/steer", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusAccepted {
		t.Fatalf("steer status = %d, want 202", r.StatusCode)
	}
	if gotTrace != "sess-7" || gotText != "switch to plan B" {
		t.Fatalf("steer delivered trace=%q text=%q, want sess-7 / 'switch to plan B'", gotTrace, gotText)
	}
	if gotPrincipal != "" {
		t.Fatalf("an operator steer (no principal field) must reach steerSession with an empty principal, got %q", gotPrincipal)
	}
}

// TestSteerRouteThreadsMachinePrincipal proves the #3529 seam: a SteerRequest.Principal is
// carried through to steerSession as the `from` attribution (a machine guard naming itself),
// so a doomloop nudge is not misattributed to the human "operator".
func TestSteerRouteThreadsMachinePrincipal(t *testing.T) {
	srv := newTestServer(t)
	srv.native = true
	gotPrincipal := ""
	srv.steerSession = func(_ context.Context, _, principal, _ string) error {
		gotPrincipal = principal
		return nil
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(SteerRequest{Text: "re-anchor: step back", Principal: "doomloop-guard"})
	r, err := http.Post(ts.URL+"/v1/fak/session/sess-dl/steer", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusAccepted {
		t.Fatalf("machine steer status = %d, want 202", r.StatusCode)
	}
	if gotPrincipal != "doomloop-guard" {
		t.Fatalf("machine principal not threaded to steerSession: got %q, want doomloop-guard", gotPrincipal)
	}
}

func TestSteerRouteRefusalMapsTo422(t *testing.T) {
	srv := newTestServer(t)
	srv.native = true // reach the floor Send: the 422 is the floor's deny, not the owned-loop 409
	srv.steerSession = func(_ context.Context, _, _, _ string) error {
		return errors.New("a2a floor refused (TRUST_VIOLATION)")
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(SteerRequest{Text: "tainted"})
	r, err := http.Post(ts.URL+"/v1/fak/session/sess-8/steer", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("refused steer status = %d, want 422", r.StatusCode)
	}
	raw, _ := io.ReadAll(r.Body)
	if !strings.Contains(string(raw), "refused") {
		t.Fatalf("422 body = %q, want it to mention the refusal", raw)
	}
}

// TestSteerRouteProxyServedRefusesNoOwnedLoop is the honest-refusal keystone (#3528): a
// serve process that owns no agent loop (the default proxy path, native=false) refuses a
// well-formed steer with 409 STEER_NO_OWNED_LOOP and NEVER calls steerSession — no phantom
// enqueue onto a bus nothing drains, and no false 202 "delivered" ack.
func TestSteerRouteProxyServedRefusesNoOwnedLoop(t *testing.T) {
	srv := newTestServer(t) // native defaults to false: proxy serve, no owned loop
	delivered := false
	srv.steerSession = func(_ context.Context, _, _, _ string) error {
		delivered = true
		return nil
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(SteerRequest{Text: "switch to plan B"})
	r, err := http.Post(ts.URL+"/v1/fak/session/sess-proxy/steer", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusConflict {
		t.Fatalf("proxy-served steer status = %d, want 409 (fail-closed, no owned loop)", r.StatusCode)
	}
	if delivered {
		t.Fatal("proxy-served steer must NOT reach steerSession: no owned loop to consume it")
	}
	var env struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	raw, _ := io.ReadAll(r.Body)
	if json.Unmarshal(raw, &env) != nil || env.Error.Code != "steer_no_owned_loop" {
		t.Fatalf("409 body should carry code=steer_no_owned_loop; got %q", raw)
	}
	if !strings.Contains(env.Error.Message, "STEER_NO_OWNED_LOOP") || !strings.Contains(env.Error.Message, "--native") {
		t.Fatalf("409 message should name the reason and the native remedy; got %q", env.Error.Message)
	}
}

// TestSteerRouteEmptyTextIs400EvenOnProxy proves request-shape checks win over the
// owned-loop gate: an empty steer is a 400 whether or not the process owns a loop, so a
// malformed request is never masked by the 409.
func TestSteerRouteEmptyTextIs400EvenOnProxy(t *testing.T) {
	srv := newTestServer(t) // native=false
	srv.steerSession = func(_ context.Context, _, _, _ string) error { return nil }
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(SteerRequest{Text: "   "})
	r, err := http.Post(ts.URL+"/v1/fak/session/sess-11/steer", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusBadRequest {
		t.Fatalf("empty steer on proxy status = %d, want 400 (shape check before owned-loop gate)", r.StatusCode)
	}
}

func TestSteerRouteNilInjectionIs404(t *testing.T) {
	srv := newTestServer(t)
	srv.steerSession = nil // not configured
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(SteerRequest{Text: "x"})
	r, err := http.Post(ts.URL+"/v1/fak/session/sess-9/steer", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusNotFound {
		t.Fatalf("nil steer injection status = %d, want 404 (fail-closed)", r.StatusCode)
	}
}

func TestSteerRouteEmptyTextIs400(t *testing.T) {
	srv := newTestServer(t)
	srv.steerSession = func(_ context.Context, _, _, _ string) error { return nil }
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(SteerRequest{Text: "   "})
	r, err := http.Post(ts.URL+"/v1/fak/session/sess-10/steer", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusBadRequest {
		t.Fatalf("empty steer text status = %d, want 400", r.StatusCode)
	}
}
