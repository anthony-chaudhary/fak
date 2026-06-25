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
	gotTrace, gotText := "", ""
	srv.steerSession = func(_ context.Context, traceID, text string) error {
		gotTrace, gotText = traceID, text
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
}

func TestSteerRouteRefusalMapsTo422(t *testing.T) {
	srv := newTestServer(t)
	srv.steerSession = func(_ context.Context, _, _ string) error {
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
	srv.steerSession = func(_ context.Context, _, _ string) error { return nil }
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
