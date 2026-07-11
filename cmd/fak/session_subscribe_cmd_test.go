package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// session_subscribe_cmd_test.go — the CLI half of the #2767 re-attach op: `fak
// session subscribe` dials the subscribe route with the operator's cursor and
// renders the resumed tail plus the next cursor to re-attach with. The cursor
// semantics themselves are proven gateway-side
// (internal/gateway/session_subscribe_test.go); this pins the CLI wiring — the
// path, the --since round-trip, both render modes, and the usage edges.
func TestSessionSubscribeCLIReattachRoundTrip(t *testing.T) {
	var gotPath, gotSince string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotSince = r.URL.Path, r.URL.Query().Get("since")
		json.NewEncoder(w).Encode(gateway.SessionSubscribeResponse{
			TraceID: "gw-1",
			Events: []gateway.SessionChangeEvent{
				{Seq: 4, SessionState: gateway.SessionState{TraceID: "gw-1", Run: "running", Rev: 3}},
				{Seq: 6, SessionState: gateway.SessionState{TraceID: "gw-1", Run: "paused", Rev: 4}},
			},
			Cursor:   6,
			Complete: true,
		})
	}))
	defer ts.Close()

	// Re-attach with a saved cursor, human render.
	var stdout, stderr bytes.Buffer
	if rc := runSession(&stdout, &stderr, []string{"subscribe", "gw-1", "--addr", ts.URL, "--since", "3"}); rc != 0 {
		t.Fatalf("exit = %d, stderr %s", rc, stderr.String())
	}
	if gotPath != "/v1/fak/session/gw-1/subscribe" || gotSince != "3" {
		t.Fatalf("dialed %s?since=%s; want /v1/fak/session/gw-1/subscribe?since=3", gotPath, gotSince)
	}
	out := stdout.String()
	for _, want := range []string{"gw-1", "cursor=6", "complete=true", "rev=3", "rev=4", "--since 6"} {
		if !strings.Contains(out, want) {
			t.Errorf("human render missing %q:\n%s", want, out)
		}
	}

	// --json emits the raw response document.
	stdout.Reset()
	if rc := runSession(&stdout, &stderr, []string{"subscribe", "gw-1", "--addr", ts.URL, "--json"}); rc != 0 {
		t.Fatalf("json exit = %d, stderr %s", rc, stderr.String())
	}
	if gotSince != "" {
		t.Fatalf("bare subscribe sent since=%q; want none", gotSince)
	}
	var doc gateway.SessionSubscribeResponse
	if err := json.Unmarshal(stdout.Bytes(), &doc); err != nil {
		t.Fatalf("--json output not a SessionSubscribeResponse: %v\n%s", err, stdout.String())
	}
	if doc.Cursor != 6 || len(doc.Events) != 2 {
		t.Fatalf("--json doc = %+v; want cursor 6, 2 events", doc)
	}
}

func TestSessionSubscribeCLIUsageEdges(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if rc := runSession(&stdout, &stderr, []string{"subscribe"}); rc != 2 {
		t.Fatalf("missing id exit = %d, want 2", rc)
	}
	stderr.Reset()
	if rc := runSession(&stdout, &stderr, []string{"subscribe", "gw-1", "stray"}); rc != 2 {
		t.Fatalf("stray positional exit = %d, want 2", rc)
	}
}
