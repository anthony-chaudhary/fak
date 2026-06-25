package gateway

// session_admit_test.go — the PROXY-path enforcement of session control: a paused /
// draining / stopped session's next /v1/chat/completions request is refused with 409
// (operator "cancel a request in flight"), while running / unknown / no-route-wired
// fail OPEN (the request proceeds, byte-for-byte the pre-control path).

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func chatPostWithTrace(t *testing.T, url, trace string) *http.Response {
	t.Helper()
	body, _ := json.Marshal(map[string]any{
		"model":    "test-model",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	req, err := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if trace != "" {
		req.Header.Set("X-Trace-Id", trace)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestSessionAdmitRefusesHeldSessionsOnProxy(t *testing.T) {
	srv := newTestServer(t)
	// The operator-set DRIVE state for this trace; the test flips it per case.
	var run, reason string
	srv.observeSession = func(_ context.Context, trace string) SessionState {
		return SessionState{TraceID: trace, Run: run, Reason: reason}
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Non-advancing states refuse the next request with 409 + a session_<state> code.
	for _, st := range []string{"paused", "draining", "stopped"} {
		run, reason = st, "operator-"+st
		resp := chatPostWithTrace(t, ts.URL, "held-"+st)
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusConflict {
			t.Fatalf("run=%s: status = %d, want 409; body=%s", st, resp.StatusCode, raw)
		}
		if !strings.Contains(string(raw), "session_"+st) || !strings.Contains(string(raw), "operator-"+st) {
			t.Fatalf("run=%s: refusal body missing code/reason: %s", st, raw)
		}
	}
}

func TestSessionAdmitFailsOpenWhenRunningOrUnwired(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// (1) No observeSession wired ⇒ fail OPEN: the guard must not produce a 409.
	resp := chatPostWithTrace(t, ts.URL, "trace-unwired")
	resp.Body.Close()
	if resp.StatusCode == http.StatusConflict {
		t.Fatalf("unwired session route must not refuse; got 409")
	}

	// (2) An advancing state (running) is admitted — the guard does not 409 it.
	srv.observeSession = func(_ context.Context, trace string) SessionState {
		return SessionState{TraceID: trace, Run: "running"}
	}
	resp = chatPostWithTrace(t, ts.URL, "trace-running")
	resp.Body.Close()
	if resp.StatusCode == http.StatusConflict {
		t.Fatalf("running session must be admitted; got 409")
	}

	// (3) throttled is admitted too (pace shapes fak's own loop, not proxy admission).
	srv.observeSession = func(_ context.Context, trace string) SessionState {
		return SessionState{TraceID: trace, Run: "throttled"}
	}
	resp = chatPostWithTrace(t, ts.URL, "trace-throttled")
	resp.Body.Close()
	if resp.StatusCode == http.StatusConflict {
		t.Fatalf("throttled session must be admitted; got 409")
	}
}
