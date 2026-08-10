package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// The end-to-end witness for #2258 — "never sleep past the client". The unit tests around
// upstreamErrorStatus pin the error LADDER from a hand-built *agent.RetryCeilingError; this
// one pins the whole wire, from a real upstream 429 to the bytes the wrapped client reads,
// and is the only place that proves the two things the 2026-07-01 evidence log actually
// showed going wrong:
//
//   - fak must NOT sleep toward a provider-named wait the client cannot outlast. The
//     regression it guards is not a wrong status, it is 50 minutes of wall-clock: the
//     gateway announced a ~1h10m honored wait, slept in-handler, and the wrapped Claude Code
//     client timed out its own request at ~300s, twelve times over, before dying with an
//     opaque "Request timed out" that named none of the rate-limit truth fak already held.
//   - the truth must ride downstream INSTEAD: the real 429, the real Retry-After, and a code
//     that is distinguishable from a bare throttle.
//
// Both halves are asserted here against elapsed time and the upstream hit count, because a
// status assertion alone would still pass if fak slept an hour first.
func TestRetryCeiling_OverCeilingCapSurfacesTruthfulFast(t *testing.T) {
	// Leave FAK_INHANDLER_WAIT_CEILING unset on purpose: the DEFAULT (90s) is the value the
	// flagship proxy path actually runs with, so the default is what needs the witness.
	// A generous attempt budget makes the "exactly one hit" assertion below meaningful —
	// without the ceiling the loop would have plenty of budget left to burn.
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "8")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	// The upstream answers the shape from the evidence log: a cap 429 naming a reset far
	// beyond anything a wrapped client can hold a request open for (4200s = 1h10m).
	const retryAfter = "4200"
	var hits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.Header().Set("Retry-After", retryAfter)
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"usage limit reached"}}`))
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "k", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	inbound := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	elapsed := time.Since(start)

	// THE regression assertion. The announced wait was 4200s and the observed client timeout
	// is ~300s; anything in that neighbourhood means fak went back to sleeping past the
	// client. The bound is deliberately far below the ~300s client timeout AND far above any
	// plausible scheduling hiccup, so it can only fail if a real in-handler sleep returned.
	if elapsed > 30*time.Second {
		t.Fatalf("took %s to answer an over-ceiling cap 429 — fak slept in-handler past what the client can survive (#2258)", elapsed)
	}

	// The client sees the REAL upstream status, never an opaque 502 and never a timeout.
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("client status = %d, want 429 (the true upstream status rides downstream)", resp.StatusCode)
	}
	// The real Retry-After rides downstream as a HEADER, so a Retry-After-honoring harness
	// backs off correctly instead of hammering the same wall.
	if got := resp.Header.Get("Retry-After"); got != retryAfter {
		t.Fatalf("Retry-After header = %q, want %q (the provider's own value, verbatim)", got, retryAfter)
	}

	var env struct {
		Error struct {
			Code    string `json:"code"`
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatalf("error envelope did not decode: %v (body %q)", err, body)
	}
	// A ceiling bail must be distinguishable from a bare throttle — that distinction is what
	// lets a supervisor (#2256) park on it instead of retrying into the same wall.
	if env.Error.Code != "upstream_retry_ceiling" {
		t.Fatalf("error code = %q, want upstream_retry_ceiling (a ceiling stop is not a plain throttle)", env.Error.Code)
	}
	if env.Error.Message == "" {
		t.Fatalf("ceiling bail must carry an actionable message, got an empty one (body %q)", body)
	}

	// The decisive cost assertion: fak stopped at the FIRST retry decision. The evidence log's
	// failure was twelve futile cycles; with an 8-attempt budget, any hit count above one means
	// the loop is still burning attempts against a wall it can already name.
	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Fatalf("upstream hits = %d, want 1 — the ceiling must stop the loop at the first over-ceiling wait, not retry into the same wall", got)
	}
}

// The other half of the DoD: a wait AT OR UNDER the ceiling keeps today's absorb-in-handler
// behavior. That path is the right UX for a transient throttle and #2258 must not have
// touched it, so this pins the boundary from the under side — same upstream, same wire, only
// the size of the announced wait differs. Without this, a future "fix" could satisfy the test
// above by simply never retrying at all.
func TestRetryCeiling_UnderCeilingWaitStillAbsorbsInHandler(t *testing.T) {
	// A ceiling far above the announced wait puts this request unambiguously on the absorb
	// side of the boundary; the 1s Retry-After keeps the test cheap while still exercising a
	// real honored sleep.
	t.Setenv("FAK_INHANDLER_WAIT_CEILING", "1h")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var hits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.Header().Set("Retry-After", "1")
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "k", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	inbound := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("client status = %d, want 429", resp.StatusCode)
	}
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatalf("error envelope did not decode: %v (body %q)", err, body)
	}
	// Under the ceiling nothing about #2258 applies: this is a plain exhausted-retry throttle,
	// and it must still read as one.
	if env.Error.Code != "upstream_rate_limited" {
		t.Fatalf("error code = %q, want upstream_rate_limited (an under-ceiling wait is absorbed, not a ceiling bail)", env.Error.Code)
	}
	// It ABSORBED the wait and spent its full attempt budget, exactly as before #2258.
	if got := atomic.LoadInt32(&hits); got != 2 {
		t.Fatalf("upstream hits = %d, want 2 — an under-ceiling wait must still be slept and retried in-handler", got)
	}
}
