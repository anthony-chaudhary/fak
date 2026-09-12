package gateway

// first_token_watchdog_test.go — the buffered first-token watchdog. A planner that never
// yields a first token (a wedged/slow-prefill model) must fail loud with the typed
// UpstreamStalledError{first-token} mapped to the 504 upstream_stalled surface, well before
// the planner's whole-request timeout, rather than hanging the buffered path silently.

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// firstTokenWatchdogBudget bounds how long the test waits for the watchdog to fire — far
// shorter than the planner timeout the parked planner would otherwise hold the request for.
const firstTokenWatchdogBudget = 3 * time.Second

// TestFirstTokenWatchdogBufferedStallIsTyped504 parks the planner inside the modeled
// collective (never released) and asserts a stream:false chat request returns a typed
// 504 upstream_stalled within the short watchdog window.
func TestFirstTokenWatchdogBufferedStallIsTyped504(t *testing.T) {
	planner := newCollectivePlanner("test-model", "never delivered", nil)
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(planner.release) }) }

	srv := newTestServer(t)
	srv.planner = planner
	srv.firstTokenWatchdog = func() time.Duration { return 150 * time.Millisecond }
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	// LIFO: free the parked decode before Close waits on the live connection.
	t.Cleanup(release)

	body, err := json.Marshal(map[string]any{
		"model":    "test-model",
		"messages": []map[string]string{{"role": "user", "content": "wedge the prefill"}},
		"stream":   false,
	})
	if err != nil {
		t.Fatal(err)
	}

	done := make(chan *http.Response, 1)
	go func() {
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
		if err != nil {
			done <- nil
			return
		}
		done <- resp
	}()

	var resp *http.Response
	select {
	case resp = <-done:
		if resp == nil {
			t.Fatal("non-streaming request failed at the transport")
		}
	case <-time.After(firstTokenWatchdogBudget):
		t.Fatal("non-streaming request hung past the first-token watchdog budget")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("status = %d, want 504", resp.StatusCode)
	}
	raw, _ := io.ReadAll(resp.Body)
	var env struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatalf("decode error envelope %q: %v", raw, err)
	}
	if env.Error.Code != "upstream_stalled" {
		t.Fatalf("error.code = %q, want upstream_stalled (body: %s)", env.Error.Code, raw)
	}
	if !containsFirstToken(env.Error.Message) {
		t.Fatalf("error.message = %q, want it to name the first-token watchdog", env.Error.Message)
	}
}

// TestFirstTokenWatchdogFastTurnUnaffected proves the watchdog is default-conservative: a
// planner returning a normal completion still answers 200.
func TestFirstTokenWatchdogFastTurnUnaffected(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = &recordingPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "fast"},
		FinishReason: "stop",
		Model:        "test-model",
		Usage:        agent.Usage{PromptTokens: 3, CompletionTokens: 1, TotalTokens: 4},
	}}
	srv.firstTokenWatchdog = func() time.Duration { return 150 * time.Millisecond }
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	body, err := json.Marshal(map[string]any{
		"model":    "test-model",
		"messages": []map[string]string{{"role": "user", "content": "hello"}},
		"stream":   false,
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200 (body: %s)", resp.StatusCode, raw)
	}
}

func containsFirstToken(msg string) bool {
	return bytes.Contains([]byte(msg), []byte("first-token watchdog"))
}

// TestFirstTokenWatchdogDefaultWindowIsConservative exercises the production default branch:
// with no test seam the window comes from agent.FirstTokenWatchdogTimeout (60s default), so a
// parked planner must NOT be failed within a short budget. This pins that the watchdog is
// default-conservative — it converts a hang, not a merely-slow turn, and the no-seam branch is
// reachable. The planner is released immediately after the budget check so the request drains.
func TestFirstTokenWatchdogDefaultWindowIsConservative(t *testing.T) {
	planner := newCollectivePlanner("test-model", "late", nil)
	srv := newTestServer(t)
	srv.planner = planner
	// No srv.firstTokenWatchdog seam: the real agent default applies (>= 60s).
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	t.Cleanup(func() { close(planner.release) })

	body, err := json.Marshal(map[string]any{
		"model":    "test-model",
		"messages": []map[string]string{{"role": "user", "content": "hold"}},
		"stream":   false,
	})
	if err != nil {
		t.Fatal(err)
	}

	done := make(chan struct{})
	go func() {
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
		if err == nil {
			resp.Body.Close()
		}
		close(done)
	}()

	select {
	case <-done:
		t.Fatal("default-window watchdog fired within a short budget; it must be default-conservative")
	case <-time.After(750 * time.Millisecond):
		// Expected: the 60s default has not elapsed, so the request is still in flight.
	}
}
