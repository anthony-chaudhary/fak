package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// cache_break_live_test.go — the LIVE-producer arm of the mid-conversation cache-break
// detector (#2847, Track C of epic #2834). The pure detector shipped in
// internal/metrics/cache_break_detector.go; this is the gateway call site its own
// closing commit fenced out. It is the acceptance gate the issue names:
//
//	go test ./internal/gateway/... -run TestCacheBreakDetector
//
// The detector hashes each turn's stable prefix (system prompt + tool schema + the
// already-sent history head) and, when it diverges from the session's established
// prefix, witnesses "cache broken here, +N tokens" and folds the priced event into
// #2916's per-session counter sink (recordCacheBreak). A normal appended turn is NOT a
// break — the false-positive guard. All turns in one session share an X-Trace-Id.

// cacheBreakBody builds a minimal valid Anthropic /v1/messages body whose stable prefix
// is parameterizable: a system prompt, a tools[] list, and a history of text turns.
func cacheBreakBody(system string, tools []string, history []map[string]string) []byte {
	msgs := make([]map[string]any, 0, len(history))
	for _, h := range history {
		msgs = append(msgs, map[string]any{"role": h["role"], "content": h["content"]})
	}
	var toolDefs []map[string]any
	for _, name := range tools {
		toolDefs = append(toolDefs, map[string]any{
			"name":         name,
			"description":  name + " tool",
			"input_schema": map[string]any{"type": "object"},
		})
	}
	body := map[string]any{"model": "claude-opus-4-8", "max_tokens": 128, "system": system, "messages": msgs}
	if len(toolDefs) > 0 {
		body["tools"] = toolDefs
	}
	raw, _ := json.Marshal(body)
	return raw
}

// postCacheBreakTurn posts one body under a fixed trace id so every call in a test is
// the SAME session (traceFor mints a fresh trace otherwise, which would re-baseline
// every turn and make a detector test vacuous).
func postCacheBreakTurn(t *testing.T, url, trace string, body []byte) int {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url+"/v1/messages", strings.NewReader(string(body)))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", trace)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	return resp.StatusCode
}

// cacheBreakEventCount reads the per-cause incurred count out of the gateway's live
// cache-break report (#2916), the sink the detector folds into via recordCacheBreak.
func cacheBreakEventCount(s *Server, cause metrics.CacheBreakCause) int {
	for _, c := range s.metrics.cacheBreakReport().ByCause {
		if c.Cause == cause {
			return c.Events
		}
	}
	return 0
}

// TestCacheBreakDetectorWitnessesMidConversationMutation is the reproduction: a session
// whose tool set changes mid-conversation must be detected by prefix-hash divergence and
// counted, while a pure history append must not be. The first turn arms the baseline.
func TestCacheBreakDetectorWitnessesMidConversationMutation(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{
		EngineID:   "test",
		Model:      "test-model",
		VDSO:       true,
		CacheBreak: "warn",
	})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	const trace = "cache-break-live-mutation"

	history := []map[string]string{
		{"role": "user", "content": "read the file"},
		{"role": "assistant", "content": "which file?"},
	}
	if code := postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("you are fak", []string{"read", "write"}, history)); code != 200 {
		t.Fatalf("turn 1 status = %d", code)
	}
	if got := cacheBreakEventCount(srv, metrics.CacheBreakToolsetChange); got != 0 {
		t.Fatalf("baseline turn booked %d toolset-change breaks, want 0", got)
	}

	// A pure history APPEND must NOT be a break (the false-positive guard).
	appended := append(append([]map[string]string(nil), history...),
		map[string]string{"role": "user", "content": "the go file"})
	if code := postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("you are fak", []string{"read", "write"}, appended)); code != 200 {
		t.Fatalf("turn 2 status = %d", code)
	}
	if got := cacheBreakEventCount(srv, metrics.CacheBreakToolsetChange); got != 0 {
		t.Fatalf("appended turn falsely booked %d toolset-change breaks", got)
	}

	// The tool set changed mid-conversation — a real cache break, witnessed and priced.
	if code := postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("you are fak", []string{"read", "write", "bash"}, appended)); code != 200 {
		t.Fatalf("turn 3 status = %d", code)
	}
	if got := cacheBreakEventCount(srv, metrics.CacheBreakToolsetChange); got != 1 {
		t.Fatalf("mid-conversation toolset change not witnessed: got %d, want 1", got)
	}
	if rep := srv.metrics.cacheBreakReport(); rep.CostTokens <= 0 {
		t.Fatalf("cache break was not priced: %+v", rep)
	}
}

// TestCacheBreakDetectorCleanPassOnUnmutatedTurn is the contract's clean-pass witness:
// two byte-identical turns book nothing; a rebuilt system prompt on the third books a
// rebuilt_prompt break.
func TestCacheBreakDetectorCleanPassOnUnmutatedTurn(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{
		EngineID:   "test",
		Model:      "test-model",
		VDSO:       true,
		CacheBreak: "warn",
	})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	const trace = "cache-break-live-clean"

	body := cacheBreakBody("you are fak", []string{"read"}, nil)
	for i := 0; i < 2; i++ {
		if code := postCacheBreakTurn(t, ts.URL, trace, body); code != 200 {
			t.Fatalf("identical turn %d status = %d", i+1, code)
		}
	}
	if rep := srv.metrics.cacheBreakReport(); rep.Events != 0 || rep.CostTokens != 0 {
		t.Fatalf("clean pass booked breaks: %+v", rep)
	}

	rebuilt := cacheBreakBody("you are something else", []string{"read"}, nil)
	if code := postCacheBreakTurn(t, ts.URL, trace, rebuilt); code != 200 {
		t.Fatalf("rebuilt turn status = %d", code)
	}
	if got := cacheBreakEventCount(srv, metrics.CacheBreakRebuiltPrompt); got != 1 {
		t.Fatalf("rebuilt system prompt not witnessed: got %d, want 1", got)
	}
}

// TestCacheBreakDetectorOffIsDisarmed proves the lever is default-OFF and inert: an
// unarmed server running the same mutating sequence books nothing.
func TestCacheBreakDetectorOffIsDisarmed(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model", VDSO: true})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	const trace = "cache-break-live-off"

	postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("you are fak", []string{"read"}, nil))
	postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("rebuilt", []string{"read", "bash"}, nil))

	if rep := srv.metrics.cacheBreakReport(); rep.Events != 0 {
		t.Fatalf("disarmed lever booked events: %+v", rep)
	}
}

// TestCacheBreakDetectorDenyModeRefusesAndKeepsBaseline proves deny is structurally
// distinct from warn: a mid-conversation rebuild is refused on the wire, booked as
// AVOIDED (never incurred), and the established prefix survives so returning to it is
// clean.
func TestCacheBreakDetectorDenyModeRefusesAndKeepsBaseline(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{
		EngineID:   "test",
		Model:      "test-model",
		VDSO:       true,
		CacheBreak: "deny",
	})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	const trace = "cache-break-live-deny"

	if code := postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("you are fak", []string{"read"}, nil)); code != 200 {
		t.Fatalf("baseline turn status = %d", code)
	}
	// The mutation is refused, never forwarded upstream.
	if code := postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("rebuilt", []string{"read"}, nil)); code != http.StatusConflict {
		t.Fatalf("deny did not refuse the mutation: status = %d, want %d", code, http.StatusConflict)
	}
	if got := cacheBreakEventCount(srv, metrics.CacheBreakRebuiltPrompt); got != 0 {
		t.Fatalf("denied break was booked as incurred: got %d", got)
	}
	// Baseline kept: returning to the established prefix is clean again.
	if code := postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("you are fak", []string{"read"}, nil)); code != 200 {
		t.Fatalf("return-to-baseline after deny status = %d", code)
	}
	if rep := srv.metrics.cacheBreakReport(); rep.Events != 0 {
		t.Fatalf("deny mode booked incurred cost: %+v", rep)
	}
}

// TestCacheBreakDetectorWitnessesHistoryRewrite is the third mutation class the issue
// names: an already-sent turn rewritten mid-conversation (a downstream harness mutating its
// own history) must be attributed altered_turn, with system+tools left warm.
func TestCacheBreakDetectorWitnessesHistoryRewrite(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{
		EngineID:   "test",
		Model:      "test-model",
		VDSO:       true,
		CacheBreak: "warn",
	})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	const trace = "cache-break-live-rewrite"

	history := []map[string]string{
		{"role": "user", "content": "read the file"},
		{"role": "assistant", "content": "which file?"},
	}
	if code := postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("you are fak", []string{"read"}, history)); code != 200 {
		t.Fatalf("baseline turn status = %d", code)
	}

	// Rewrite the ALREADY-SENT user turn (same system, same tools): altered_turn.
	rewritten := []map[string]string{
		{"role": "user", "content": "read the OTHER file"},
		{"role": "assistant", "content": "which file?"},
	}
	if code := postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("you are fak", []string{"read"}, rewritten)); code != 200 {
		t.Fatalf("rewrite turn status = %d", code)
	}
	if got := cacheBreakEventCount(srv, metrics.CacheBreakAlteredTurn); got != 1 {
		t.Fatalf("history rewrite not witnessed as altered_turn: got %d, want 1", got)
	}
	// System+tools stayed warm, so no rebuilt_prompt/toolset_change was booked.
	if got := cacheBreakEventCount(srv, metrics.CacheBreakRebuiltPrompt); got != 0 {
		t.Fatalf("history rewrite misattributed as rebuilt_prompt: %d", got)
	}
}

// TestCacheBreakDetectorWitnessOnLiveMetrics renders the witness on the same Prometheus
// surface #2916 lowered: a witnessed break names its cause and token cost.
func TestCacheBreakDetectorWitnessOnLiveMetrics(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{
		EngineID:   "test",
		Model:      "test-model",
		VDSO:       true,
		CacheBreak: "warn",
	})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	const trace = "cache-break-live-metrics"

	postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("you are fak", []string{"read"}, nil))
	postCacheBreakTurn(t, ts.URL, trace, cacheBreakBody("you are nothing", []string{"read"}, nil))

	rendered := srv.renderMetrics()
	if !strings.Contains(rendered, `fak_cache_break_events_total{cause="rebuilt_prompt"} 1`) {
		t.Fatalf("live /metrics did not witness the rebuilt_prompt break:\n%s", rendered)
	}
	if !strings.Contains(rendered, `fak_cache_break_cost_tokens_total{cause="rebuilt_prompt"}`) {
		t.Fatalf("live /metrics did not price the break:\n%s", rendered)
	}
}
