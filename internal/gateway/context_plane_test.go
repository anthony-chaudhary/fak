package gateway

// context_plane_test.go — the regression lock for the 2026-11-10 hosting-simplification
// flip of the context-plane postures. The serving router (`fak serve`) resolves
// Config.CtxPaging = off and Config.ChatToolResultElide = off by default, so the chat
// and responses wires forward the client's declared tools and tool results
// byte-identically: no oversize {"_paged":true} restore stubs, no
// "...[fak: tool output elided ...]" rewrite, no restore-tool auto-advertisement. The
// library default (empty Config) stays ARMED so `fak guard`, the demos and every
// existing caller keep the savers byte-for-byte. Each lever re-arms by name.

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestParseCtxPagingModeSpelling(t *testing.T) {
	cases := []struct {
		raw  string
		want string
		ok   bool
	}{
		{"", CtxPagingAuto, true},
		{"auto", CtxPagingAuto, true},
		{" on ", CtxPagingOn, true},
		{"1", CtxPagingOn, true},
		{"OFF", CtxPagingOff, true},
		{"raw", CtxPagingOff, true},
		{"maybe", "", false},
	}
	for _, tc := range cases {
		got, err := ParseCtxPagingMode(tc.raw)
		if tc.ok && (err != nil || got != tc.want) {
			t.Errorf("ParseCtxPagingMode(%q) = (%q,%v), want (%q,nil)", tc.raw, got, err, tc.want)
		}
		if !tc.ok && err == nil {
			t.Errorf("ParseCtxPagingMode(%q) must fail loud, got %q", tc.raw, got)
		}
	}
}

func TestParseChatToolResultElideModeSpelling(t *testing.T) {
	cases := []struct {
		raw  string
		want string
		ok   bool
	}{
		{"", ChatToolResultElideOn, true},
		{"on", ChatToolResultElideOn, true},
		{"TRUE", ChatToolResultElideOn, true},
		{"off", ChatToolResultElideOff, true},
		{"disabled", ChatToolResultElideOff, true},
		{"perhaps", "", false},
	}
	for _, tc := range cases {
		got, err := ParseChatToolResultElideMode(tc.raw)
		if tc.ok && (err != nil || got != tc.want) {
			t.Errorf("ParseChatToolResultElideMode(%q) = (%q,%v), want (%q,nil)", tc.raw, got, err, tc.want)
		}
		if !tc.ok && err == nil {
			t.Errorf("ParseChatToolResultElideMode(%q) must fail loud, got %q", tc.raw, got)
		}
	}
}

// TestServePostureConstsAreRaw pins the serving-router posture constants: the
// hosting-simplification flip values. A peer who silently re-arms either default reds
// here and must record the reversal deliberately.
func TestServePostureConstsAreRaw(t *testing.T) {
	if DefaultServeCtxPaging != CtxPagingOff {
		t.Fatalf("DefaultServeCtxPaging must be off (the raw serving-router posture), got %q", DefaultServeCtxPaging)
	}
	if DefaultServeChatToolResultElide != ChatToolResultElideOff {
		t.Fatalf("DefaultServeChatToolResultElide must be off (the raw serving-router posture), got %q", DefaultServeChatToolResultElide)
	}
}

// TestUnparsableContextPlanePosturesFailLoud: a context rewriter is a model-visible
// boundary; an explicit unparsable posture must refuse boot, never silently resolve.
func TestUnparsableContextPlanePosturesFailLoud(t *testing.T) {
	if _, err := New(Config{EngineID: "mock", Model: "m", CtxPaging: "perhaps"}); err == nil {
		t.Fatal("unparsable CtxPaging must fail New loud")
	}
	if _, err := New(Config{EngineID: "mock", Model: "m", ChatToolResultElide: "perhaps"}); err == nil {
		t.Fatal("unparsable ChatToolResultElide must fail New loud")
	}
}

// TestChatCtxPagingPosturePrecedence pins the resolution order on the chat wire:
// per-request header > FAK_CTX_PAGING env > the server's resolved Config.CtxPaging
// posture > auto. The serve door passes off; the guard door leaves the config empty
// (auto) and both keep their historical behavior.
func TestChatCtxPagingPosturePrecedence(t *testing.T) {
	newReq := func(header string) *http.Request {
		r, _ := http.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		if header != "" {
			r.Header.Set(HeaderCtxPaging, header)
		}
		return r
	}
	tools := []agent.ToolDef{{Type: "function", Function: agent.ToolDefFunction{Name: "mcp__fak__fak_read", Description: "read file"}}}

	// Empty config (the library/guard posture): env unset → auto (no suppression).
	srv := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model"})
	if got := srv.resolveChatCtxPaging(newReq(""), tools); got.suppressReason != "" {
		t.Fatalf("empty config with no header/env must stay auto (pageable), got suppression %q", got.suppressReason)
	}
	// Empty config: the operator env still wins.
	t.Setenv(envCtxPaging, "off")
	if got := srv.resolveChatCtxPaging(newReq(""), tools); got.suppressReason == "" {
		t.Fatal("env off must suppress paging over the empty-config auto posture")
	}
	// The per-request header wins over the env.
	if got := srv.resolveChatCtxPaging(newReq("on"), tools); got.suppressReason != "" {
		t.Fatalf("header on must win over env off, got suppression %q", got.suppressReason)
	}

	// Serve posture (off): env unset → suppressed operator_off.
	serve := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model", CtxPaging: CtxPagingOff})
	osUnset := func() { t.Setenv(envCtxPaging, "") }
	osUnset()
	if got := serve.resolveChatCtxPaging(newReq(""), tools); got.suppressReason == "" {
		t.Fatal("serve posture off must suppress paging when no header/env overrides")
	}
	// The env still wins over the serve posture.
	t.Setenv(envCtxPaging, "on")
	if got := serve.resolveChatCtxPaging(newReq(""), tools); got.suppressReason != "" {
		t.Fatal("env on must win over the serve off posture")
	}
	// And the header wins over both.
	if got := serve.resolveChatCtxPaging(newReq("off"), tools); got.suppressReason == "" {
		t.Fatal("header off must win over env on")
	}
}

// TestChatElideOffPostureKeepsWireByteIdentical is the live-path witness for the raw
// context plane: with the serve posture (elide off, paging off) a chat request
// carrying a fak-MCP client tool and an oversized old tool result forwards the
// client's declared tools and the tool result BYTE-IDENTICALLY — no elision marker,
// no injected restore tool, no paged stub.
func TestChatElideOffPostureKeepsWireByteIdentical(t *testing.T) {
	t.Setenv("FAK_RESPONSES_ELIDE_THRESHOLD", "256") // would elide at the armed posture
	serve := newTestServerWithConfig(t, Config{
		EngineID:            "test",
		Model:               "test-model",
		CtxPaging:           CtxPagingOff,
		ChatToolResultElide: ChatToolResultElideOff,
	})
	planner := &capturingResponsesPlanner{
		comp: &agent.Completion{
			FinishReason: "stop",
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		},
	}
	serve.planner = planner

	ts := httptest.NewServer(serve.Handler())
	defer ts.Close()

	bigOutput := strings.Repeat("C", 2000)
	body := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{"role": "user", "content": "turn 0"},
			map[string]any{"role": "tool", "tool_call_id": "c0", "content": bigOutput},
			map[string]any{"role": "user", "content": "turn 1"},
			map[string]any{"role": "tool", "tool_call_id": "c1", "content": "r1"},
			map[string]any{"role": "tool", "tool_call_id": "c2", "content": "r2"},
			map[string]any{"role": "tool", "tool_call_id": "c3", "content": "r3"},
			map[string]any{"role": "tool", "tool_call_id": "c4", "content": "r4"},
			map[string]any{"role": "user", "content": "turn 2"},
		},
		"tools": []any{
			map[string]any{"type": "function", "function": map[string]any{"name": "mcp__fak__fak_read", "description": "read file"}},
		},
	}
	raw, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", "t-ctx-plane-raw")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	// The oversized old tool result reached the model whole — no elision rewrite.
	if len(planner.messages) < 8 {
		t.Fatalf("expected >= 8 messages forwarded, got %d", len(planner.messages))
	}
	if got := planner.messages[1].Content; got != bigOutput {
		t.Fatalf("raw context plane must forward the oversized tool result byte-identically, got %d bytes with prefix %q", len(got), truncate(got, 64))
	}
	// The client's declared tools passed through untouched — no injected restore tool.
	for _, tool := range planner.tools {
		if strings.Contains(tool.Function.Name, "fak_context_restore") {
			t.Fatalf("raw context plane must not auto-advertise the restore tool, found %q", tool.Function.Name)
		}
	}
	// The paging decision is off.
	if h := resp.Header.Get(HeaderCtxPaging); !strings.HasPrefix(h, "off;") {
		t.Fatalf("raw context plane paging header = %q, want off;...", h)
	}
}

// TestChatElideOffPostureStillRecoversCarriedMarkers keeps the defensive recovery
// path: when the request history ALREADY carries an elided marker (left by an
// earlier-armed posture or by the client itself), the restore tool is still
// advertised so the model can page the stashed bytes back in.
func TestChatElideOffPostureStillRecoversCarriedMarkers(t *testing.T) {
	serve := newTestServerWithConfig(t, Config{
		EngineID:            "test",
		Model:               "test-model",
		CtxPaging:           CtxPagingOff,
		ChatToolResultElide: ChatToolResultElideOff,
	})
	planner := &capturingResponsesPlanner{
		comp: &agent.Completion{
			FinishReason: "stop",
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		},
	}
	serve.planner = planner

	ts := httptest.NewServer(serve.Handler())
	defer ts.Close()

	carried := "...[fak: tool output elided (len=48000 bytes); recover original via fak_context_restore id=sha256:" + strings.Repeat("a", 64) + "]..."
	body := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{"role": "user", "content": "turn 0"},
			map[string]any{"role": "tool", "tool_call_id": "c0", "content": carried},
			map[string]any{"role": "user", "content": "turn 1"},
		},
		"tools": []any{
			map[string]any{"type": "function", "function": map[string]any{"name": "mcp__fak__fak_read", "description": "read file"}},
		},
	}
	raw, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", "t-ctx-plane-carried")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	found := false
	for _, tool := range planner.tools {
		if strings.Contains(tool.Function.Name, "fak_context_restore") {
			found = true
		}
	}
	if !found {
		t.Fatal("a history already carrying an elided marker must still get the restore tool advertised (defensive recovery)")
	}
}

// TestRestoreAffordanceArmed pins the affordance gate: armed when the elision
// posture is on (the library default) or when the request's paging decision pages;
// unarmed only when BOTH postures are off (the serving-router default).
func TestRestoreAffordanceArmed(t *testing.T) {
	armed := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model"})
	if !armed.restoreAffordanceArmed(ctxPagingDecision{}) {
		t.Fatal("library default (elide armed) must report the affordance armed")
	}
	raw := newTestServerWithConfig(t, Config{
		EngineID:            "test",
		Model:               "test-model",
		CtxPaging:           CtxPagingOff,
		ChatToolResultElide: ChatToolResultElideOff,
	})
	if raw.restoreAffordanceArmed(ctxPagingDecision{suppressReason: "operator_off"}) {
		t.Fatal("raw context plane with a suppressed paging decision must report the affordance unarmed")
	}
	if !raw.restoreAffordanceArmed(ctxPagingDecision{}) {
		t.Fatal("raw context plane with a live paging decision (header/env on) must report the affordance armed")
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
