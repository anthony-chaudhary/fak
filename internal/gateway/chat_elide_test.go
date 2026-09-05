package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestChatElide_NonStreaming(t *testing.T) {
	srv := newTestServer(t)
	planner := &capturingResponsesPlanner{
		comp: &agent.Completion{
			FinishReason: "stop",
			Message: agent.Message{
				Role:    agent.RoleAssistant,
				Content: "chat analysis finished",
			},
		},
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	bigOutput := strings.Repeat("C", 2000)
	smallOutput := strings.Repeat("D", 500)

	body := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{"role": "user", "content": "turn 0"},
			map[string]any{"role": "tool", "tool_call_id": "c0", "content": bigOutput},
			map[string]any{"role": "user", "content": "turn 1"},
			map[string]any{"role": "tool", "tool_call_id": "c1", "content": smallOutput},
			map[string]any{"role": "user", "content": "turn 2"},
			map[string]any{"role": "tool", "tool_call_id": "c2", "content": bigOutput},
			map[string]any{"role": "user", "content": "turn 3"},
			map[string]any{"role": "tool", "tool_call_id": "c3", "content": bigOutput},
			map[string]any{"role": "user", "content": "turn 4"},
			map[string]any{"role": "tool", "tool_call_id": "c4", "content": bigOutput},
			map[string]any{"role": "user", "content": "turn 5"},
			map[string]any{"role": "tool", "tool_call_id": "c5", "content": bigOutput},
		},
		"stream": false,
	}

	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", "t-chat-elide-nonstream")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	// Verify planner received elided messages
	if len(planner.messages) < 12 {
		t.Fatalf("expected >= 12 messages forwarded, got %d", len(planner.messages))
	}

	// Tool c0 should be elided
	c0Msg := planner.messages[1]
	if !strings.HasPrefix(c0Msg.Content, "...[fak: tool output elided") {
		t.Fatalf("expected c0 to be elided, got: %q", c0Msg.Content)
	}

	// Tool c1 was <= 1024 bytes, should remain intact
	c1Msg := planner.messages[3]
	if c1Msg.Content != smallOutput {
		t.Fatalf("expected c1 to remain intact, got: %q", c1Msg.Content)
	}

	// Recent 4 tool messages (c2..c5) must remain intact
	for i, idx := range []int{5, 7, 9, 11} {
		if planner.messages[idx].Content != bigOutput {
			t.Fatalf("recent tool %d (msg index %d) was elided", i, idx)
		}
	}

	// Extract digest from marker and verify recovery via restoreContext
	re := regexp.MustCompile(`id=(sha256:[0-9a-f]{64})`)
	match := re.FindStringSubmatch(c0Msg.Content)
	if len(match) < 2 {
		t.Fatalf("failed to extract id from marker: %s", c0Msg.Content)
	}
	markerID := match[1]

	res, err := srv.restoreContext("", ContextRestoreRequest{ID: markerID, TraceID: "t-chat-elide-nonstream"})
	if err != nil {
		t.Fatalf("restoreContext failed: %v", err)
	}
	if res.Bytes != bigOutput {
		t.Fatalf("restored bytes mismatch: got %d, want %d", len(res.Bytes), len(bigOutput))
	}
}

func TestChatElide_Streaming(t *testing.T) {
	srv := newTestServer(t)
	planner := &capturingResponsesPlanner{
		comp: &agent.Completion{
			FinishReason: "stop",
			Message: agent.Message{
				Role:    agent.RoleAssistant,
				Content: "stream finished",
			},
		},
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	bigOutput := strings.Repeat("S", 2000)
	body := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{"role": "user", "content": "turn 0"},
			map[string]any{"role": "tool", "tool_call_id": "c0", "content": bigOutput},
			map[string]any{"role": "tool", "tool_call_id": "c1", "content": "r1"},
			map[string]any{"role": "tool", "tool_call_id": "c2", "content": "r2"},
			map[string]any{"role": "tool", "tool_call_id": "c3", "content": "r3"},
			map[string]any{"role": "tool", "tool_call_id": "c4", "content": "r4"},
		},
		"stream": true,
	}

	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", "t-chat-elide-stream")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	_, _ = io.ReadAll(resp.Body)

	if len(planner.messages) < 6 {
		t.Fatalf("expected >= 6 messages forwarded, got %d", len(planner.messages))
	}

	c0Msg := planner.messages[1]
	if !strings.HasPrefix(c0Msg.Content, "...[fak: tool output elided") {
		t.Fatalf("expected c0 to be elided in streaming mode, got: %q", c0Msg.Content)
	}
}

func TestChatElide_RestoreToolAutoAdvertiseVariants(t *testing.T) {
	srv := newTestServer(t)
	planner := &capturingResponsesPlanner{
		comp: &agent.Completion{
			FinishReason: "stop",
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		},
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	t.Run("mcp_canon_prefix_advertises_mcp__fak_guard__fak_context_restore", func(t *testing.T) {
		body := map[string]any{
			"model": "test-model",
			"messages": []any{
				map[string]any{"role": "user", "content": "hello"},
			},
			"tools": []any{
				map[string]any{
					"type": "function",
					"function": map[string]any{
						"name":        "mcp__fak_guard__fak_read",
						"description": "read file",
					},
				},
			},
		}
		raw, _ := json.Marshal(body)
		req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(raw))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Trace-Id", "t-chat-canon")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		var found bool
		for _, tool := range planner.tools {
			if tool.Function.Name == "mcp__fak_guard__fak_context_restore" {
				found = true
				break
			}
		}
		if !found {
			t.Fatal("expected mcp__fak_guard__fak_context_restore to be auto-advertised")
		}
	})

	t.Run("mcp_legacy_prefix_advertises_mcp__fak__fak_context_restore", func(t *testing.T) {
		body := map[string]any{
			"model": "test-model",
			"messages": []any{
				map[string]any{"role": "user", "content": "hello"},
			},
			"tools": []any{
				map[string]any{
					"type": "function",
					"function": map[string]any{
						"name":        "mcp__fak__fak_read",
						"description": "read file",
					},
				},
			},
		}
		raw, _ := json.Marshal(body)
		req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(raw))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Trace-Id", "t-chat-legacy")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		var found bool
		for _, tool := range planner.tools {
			if tool.Function.Name == "mcp__fak__fak_context_restore" {
				found = true
				break
			}
		}
		if !found {
			t.Fatal("expected mcp__fak__fak_context_restore to be auto-advertised")
		}
	})

	t.Run("no_tools_provided_does_not_auto-advertise_tools", func(t *testing.T) {
		planner.tools = nil
		body := map[string]any{
			"model": "test-model",
			"messages": []any{
				map[string]any{"role": "user", "content": "plain conversational text"},
			},
		}
		raw, _ := json.Marshal(body)
		req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(raw))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Trace-Id", "t-chat-notools")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if len(planner.tools) != 0 {
			t.Fatalf("expected 0 tools when client supplies none, got %d", len(planner.tools))
		}
	})
}
