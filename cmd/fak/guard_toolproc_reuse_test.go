package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

// fak-test:runtime fast est=100ms
func TestGuardToolprocReuseModes(t *testing.T) {
	tmpDir := t.TempDir()

	t.Run("reads arms on loopback", func(t *testing.T) {
		t.Setenv("FAK_TOOLPROC_REUSE", "")
		srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "claude"})
		if err != nil {
			t.Fatalf("gateway.New: %v", err)
		}
		defer srv.Close()
		defer srv.DisarmToolprocReuse()

		armed, reason := armGuardToolprocReuse(srv, "reads", true, tmpDir)
		if !armed || reason != "reads" {
			t.Fatalf("armGuardToolprocReuse(reads, true) = (%v, %q), want (true, \"reads\")", armed, reason)
		}
	})

	t.Run("off stays unarmed", func(t *testing.T) {
		t.Setenv("FAK_TOOLPROC_REUSE", "")
		srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "claude"})
		if err != nil {
			t.Fatalf("gateway.New: %v", err)
		}
		defer srv.Close()

		armed, reason := armGuardToolprocReuse(srv, "off", true, tmpDir)
		if armed || reason != "disabled" {
			t.Fatalf("armGuardToolprocReuse(off, true) = (%v, %q), want (false, \"disabled\")", armed, reason)
		}
	})

	t.Run("env overrides default reads", func(t *testing.T) {
		t.Setenv("FAK_TOOLPROC_REUSE", "off")
		srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "claude"})
		if err != nil {
			t.Fatalf("gateway.New: %v", err)
		}
		defer srv.Close()

		armed, reason := armGuardToolprocReuse(srv, "reads", true, tmpDir)
		if armed || reason != "disabled" {
			t.Fatalf("env FAK_TOOLPROC_REUSE=off: got (%v, %q), want (false, \"disabled\")", armed, reason)
		}

		t.Setenv("FAK_TOOLPROC_REUSE", "all")
		defer srv.DisarmToolprocReuse()
		armed, reason = armGuardToolprocReuse(srv, "reads", true, tmpDir)
		if !armed || reason != "all" {
			t.Fatalf("env FAK_TOOLPROC_REUSE=all: got (%v, %q), want (true, \"all\")", armed, reason)
		}
	})

	t.Run("non-loopback refuses", func(t *testing.T) {
		t.Setenv("FAK_TOOLPROC_REUSE", "")
		srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "claude"})
		if err != nil {
			t.Fatalf("gateway.New: %v", err)
		}
		defer srv.Close()

		armed, reason := armGuardToolprocReuse(srv, "reads", false, tmpDir)
		if armed || reason != "non_loopback_or_no_workdir" {
			t.Fatalf("armGuardToolprocReuse(reads, false) = (%v, %q), want (false, \"non_loopback_or_no_workdir\")", armed, reason)
		}
	})

	t.Run("empty workdir refuses", func(t *testing.T) {
		t.Setenv("FAK_TOOLPROC_REUSE", "")
		srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "claude"})
		if err != nil {
			t.Fatalf("gateway.New: %v", err)
		}
		defer srv.Close()

		armed, reason := armGuardToolprocReuse(srv, "reads", true, "   ")
		if armed || reason != "non_loopback_or_no_workdir" {
			t.Fatalf("armGuardToolprocReuse(reads, empty) = (%v, %q), want (false, \"non_loopback_or_no_workdir\")", armed, reason)
		}
	})

	t.Run("all arms query coalescing", func(t *testing.T) {
		t.Setenv("FAK_TOOLPROC_REUSE", "")
		srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "claude"})
		if err != nil {
			t.Fatalf("gateway.New: %v", err)
		}
		defer srv.Close()
		defer srv.DisarmToolprocReuse()

		armed, reason := armGuardToolprocReuse(srv, "all", true, tmpDir)
		if !armed || reason != "all" {
			t.Fatalf("armGuardToolprocReuse(all, true) = (%v, %q), want (true, \"all\")", armed, reason)
		}
	})

	t.Run("unknown mode refuses", func(t *testing.T) {
		t.Setenv("FAK_TOOLPROC_REUSE", "")
		srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "claude"})
		if err != nil {
			t.Fatalf("gateway.New: %v", err)
		}
		defer srv.Close()

		armed, reason := armGuardToolprocReuse(srv, "bogus", true, tmpDir)
		if armed || reason != "unknown_mode" {
			t.Fatalf("armGuardToolprocReuse(bogus, true) = (%v, %q), want (false, \"unknown_mode\")", armed, reason)
		}
	})
}

// TestGuardToolprocReuseWireWitness exercises the live HTTP /v1/messages seam:
// deposits an admitted Read result, verifies an identical Read is served inline
// from toolproc without forwarding tool_use, mutates the file on disk, and
// verifies the next Read misses and forwards the tool_use fresh.
// fak-test:runtime fast est=250ms
func TestGuardToolprocReuseWireWitness(t *testing.T) {
	t.Setenv("FAK_TOOLPROC_REUSE", "")
	tmpDir := t.TempDir()
	const fileName = "skill.md"
	filePath := filepath.Join(tmpDir, fileName)
	if err := os.WriteFile(filePath, []byte("# initial skill content\n"), 0o600); err != nil {
		t.Fatalf("write initial file: %v", err)
	}

	turn := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		turn++
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		callID := "call_read_2"
		if turn > 1 {
			callID = "call_read_3"
		}
		resp := map[string]any{
			"id":          "msg_upstream",
			"type":        "message",
			"role":        "assistant",
			"stop_reason": "tool_use",
			"content": []map[string]any{
				{
					"type":  "tool_use",
					"id":    callID,
					"name":  "Read",
					"input": map[string]any{"file_path": fileName},
				},
			},
			"usage": map[string]any{"input_tokens": 10, "output_tokens": 10},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer upstream.Close()

	srv, err := gateway.New(gateway.Config{
		EngineID:  "mock",
		Model:     "claude",
		BaseURL:   upstream.URL,
		Provider:  "anthropic",
		ToolGuard: "on",
		APIKey:    "test-api-key",
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	defer srv.Close()

	armed, reason := armGuardToolprocReuse(srv, "reads", true, tmpDir)
	if !armed || reason != "reads" {
		t.Fatalf("armGuardToolprocReuse = (%v, %q), want (true, \"reads\")", armed, reason)
	}
	defer srv.DisarmToolprocReuse()

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	client := ts.Client()
	client.Timeout = 5 * time.Second

	postMessages := func(payload map[string]any) []byte {
		t.Helper()
		raw, err := json.Marshal(payload)
		if err != nil {
			t.Fatalf("marshal payload: %v", err)
		}
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, ts.URL+"/v1/messages", bytes.NewReader(raw))
		if err != nil {
			t.Fatalf("new request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", "test-api-key")
		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("post /v1/messages: %v", err)
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("read response: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
		}
		return body
	}

	// Turn 1: inbound history carries an admitted Read tool_result; upstream proposes
	// the identical Read. Toolproc reuse must serve it inline and suppress tool_use.
	body1 := postMessages(map[string]any{
		"model":      "claude",
		"max_tokens": 256,
		"messages": []map[string]any{
			{"role": "user", "content": "read skill.md"},
			{
				"role": "assistant",
				"content": []map[string]any{
					{"type": "tool_use", "id": "call_read_1", "name": "Read", "input": map[string]any{"file_path": fileName}},
				},
			},
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "tool_result", "tool_use_id": "call_read_1", "content": `{"content":"# initial skill content"}`},
				},
			},
		},
	})
	if strings.Contains(string(body1), `"type":"tool_use"`) {
		t.Fatalf("expected identical Read to be served inline without tool_use, got: %s", body1)
	}
	if !strings.Contains(string(body1), "initial skill content") {
		t.Fatalf("expected served inline body to contain cached file content, got: %s", body1)
	}

	// Mutate the file on disk so the content digest changes.
	if err := os.WriteFile(filePath, []byte("# mutated skill content\n"), 0o600); err != nil {
		t.Fatalf("mutate file: %v", err)
	}

	// Turn 2: re-propose Read; digest mismatch must miss and forward tool_use fresh.
	body2 := postMessages(map[string]any{
		"model":      "claude",
		"max_tokens": 256,
		"messages": []map[string]any{
			{"role": "user", "content": "read skill.md again"},
		},
	})
	if !strings.Contains(string(body2), `"type":"tool_use"`) || !strings.Contains(string(body2), "call_read_3") {
		t.Fatalf("expected mutated file Read to miss and forward tool_use fresh, got: %s", body2)
	}
}
