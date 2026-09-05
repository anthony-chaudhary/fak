package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestResponsesRestoreToolAutoAdvertise(t *testing.T) {
	t.Run("determine_priority_order", func(t *testing.T) {
		// 1. mcp__fak_guard__fak_context_restore when explicitly present
		name, present := determineResponsesRestoreTool([]responsesTool{
			{Name: "exec_command"},
			{Name: "mcp__fak_guard__fak_context_restore"},
		})
		if !present || name != "mcp__fak_guard__fak_context_restore" {
			t.Fatalf("expected mcp__fak_guard__fak_context_restore, got %q (present=%v)", name, present)
		}

		// 2. mcp__fak__fak_context_restore when present
		name, present = determineResponsesRestoreTool([]responsesTool{
			{Name: "exec_command"},
			{Name: "mcp__fak__fak_context_restore"},
		})
		if !present || name != "mcp__fak__fak_context_restore" {
			t.Fatalf("expected mcp__fak__fak_context_restore, got %q (present=%v)", name, present)
		}

		// 3. fak_context_restore when present
		name, present = determineResponsesRestoreTool([]responsesTool{
			{Name: "fak_context_restore"},
		})
		if !present || name != "fak_context_restore" {
			t.Fatalf("expected fak_context_restore, got %q (present=%v)", name, present)
		}

		// 4. mcp__fak_guard__ prefix inference
		name, present = determineResponsesRestoreTool([]responsesTool{
			{Name: "mcp__fak_guard__fak_read"},
		})
		if present || name != "mcp__fak_guard__fak_context_restore" {
			t.Fatalf("expected inferred mcp__fak_guard__fak_context_restore, got %q (present=%v)", name, present)
		}

		// 5. Default fallback to mcp__fak__fak_context_restore
		name, present = determineResponsesRestoreTool([]responsesTool{
			{Name: "exec_command"},
			{Name: "custom_tool"},
		})
		if present || name != "mcp__fak__fak_context_restore" {
			t.Fatalf("expected default mcp__fak__fak_context_restore, got %q (present=%v)", name, present)
		}
	})

	t.Run("auto-advertise_over_http", func(t *testing.T) {
		srv := newTestServer(t)
		planner := &capturingResponsesPlanner{
			comp: &agent.Completion{
				Message: agent.Message{Role: agent.RoleAssistant, Content: "done"},
			},
		}
		srv.planner = planner

		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()

		// Request without any restore tool declared
		body := map[string]any{
			"model": "test-model",
			"input": "test prompt",
			"tools": []any{
				map[string]any{
					"type":        "function",
					"name":        "fetch_metric",
					"description": "fetch a metric",
					"parameters":  map[string]any{"type": "object"},
				},
			},
		}

		status, _ := postResponses(t, ts.URL, body)
		if status != http.StatusOK {
			t.Fatalf("expected 200, got %d", status)
		}

		// Verify the restore tool was auto-advertised to the planner
		var found bool
		for _, tool := range planner.tools {
			if tool.Function.Name == "mcp__fak__fak_context_restore" {
				found = true
				if !strings.Contains(tool.Function.Description, "Restore dropped context") {
					t.Errorf("unexpected description: %s", tool.Function.Description)
				}
				break
			}
		}
		if !found {
			t.Fatal("mcp__fak__fak_context_restore was not auto-advertised in planner tools")
		}
	})
}

func TestResponsesRestoreToolInBandInterception(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-restore-in-band"

	// Stash a known context payload
	originalBytes := []byte("CRITICAL_RECOVERY_PAYLOAD_FOR_IN_BAND_TEST")
	sum := sha256.Sum256(originalBytes)
	digest := hex.EncodeToString(sum[:])
	srv.stashRestore(trace, digest, "recovery test", originalBytes)
	srv.stashRestore(trace, "sha256:"+digest, "recovery test", originalBytes)

	// Planner returns a tool call to mcp__fak__fak_context_restore
	planner := &capturingResponsesPlanner{
		comp: &agent.Completion{
			Message: agent.Message{
				Role: agent.RoleAssistant,
				ToolCalls: []agent.ToolCall{
					{
						ID:   "call_restore_1",
						Type: "function",
						Function: agent.Func{
							Name:      "mcp__fak__fak_context_restore",
							Arguments: `{"id":"sha256:` + digest + `"}`,
						},
					},
				},
			},
		},
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := map[string]any{
		"model": "test-model",
		"input": "please restore dropped context",
	}

	reqBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/responses", strings.NewReader(string(reqBytes)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", trace)

	httpResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", httpResp.StatusCode)
	}

	var resp responsesResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	// Verify the tool call was intercepted in-band and served inline
	for _, item := range resp.Output {
		if item.Type == "function_call" && item.Name == "mcp__fak__fak_context_restore" {
			t.Fatalf("tool call was leaked to client instead of intercepted in-band: %+v", item)
		}
	}

	if !strings.Contains(resp.OutputText, string(originalBytes)) {
		t.Fatalf("expected output_text to contain restored bytes, got: %q", resp.OutputText)
	}
}

func TestResponsesElideMarkerUsesAlignedToolName(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-responses-elide-aligned-name"

	bigOutput := strings.Repeat("M", 2000)
	messages := []agent.Message{
		{Role: agent.RoleTool, ToolCallID: "c0", Content: bigOutput},
		{Role: agent.RoleTool, ToolCallID: "c1", Content: "r1"},
		{Role: agent.RoleTool, ToolCallID: "c2", Content: "r2"},
		{Role: agent.RoleTool, ToolCallID: "c3", Content: "r3"},
		{Role: agent.RoleTool, ToolCallID: "c4", Content: "r4"},
	}

	elided := srv.maybeElideResponsesToolResults(trace, messages, "mcp__fak__fak_context_restore")
	sum := sha256.Sum256([]byte(bigOutput))
	digest := hex.EncodeToString(sum[:])
	wantMarker := "...[fak: tool output elided (len=2000 bytes); recover original via mcp__fak__fak_context_restore id=sha256:" + digest + "]..."

	if elided[0].Content != wantMarker {
		t.Fatalf("marker mismatch:\ngot:  %q\nwant: %q", elided[0].Content, wantMarker)
	}
}
