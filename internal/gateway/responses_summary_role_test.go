package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

// The estimate is unmeasured. Requests are serial because the fixture owns the ABI
// registry; each real Handler request has a five-second cancellation backstop.
// fak-test:runtime integration est=200ms lane=default
func TestResponsesSummaryRetainsTrustBoundary(t *testing.T) {
	t.Run("summary cases", testResponsesSummaryTrustBoundary)

	// The next test may construct a server directly, without newTestServer.
	// Exercise that ordering after the summary fixture's cleanup has completed.
	if _, ok := abi.ActiveResolver().(inlineRes); !ok {
		t.Fatalf("summary cleanup resolver = %T, want inlineRes", abi.ActiveResolver())
	}
	if len(abi.ResultAdmitters()) != 0 {
		t.Fatal("summary cleanup left result admitters registered")
	}
	srv, err := New(Config{EngineID: "mock", Model: "test-model"})
	if err != nil {
		t.Fatalf("New after summary cleanup: %v", err)
	}
	t.Cleanup(srv.Close)
}

func testResponsesSummaryTrustBoundary(t *testing.T) {
	t.Cleanup(func() {
		abi.ResetForTest()
		// Restore both construction prerequisites used by direct New callers.
		abi.RegisterRegionBackend(inlineBackend{})
		abi.RegisterEngine("mock", engine.MockEngine)
	})
	const secret = "sk-abcdef0123456789abcdef0123"
	for _, tc := range []struct {
		name, role, body string
		quarantined      bool
	}{
		{"user summary", agent.RoleUser, "<summary>archived user notes</summary>", false},
		{"assistant summary", agent.RoleAssistant, "<compaction_summary>archived assistant notes</compaction_summary>", false},
		{"tool summary", agent.RoleTool, "<summary>archived tool notes</summary>", false},
		{"quarantined tool summary", agent.RoleTool, "<compaction_summary>api_key=" + secret + "</compaction_summary>", true},
		{"trusted instructions", agent.RoleSystem, "<summary>trusted compacted history</summary>", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			abi.RegisterResultAdmitter(10, ctxmmu.New())
			planner := &continuationPlanner{completions: []*agent.Completion{{
				Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop",
			}}}
			srv.planner = planner
			instructions := "You are a helpful assistant."
			input := []map[string]any{{"type": "message", "role": "user", "content": "Continue."}}
			switch tc.role {
			case agent.RoleSystem:
				instructions += "\n" + tc.body
			case agent.RoleTool:
				input = append(input,
					map[string]any{"type": "function_call", "call_id": "summary_call", "name": "fetch_url", "arguments": "{}"},
					map[string]any{"type": "function_call_output", "call_id": "summary_call", "output": tc.body},
				)
			default:
				input = append(input, map[string]any{"type": "message", "role": tc.role, "content": tc.body})
			}
			raw, err := json.Marshal(map[string]any{
				"model": "fixture", "instructions": instructions, "input": input, "store": false,
			})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(raw)).WithContext(ctx)
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			srv.Handler().ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("Responses status=%d: %s", rec.Code, rec.Body.String())
			}
			if len(planner.messages) != 1 {
				t.Fatalf("planner calls=%d, want one served turn", len(planner.messages))
			}
			messages := planner.captured(0)
			var found, sawAnchor, sawCall bool
			for _, msg := range messages {
				if msg.Role == agent.RoleUser && msg.Content == "Continue." {
					sawAnchor = true
				}
				if msg.Role == agent.RoleAssistant && len(msg.ToolCalls) == 1 && msg.ToolCalls[0].ID == "summary_call" {
					sawCall = true
				}
				if tc.quarantined && strings.Contains(msg.Content, secret) {
					t.Fatalf("quarantined result survived at role %q: %q", msg.Role, msg.Content)
				}
				if tc.role != agent.RoleSystem && (msg.Role == agent.RoleSystem || msg.Role == "developer") && strings.Contains(msg.Content, tc.body) {
					t.Fatalf("lower-trust summary promoted to %q: %q", msg.Role, msg.Content)
				}
				if msg.Role != tc.role {
					continue
				}
				if tc.quarantined {
					found = msg.ToolCallID == "summary_call" && strings.Contains(msg.Content, "_quarantined")
				} else if tc.role == agent.RoleSystem {
					found = found || strings.Contains(msg.Content, tc.body)
				} else if msg.Content == tc.body {
					found = tc.role != agent.RoleTool || msg.ToolCallID == "summary_call"
				}
				if found && tc.role != agent.RoleSystem && (!sawAnchor || (tc.role == agent.RoleTool && !sawCall)) {
					t.Fatal("summary moved ahead of its original conversation context")
				}
			}
			if !found {
				t.Fatalf("original role/result identity not retained: %#v", messages)
			}
			if tc.role == agent.RoleTool {
				var response responsesResponse
				if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
					t.Fatal(err)
				}
				if response.Fak == nil || len(response.Fak.ResultAdmissions) != 1 {
					t.Fatalf("tool result did not cross admission: %+v", response.Fak)
				}
				admission := response.Fak.ResultAdmissions[0]
				want := "ALLOW"
				if tc.quarantined {
					want = "QUARANTINE"
				}
				if admission.ToolCallID != "summary_call" || admission.Tool != "fetch_url" || admission.Verdict.Kind != want {
					t.Fatalf("result admission=%+v, want summary_call/fetch_url/%s", admission, want)
				}
			}
		})
	}
}
