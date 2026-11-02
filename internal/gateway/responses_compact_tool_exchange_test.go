package gateway

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=1s
func TestResponsesCompactionPreservesToolExchangeOverHTTP(t *testing.T) {
	srv := newTestServer(t)
	planner := &capturingResponsesPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, Content: "done"}, FinishReason: "stop",
	}}
	srv.planner = planner
	srv.compactHistoryBudget = 200
	server := httptest.NewServer(srv.Handler())
	defer server.Close()

	const system = "Keep the user's standing instructions."
	const firstUser = "Inspect the package and retain the goal."
	const callID = "call_compact_preserved"
	const arguments = `{ "path": "src/main.go", "offset": 7 }`
	const result = "[fak:goal] Preserve the package contract.\nRead src/main.go lines 7-12: package main"
	const recentAssistant = "The package needs a focused regression."
	const recentUser = "Continue with that regression."
	plainMiddle := strings.Repeat("unprotected historical prose ", 600)
	status, _ := postResponses(t, server.URL, map[string]any{
		"model": "test-model", "instructions": system,
		"input": []map[string]any{
			{"type": "message", "role": "user", "content": firstUser},
			{"type": "function_call", "call_id": callID, "name": "read_file", "namespace": "functions", "arguments": arguments},
			{"type": "function_call_output", "call_id": callID, "output": result},
			{"type": "message", "role": "assistant", "content": plainMiddle},
			{"type": "message", "role": "assistant", "content": recentAssistant},
			{"type": "message", "role": "user", "content": recentUser},
		},
		"tools": []map[string]any{{"type": "function", "name": "read_file", "parameters": map[string]any{"type": "object"}}},
	})
	if status != http.StatusOK || !planner.called {
		t.Fatalf("responses request status=%d planner called=%v", status, planner.called)
	}

	wantCall := agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{
		ID: callID, Type: "function", Function: agent.Func{Name: "read_file", Namespace: "functions", Arguments: arguments},
	}}}
	wantResult := agent.Message{Role: agent.RoleTool, ToolCallID: callID, Content: result}
	callIndex, resultIndex := -1, -1
	stubs := 0
	for i, message := range planner.messages {
		if len(message.ToolCalls) > 0 {
			if !reflect.DeepEqual(message, wantCall) || callIndex >= 0 {
				t.Fatalf("compaction changed or duplicated original call: role=%q calls=%d prior_index=%d", message.Role, len(message.ToolCalls), callIndex)
			}
			callIndex = i
		}
		if message.ToolCallID == callID {
			if !reflect.DeepEqual(message, wantResult) || resultIndex >= 0 {
				t.Fatalf("compaction changed or duplicated goal-bearing result: role=%q call_id=%q exact_content=%t", message.Role, message.ToolCallID, message.Content == result)
			}
			resultIndex = i
		}
		if message.Content == plainMiddle {
			t.Fatal("ordinary middle prose was not compacted; HTTP witness did not cross the compaction boundary")
		}
		if _, _, _, ok := agent.ParseResponsesCompactStub(message.Content); ok {
			stubs++
		}
	}
	if callIndex < 0 || resultIndex != callIndex+1 {
		t.Fatalf("compaction orphaned tool exchange: call index=%d result index=%d", callIndex, resultIndex)
	}
	if stubs != 1 {
		t.Fatalf("restore stubs=%d, want exactly the ordinary middle prose stub", stubs)
	}
	if len(planner.messages) < 4 || planner.messages[0].Role != agent.RoleSystem || planner.messages[0].Content != system ||
		planner.messages[1].Role != agent.RoleUser || planner.messages[1].Content != firstUser {
		t.Fatal("compaction changed leading system/first-user anchors")
	}
	last := planner.messages[len(planner.messages)-2:]
	if !reflect.DeepEqual(last, []agent.Message{{Role: agent.RoleAssistant, Content: recentAssistant}, {Role: agent.RoleUser, Content: recentUser}}) {
		t.Fatalf("compaction changed recent suffix: roles=%q/%q exact_content=%t/%t", last[0].Role, last[1].Role, last[0].Content == recentAssistant, last[1].Content == recentUser)
	}
}
