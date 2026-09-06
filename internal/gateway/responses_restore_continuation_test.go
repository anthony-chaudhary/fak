package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// #11831: restoration is a tool result, not the answer to the user's task.
func TestResponsesRestoreContinuation(t *testing.T) {
	t.Run("failed quarantine reset stops sampling", func(t *testing.T) {
		cache := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "reset denied", http.StatusInternalServerError)
		}))
		defer cache.Close()
		srv := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model", EngineCacheEngine: "vllm", EngineCacheBaseURL: cache.URL})
		abi.RegisterAdjudicator(1, readAdj{})
		abi.RegisterResultAdmitter(0, quarantineAdmitter{})
		const trace = "restore-reset-failed"
		id := ctxplan.Digest([]byte("held page"))
		srv.stashRestore(trace, id, "evidence", []byte("held page"))
		planner := &scriptedMultiTurnPlanner{turns: []*agent.Completion{toolCallTurn("restore", "fak_context_restore", fmt.Sprintf(`{"id":%q}`, id))}}
		srv.planner = planner
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()
		resp := postResponsesTrace(t, ts.URL, trace, map[string]any{"input": "Finish the task."})
		if planner.turn != 1 || len(planner.capturedMsgs) != 1 || resp.Status != "incomplete" {
			t.Fatalf("sampled after failed cache reset: samples=%d response=%+v", len(planner.capturedMsgs), resp)
		}
	})
	t.Run("malformed next completion remains incomplete", func(t *testing.T) {
		srv := newTestServer(t)
		abi.RegisterAdjudicator(1, readAdj{})
		const trace = "restore-malformed"
		id := ctxplan.Digest([]byte("saved page"))
		srv.stashRestore(trace, id, "evidence", []byte("saved page"))
		planner := &scriptedMultiTurnPlanner{turns: []*agent.Completion{
			toolCallTurn("restore", "fak_context_restore", fmt.Sprintf(`{"id":%q}`, id)),
			{Message: agent.Message{Role: agent.RoleAssistant}, ToolCallsDropped: true},
		}}
		srv.planner = planner
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()
		resp := postResponsesTrace(t, ts.URL, trace, map[string]any{"input": "Finish the task."})
		if planner.turn != 2 || resp.Status != "incomplete" {
			t.Fatalf("malformed completion did not terminate safely: %+v", resp)
		}
	})
	for _, resultGate := range []bool{false, true} {
		t.Run(fmt.Sprintf("refusal before model continuation resultGate=%v", resultGate), func(t *testing.T) {
			srv := newTestServer(t)
			if resultGate {
				abi.RegisterAdjudicator(1, readAdj{})
				abi.RegisterResultAdmitter(0, quarantineAdmitter{})
			}
			const trace = "restore-refused"
			const secret = "held restoration must never enter the model"
			id := ctxplan.Digest([]byte(secret))
			srv.stashRestore(trace, id, "evidence", []byte(secret))
			planner := &scriptedMultiTurnPlanner{turns: []*agent.Completion{
				toolCallTurn("held-restore", "mcp__fak__fak_context_restore", fmt.Sprintf(`{"id":%q}`, id)),
				toolCallTurn("allowed-next", "allow_read", `{}`),
			}}
			srv.planner = planner
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()
			resp := postResponsesTrace(t, ts.URL, trace, map[string]any{"input": "Finish the original task."})
			if planner.turn != 2 || functionCallItems(resp.Output)["allow_read"].CallID != "allowed-next" {
				t.Fatalf("refusal prevented allowed continuation: samples=%d response=%+v", planner.turn, resp)
			}
			found := false
			for _, msg := range planner.capturedMsgs[1] {
				if strings.Contains(msg.Content, secret) {
					t.Fatal("refused restored bytes reached the model")
				}
				if msg.Role == agent.RoleTool && msg.ToolCallID == "held-restore" {
					found = strings.Contains(msg.Content, `"verdict"`) && strings.Contains(msg.Content, `"error"`)
				}
			}
			if !found || strings.Contains(resp.OutputText, secret) {
				t.Fatal("missing structured refusal or leaked restored answer")
			}
		})
	}
	t.Run("injected tool returns bounded result to model", func(t *testing.T) {
		srv := newTestServer(t)
		abi.RegisterAdjudicator(1, readAdj{})
		const trace = "restore-continuation"
		body := strings.Repeat("restorable evidence ", 1000)
		id := ctxplan.Digest([]byte(body))
		srv.stashRestore(trace, id, "evidence", []byte(body))
		first := toolCallTurn("restore-1", "mcp__fak__fak_context_restore", `{"id":"`+id+`","limit":32}`)
		first.Message.Content = "I edited x.go."
		first.Message.ToolCalls = append(first.Message.ToolCalls, agent.ToolCall{ID: "not-executed", Type: "function", Function: agent.Func{Name: "allow_write", Arguments: `{}`}})
		first.Usage = agent.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12}
		second := toolCallTurn("read-next", "allow_read", `{}`)
		second.Usage = agent.Usage{PromptTokens: 20, CompletionTokens: 3, TotalTokens: 23}
		planner := &scriptedMultiTurnPlanner{turns: []*agent.Completion{first, second}}
		srv.planner = planner
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()
		resp := postResponsesTrace(t, ts.URL, trace, map[string]any{"input": "Use the saved evidence to finish the task."})
		if planner.turn != 2 || functionCallItems(resp.Output)["allow_read"].CallID != "read-next" {
			t.Fatalf("restore ended the model turn: samples=%d output=%+v", planner.turn, resp.Output)
		}
		if strings.Contains(resp.OutputText, "restorable evidence") || strings.Contains(resp.OutputText, "restored context") {
			t.Fatal("restored bytes escaped as an assistant answer")
		}
		var restored CtxRestoreResult
		found := false
		for _, m := range planner.capturedMsgs[1] {
			if strings.Contains(m.Content, "I edited x.go") {
				t.Fatal("unwitnessed edit claim entered the continuation")
			}
			for _, call := range m.ToolCalls {
				if call.ID == "not-executed" {
					t.Fatal("deferred client call has no result and must not enter internal history")
				}
			}
			if m.Role == agent.RoleTool && m.ToolCallID == "restore-1" {
				found = true
				if err := json.Unmarshal([]byte(m.Content), &restored); err != nil {
					t.Fatal(err)
				}
				if !strings.Contains(m.Content, "have not executed") {
					t.Fatal("mixed proposal did not explain deferred calls")
				}
			}
		}
		if !found || restored.Bytes != body[:32] || !restored.HasMore || restored.NextOffset != 32 || restored.Continuation == nil {
			t.Fatalf("missing structured restore result or pagination: found=%v result=%+v", found, restored)
		}
		if resp.Usage.InputTokens != 30 || resp.Usage.OutputTokens != 5 || resp.Usage.TotalTokens != 35 {
			t.Fatalf("continuation usage not accounted: %+v", resp.Usage)
		}
		for _, def := range planner.capturedTools[0] {
			if !isRestoreTool(def.Function.Name) {
				continue
			}
			var schema struct {
				Properties map[string]json.RawMessage `json:"properties"`
			}
			if err := json.Unmarshal(def.Function.Parameters, &schema); err != nil {
				t.Fatal(err)
			}
			for _, key := range []string{"offset", "limit", "range"} {
				if len(schema.Properties[key]) == 0 {
					t.Errorf("injected restore schema omits %s", key)
				}
			}
		}
	})

	t.Run("repeated restoration remains incomplete", func(t *testing.T) {
		srv := newTestServer(t)
		abi.RegisterAdjudicator(1, readAdj{})
		const trace = "restore-repeat"
		body := []byte("the same saved evidence")
		id := ctxplan.Digest(body)
		srv.stashRestore(trace, id, "evidence", body)
		planner := &sequencePlanner{comps: []*agent.Completion{toolCallTurn("restore-repeat", "mcp__fak__fak_context_restore", `{"id":"`+id+`"}`)}}
		srv.planner = planner
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()
		resp := postResponsesTrace(t, ts.URL, trace, map[string]any{"input": "Finish the original task."})
		if planner.calls() != 2 || resp.Status != "incomplete" || resp.IncompleteDetails == nil {
			t.Fatalf("repeated restore was not bounded: calls=%d response=%+v", planner.calls(), resp)
		}
		if strings.Contains(resp.OutputText, string(body)) {
			t.Fatal("repeated restore leaked as final answer")
		}
	})

	t.Run("registered MCP tool retains its round trip", func(t *testing.T) {
		srv := newTestServer(t)
		abi.RegisterAdjudicator(1, readAdj{})
		abi.RegisterAdjudicator(1, readAdj{})
		const trace = "restore-client"
		const name = "mcp__fak_guard__fak_context_restore"
		body := []byte("saved client evidence")
		id := ctxplan.Digest(body)
		srv.stashRestore(trace, id, "evidence", body)
		args := fmt.Sprintf(`{"id":%q,"trace_id":%q,"limit":5}`, id, trace)
		planner := &scriptedMultiTurnPlanner{turns: []*agent.Completion{
			toolCallTurn("client-restore", name, args),
			{Message: agent.Message{Role: agent.RoleAssistant, Content: "The task is finished."}, FinishReason: "stop"},
		}}
		srv.planner = planner
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()
		tools := []map[string]any{{"type": "function", "name": name}}
		resp := postResponsesTrace(t, ts.URL, trace, map[string]any{"input": "Finish the task.", "tools": tools})
		call := functionCallItems(resp.Output)[name]
		if planner.turn != 1 || call.CallID != "client-restore" || call.Arguments != args || resp.OutputText != "" {
			t.Fatalf("client restore lost its tool protocol: samples=%d response=%+v", planner.turn, resp)
		}
		result := callMCPTool[CtxRestoreResult](t, srv, name, map[string]any{"id": id, "trace_id": trace, "limit": 5})
		if result.Bytes != string(body[:5]) || result.Continuation == nil {
			t.Fatalf("MCP result lost metadata: %+v", result)
		}
		resultJSON, _ := json.Marshal(result)
		resp = postResponsesTrace(t, ts.URL, trace, map[string]any{"tools": tools, "input": []any{
			map[string]any{"role": "user", "content": "Finish the task."},
			map[string]any{"type": "function_call", "call_id": call.CallID, "name": name, "arguments": args},
			map[string]any{"type": "function_call_output", "call_id": call.CallID, "output": string(resultJSON)},
		}})
		if planner.turn != 2 || resp.OutputText != "The task is finished." {
			t.Fatalf("MCP continuation did not reach model: %+v", resp)
		}
		found := false
		for _, msg := range planner.capturedMsgs[1] {
			if msg.Role == agent.RoleTool && msg.ToolCallID == call.CallID && strings.Contains(msg.Content, `"has_more":true`) {
				found = true
			}
		}
		if !found {
			t.Fatal("model did not consume the matching MCP tool result")
		}
	})

	t.Run("oversized limits and advancing pages stay bounded", func(t *testing.T) {
		srv := newTestServer(t)
		abi.RegisterAdjudicator(1, readAdj{})
		const trace = "restore-budget"
		body := []byte(strings.Repeat("b", 80<<10))
		id := ctxplan.Digest(body)
		srv.stashRestore(trace, id, strings.Repeat("large orientation excerpt ", 8000), body)
		planner := &scriptedMultiTurnPlanner{}
		for i := 0; i < 5; i++ {
			planner.turns = append(planner.turns, toolCallTurn(fmt.Sprintf("page-%d", i), "mcp__fak__fak_context_restore", fmt.Sprintf(`{"id":%q,"offset":%d,"limit":%d}`, id, i*(16<<10), int(^uint(0)>>1))))
		}
		srv.planner = planner
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()
		resp := postResponsesTrace(t, ts.URL, trace, map[string]any{"input": "Read the saved evidence."})
		if planner.turn < 2 || planner.turn > 5 || resp.Status != "incomplete" || resp.IncompleteDetails == nil {
			t.Fatalf("paging escaped bound: samples=%d status=%s", planner.turn, resp.Status)
		}
		total := 0
		for _, msg := range planner.capturedMsgs[len(planner.capturedMsgs)-1] {
			if msg.Role != agent.RoleTool {
				continue
			}
			var result CtxRestoreResult
			if err := json.Unmarshal([]byte(msg.Content), &result); err != nil {
				t.Fatal(err)
			}
			if len(result.Bytes) > 16<<10 || result.Continuation == nil {
				t.Fatalf("unbounded page: bytes=%d", len(result.Bytes))
			}
			if len(result.Excerpt) > 512 {
				t.Fatal("unbounded orientation excerpt")
			}
			total += len(msg.Content)
		}
		if total == 0 || total > 64<<10 {
			t.Fatalf("admitted %d serialized bytes, want 1..64KiB", total)
		}
	})
}
