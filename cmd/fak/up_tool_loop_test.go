package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type turnkeyToolLoopPlanner struct {
	messages [][]agent.Message
	tools    [][]agent.ToolDef
	params   []agent.SampleParams
	dropped  bool
}

func (*turnkeyToolLoopPlanner) Model() string { return "halo-test" }

func (p *turnkeyToolLoopPlanner) Complete(_ context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	params := agent.SampleParams{}
	for _, opt := range opts {
		opt(&params)
	}
	p.messages = append(p.messages, append([]agent.Message(nil), messages...))
	p.tools = append(p.tools, append([]agent.ToolDef(nil), tools...))
	p.params = append(p.params, params)
	if p.dropped {
		return &agent.Completion{
			Message:          agent.Message{Role: agent.RoleAssistant, Content: `truncated <tool_call>{"name":"read_file"`},
			FinishReason:     "length",
			ToolCallsDropped: true,
		}, nil
	}

	if len(messages) > 0 && messages[len(messages)-1].Role == agent.RoleTool {
		return &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "accepted: " + messages[len(messages)-1].Content},
			FinishReason: "stop",
			Usage:        agent.Usage{PromptTokens: 12, CompletionTokens: 2, TotalTokens: 14},
		}, nil
	}
	return &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{
			ID: "call_read_1", Type: "function",
			Function: agent.Func{Name: "read_file", Arguments: `{"path":"main.go"}`},
		}}},
		FinishReason: "tool_calls",
		Usage:        agent.Usage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12},
	}, nil
}

func TestTurnkeyServerRejectsDroppedToolCall(t *testing.T) {
	planner := &turnkeyToolLoopPlanner{dropped: true}
	ts := httptest.NewServer(http.HandlerFunc(newTurnkeyToolTestServer(planner).handleChatCompletions))
	defer ts.Close()

	resp, err := http.Post(ts.URL, "application/json", strings.NewReader(`{"model":"halo-test","messages":[{"role":"user","content":"Read main.go"}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", resp.StatusCode)
	}
	var wire struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
		Choices []any `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&wire); err != nil {
		t.Fatal(err)
	}
	if wire.Error.Code != "tool_call_conformance" || len(wire.Choices) != 0 {
		t.Fatalf("dropped call serialized as benign completion: %#v", wire)
	}
}

func newTurnkeyToolTestServer(p agent.Planner) *turnkeyServer {
	return &turnkeyServer{planner: p, engineID: "inkernel"}
}

func TestTurnkeyServerToolCallRoundTrip(t *testing.T) {
	planner := &turnkeyToolLoopPlanner{}
	ts := httptest.NewServer(http.HandlerFunc(newTurnkeyToolTestServer(planner).handleChatCompletions))
	defer ts.Close()

	first := `{"model":"halo-test","messages":[{"role":"user","content":"Read main.go"}],"tools":[{"type":"function","function":{"name":"read_file","description":"read a repository file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"tool_choice":{"type":"function","function":{"name":"read_file"}},"temperature":0,"stream":false}`
	resp, err := http.Post(ts.URL, "application/json", strings.NewReader(first))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var firstWire struct {
		Choices []struct {
			Message      agent.Message `json:"message"`
			FinishReason string        `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&firstWire); err != nil {
		t.Fatal(err)
	}
	if len(planner.tools) != 1 || len(planner.tools[0]) != 1 || planner.tools[0][0].Function.Name != "read_file" {
		t.Fatalf("planner tools = %#v, want read_file schema", planner.tools)
	}
	wantSchema := `{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`
	if !jsonEqual(planner.tools[0][0].Function.Parameters, []byte(wantSchema)) {
		t.Fatalf("planner schema = %s, want %s", planner.tools[0][0].Function.Parameters, wantSchema)
	}
	if len(planner.params) != 1 || planner.params[0].Temperature == nil || *planner.params[0].Temperature != 0 {
		t.Fatalf("explicit temperature:0 was not preserved: %#v", planner.params)
	}
	wantChoice := `{"type":"function","function":{"name":"read_file"}}`
	if !jsonEqual(planner.params[0].ToolChoice, []byte(wantChoice)) {
		t.Fatalf("tool_choice = %s, want %s", planner.params[0].ToolChoice, wantChoice)
	}
	if len(firstWire.Choices) != 1 || firstWire.Choices[0].FinishReason != "tool_calls" || len(firstWire.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("structured tool-call response lost: %#v", firstWire)
	}
	call := firstWire.Choices[0].Message.ToolCalls[0]
	if call.ID != "call_read_1" || call.Function.Name != "read_file" || call.Function.Arguments != `{"path":"main.go"}` {
		t.Fatalf("tool call changed on wire: %#v", call)
	}

	repo := t.TempDir()
	wantFile := "package main\n\nconst haloNonce = \"halo-tool-loop-7f3a\"\n"
	if err := os.WriteFile(filepath.Join(repo, "main.go"), []byte(wantFile), 0o600); err != nil {
		t.Fatal(err)
	}
	toolResult, err := os.ReadFile(filepath.Join(repo, "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	secondBody, err := json.Marshal(map[string]any{
		"model": "halo-test", "temperature": 0,
		"messages": []any{
			map[string]any{"role": "user", "content": "Read main.go"},
			map[string]any{"role": "assistant", "content": nil, "tool_calls": []agent.ToolCall{call}},
			map[string]any{"role": "tool", "tool_call_id": call.ID, "name": call.Function.Name, "content": string(toolResult)},
		},
		"tools": []any{map[string]any{"type": "function", "function": map[string]any{"name": "read_file", "parameters": map[string]any{"type": "object"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	resp2, err := http.Post(ts.URL, "application/json", bytes.NewReader(secondBody))
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	if len(planner.messages) != 2 || len(planner.messages[1]) != 3 {
		t.Fatalf("continuation transcript = %#v", planner.messages)
	}
	gotAssistant, gotResult := planner.messages[1][1], planner.messages[1][2]
	if len(gotAssistant.ToolCalls) != 1 || gotAssistant.ToolCalls[0].ID != "call_read_1" {
		t.Fatalf("assistant tool call lost on continuation: %#v", gotAssistant)
	}
	if gotResult.Role != agent.RoleTool || gotResult.ToolCallID != "call_read_1" || gotResult.Name != "read_file" || gotResult.Content != wantFile {
		t.Fatalf("tool result binding/content lost on continuation: %#v", gotResult)
	}
	var secondWire struct {
		Choices []struct {
			Message agent.Message `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp2.Body).Decode(&secondWire); err != nil {
		t.Fatal(err)
	}
	if len(secondWire.Choices) != 1 || secondWire.Choices[0].Message.Content != "accepted: "+wantFile {
		t.Fatalf("final answer did not consume exact client tool result: %#v", secondWire)
	}
}

func jsonEqual(a, b []byte) bool {
	var av, bv any
	return json.Unmarshal(a, &av) == nil && json.Unmarshal(b, &bv) == nil && string(mustCanonicalJSON(av)) == string(mustCanonicalJSON(bv))
}

func mustCanonicalJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

func TestTurnkeyServerStreamsStructuredToolCallWithoutExecutingIt(t *testing.T) {
	planner := &turnkeyToolLoopPlanner{}
	ts := httptest.NewServer(http.HandlerFunc(newTurnkeyToolTestServer(planner).handleChatCompletions))
	defer ts.Close()

	body := `{"model":"halo-test","messages":[{"role":"user","content":"Read main.go"}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}],"temperature":0,"stream":true}`
	resp, err := http.Post(ts.URL, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var calls []agent.ToolCall
	finish := ""
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Bytes()
		if bytes.HasPrefix(line, []byte("data: ")) && !bytes.Equal(line, []byte("data: [DONE]")) {
			var chunk struct {
				Choices []struct {
					Delta struct {
						ToolCalls []agent.ToolCall `json:"tool_calls"`
					} `json:"delta"`
					FinishReason *string `json:"finish_reason"`
				} `json:"choices"`
			}
			if err := json.Unmarshal(bytes.TrimPrefix(line, []byte("data: ")), &chunk); err != nil {
				t.Fatal(err)
			}
			if len(chunk.Choices) == 1 {
				calls = append(calls, chunk.Choices[0].Delta.ToolCalls...)
				if chunk.Choices[0].FinishReason != nil {
					finish = *chunk.Choices[0].FinishReason
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || calls[0].ID != "call_read_1" || calls[0].Function.Name != "read_file" || calls[0].Function.Arguments != `{"path":"main.go"}` || finish != "tool_calls" {
		t.Fatalf("SSE tool call = %#v finish=%q", calls, finish)
	}
	if len(planner.messages) != 1 {
		t.Fatalf("handler unexpectedly executed/re-entered the tool loop: planner calls=%d", len(planner.messages))
	}
}
