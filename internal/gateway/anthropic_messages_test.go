package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

type mockQwenPlanner struct {
	output       string
	seenMessages []agent.Message
	seenTools    []agent.ToolDef
}

// bufferedQwenPlanner deliberately exposes only Planner. The compatibility
// witness still asks for Anthropic SSE while raw Qwen fragment behavior remains
// isolated to the streaming planner fixture below.
type bufferedQwenPlanner struct{ inner *mockQwenPlanner }

func (p bufferedQwenPlanner) Model() string { return p.inner.Model() }
func (p bufferedQwenPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return p.inner.Complete(ctx, messages, tools, opts...)
}

func (p *mockQwenPlanner) Model() string { return "qwen2.5-coder" }

func (p *mockQwenPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.seenMessages = append(p.seenMessages, messages...)
	p.seenTools = append(p.seenTools, tools...)
	message := agent.LiftTextToolCalls(agent.Message{
		Role:    agent.RoleAssistant,
		Content: p.output,
	})
	return &agent.Completion{
		Message:      message,
		FinishReason: "stop",
		Usage: agent.Usage{
			PromptTokens:     10,
			CompletionTokens: 15,
		},
	}, nil
}

func (p *mockQwenPlanner) StreamingSupported() bool { return true }

func (p *mockQwenPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.seenMessages = append(p.seenMessages, messages...)
	p.seenTools = append(p.seenTools, tools...)
	if sink != nil {
		_ = sink(p.output)
	}
	return &agent.Completion{
		Message: agent.Message{
			Role:    agent.RoleAssistant,
			Content: p.output,
		},
		FinishReason: "stop",
		Usage: agent.Usage{
			PromptTokens:     10,
			CompletionTokens: 15,
		},
	}, nil
}

type parsedSSEEvent struct {
	Event string
	Data  string
}

func parseSSEStream(r io.Reader) ([]parsedSSEEvent, error) {
	var events []parsedSSEEvent
	scanner := bufio.NewScanner(r)
	var currentEvent, currentData string

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if currentEvent != "" || currentData != "" {
				events = append(events, parsedSSEEvent{
					Event: currentEvent,
					Data:  currentData,
				})
				currentEvent = ""
				currentData = ""
			}
			continue
		}
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			currentData = strings.TrimPrefix(line, "data: ")
		}
	}
	if currentEvent != "" || currentData != "" {
		events = append(events, parsedSSEEvent{
			Event: currentEvent,
			Data:  currentData,
		})
	}
	return events, scanner.Err()
}

type qwenTestAdj struct{}

func (qwenTestAdj) Caps() []abi.Capability { return nil }
func (qwenTestAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test"}
}

func newQwenTestServer(t *testing.T) *Server {
	t.Helper()
	srv := newTestServer(t)
	abi.RegisterAdjudicator(0, qwenTestAdj{})
	return srv
}

// TestAnthropicMessagesAPI_QwenToolStreaming is the mandatory verifiable witness:
// Sets up a test gateway Server, simulates a mock Claude Code client sending a streaming
// request (stream: true) to /v1/messages with tools, where the model output contains
// Qwen ChatML <tool_call> syntax, and verifies that the SSE stream delivers the expected events:
// - message_start
// - content_block_start (text)
// - content_block_delta (text: "I will check the weather.")
// - content_block_stop
// - content_block_start (tool_use, name: "get_weather")
// - content_block_delta (input_json_delta)
// - content_block_stop
// - message_delta (stop_reason: "tool_use")
// - message_stop
func TestAnthropicMessagesAPI_QwenToolStreaming(t *testing.T) {
	srv := newQwenTestServer(t)
	planner := &mockQwenPlanner{
		output: "I will check the weather.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"San Francisco\"}}\n</tool_call>",
	}
	srv.planner = bufferedQwenPlanner{inner: planner}

	ts := httptest.NewServer(srv.AnthropicMessagesHandler())
	defer ts.Close()

	reqBody := map[string]any{
		"model":      "qwen2.5-coder",
		"max_tokens": 1024,
		"stream":     true,
		"messages": []map[string]any{
			{"role": "user", "content": "What is the weather in San Francisco?"},
		},
		"tools": []map[string]any{
			{
				"name":        "get_weather",
				"description": "Get current weather in a city",
				"input_schema": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{"type": "string"},
					},
					"required": []string{"city"},
				},
			},
		},
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/messages", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, body)
	}

	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}

	events, err := parseSSEStream(resp.Body)
	if err != nil {
		t.Fatalf("parse SSE stream: %v", err)
	}

	if len(events) != 9 {
		t.Fatalf("got %d SSE events, want 9. Events: %+v", len(events), events)
	}

	// 1. message_start
	if events[0].Event != "message_start" {
		t.Errorf("event[0] = %q, want message_start", events[0].Event)
	}
	var msgStart map[string]any
	if err := json.Unmarshal([]byte(events[0].Data), &msgStart); err != nil {
		t.Fatalf("unmarshal message_start data: %v", err)
	}
	if msgStart["type"] != "message_start" {
		t.Errorf("message_start type = %v, want message_start", msgStart["type"])
	}

	// 2. content_block_start (text)
	if events[1].Event != "content_block_start" {
		t.Errorf("event[1] = %q, want content_block_start", events[1].Event)
	}
	var blkStart0 map[string]any
	if err := json.Unmarshal([]byte(events[1].Data), &blkStart0); err != nil {
		t.Fatalf("unmarshal content_block_start[0]: %v", err)
	}
	if blk, ok := blkStart0["content_block"].(map[string]any); !ok || blk["type"] != "text" {
		t.Errorf("content_block[0] type = %v, want text", blkStart0["content_block"])
	}

	// 3. content_block_delta (text: "I will check the weather.")
	if events[2].Event != "content_block_delta" {
		t.Errorf("event[2] = %q, want content_block_delta", events[2].Event)
	}
	var blkDelta0 map[string]any
	if err := json.Unmarshal([]byte(events[2].Data), &blkDelta0); err != nil {
		t.Fatalf("unmarshal content_block_delta[0]: %v", err)
	}
	delta0, ok := blkDelta0["delta"].(map[string]any)
	if !ok || delta0["type"] != "text_delta" {
		t.Errorf("delta[0] type = %v, want text_delta", blkDelta0["delta"])
	}
	if gotText := delta0["text"]; gotText != "I will check the weather." {
		t.Errorf("delta[0] text = %q, want %q", gotText, "I will check the weather.")
	}

	// 4. content_block_stop
	if events[3].Event != "content_block_stop" {
		t.Errorf("event[3] = %q, want content_block_stop", events[3].Event)
	}

	// 5. content_block_start (tool_use, name: "get_weather")
	if events[4].Event != "content_block_start" {
		t.Errorf("event[4] = %q, want content_block_start", events[4].Event)
	}
	var blkStart1 map[string]any
	if err := json.Unmarshal([]byte(events[4].Data), &blkStart1); err != nil {
		t.Fatalf("unmarshal content_block_start[1]: %v", err)
	}
	blk1, ok := blkStart1["content_block"].(map[string]any)
	if !ok || blk1["type"] != "tool_use" {
		t.Errorf("content_block[1] type = %v, want tool_use", blkStart1["content_block"])
	}
	if blk1["name"] != "get_weather" {
		t.Errorf("content_block[1] name = %v, want get_weather", blk1["name"])
	}
	id1, _ := blk1["id"].(string)
	if id1 == "" {
		t.Error("content_block[1] id is empty")
	}

	// 6. content_block_delta (input_json_delta)
	if events[5].Event != "content_block_delta" {
		t.Errorf("event[5] = %q, want content_block_delta", events[5].Event)
	}
	var blkDelta1 map[string]any
	if err := json.Unmarshal([]byte(events[5].Data), &blkDelta1); err != nil {
		t.Fatalf("unmarshal content_block_delta[1]: %v", err)
	}
	delta1, ok := blkDelta1["delta"].(map[string]any)
	if !ok || delta1["type"] != "input_json_delta" {
		t.Errorf("delta[1] type = %v, want input_json_delta", blkDelta1["delta"])
	}
	partJSON, _ := delta1["partial_json"].(string)
	if !strings.Contains(partJSON, "city") || !strings.Contains(partJSON, "San Francisco") {
		t.Errorf("partial_json = %q, want to contain city and San Francisco", partJSON)
	}

	// 7. content_block_stop
	if events[6].Event != "content_block_stop" {
		t.Errorf("event[6] = %q, want content_block_stop", events[6].Event)
	}

	// 8. message_delta (stop_reason: "tool_use")
	if events[7].Event != "message_delta" {
		t.Errorf("event[7] = %q, want message_delta", events[7].Event)
	}
	var msgDelta map[string]any
	if err := json.Unmarshal([]byte(events[7].Data), &msgDelta); err != nil {
		t.Fatalf("unmarshal message_delta: %v", err)
	}
	deltaMap, ok := msgDelta["delta"].(map[string]any)
	if !ok || deltaMap["stop_reason"] != "tool_use" {
		t.Errorf("message_delta stop_reason = %v, want tool_use", deltaMap["stop_reason"])
	}

	// 9. message_stop
	if events[8].Event != "message_stop" {
		t.Errorf("event[8] = %q, want message_stop", events[8].Event)
	}
}

// TestAnthropicMessagesAPI_MultiTurnToolResult verifies that incoming tool_result blocks
// are properly translated and routed to the model, and that multi-turn tool loops succeed.
func TestAnthropicMessagesAPI_MultiTurnToolResult(t *testing.T) {
	srv := newQwenTestServer(t)

	type multiTurnPlanner struct {
		turn int
	}
	mtp := &multiTurnPlanner{}

	customPlanner := &scriptedPlannerFunc{
		completeFn: func(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
			mtp.turn++
			if mtp.turn == 1 {
				message := agent.LiftTextToolCalls(agent.Message{
					Role:    agent.RoleAssistant,
					Content: "Let me check the weather.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"San Francisco\"}}\n</tool_call>",
				})
				return &agent.Completion{
					Message:      message,
					FinishReason: "stop",
				}, nil
			}
			// In turn 2, check that incoming messages carry tool_result (RoleTool)
			var hasToolResult bool
			for _, m := range messages {
				if m.Role == agent.RoleTool {
					hasToolResult = true
					if !strings.Contains(m.Content, "68") {
						t.Errorf("tool_result content = %q, want 68", m.Content)
					}
					if m.Name != "get_weather" {
						t.Errorf("tool_result name = %q, want get_weather", m.Name)
					}
				}
			}
			if !hasToolResult {
				t.Errorf("turn 2 did not receive RoleTool message: %+v", messages)
			}
			return &agent.Completion{
				Message: agent.Message{
					Role:    agent.RoleAssistant,
					Content: "The weather in San Francisco is 68°F.",
				},
				FinishReason: "stop",
			}, nil
		},
	}
	srv.planner = customPlanner

	ts := httptest.NewServer(srv.AnthropicMessagesHandler())
	defer ts.Close()

	// Turn 1
	body1 := map[string]any{
		"model":      "qwen2.5-coder",
		"max_tokens": 512,
		"messages": []map[string]any{
			{"role": "user", "content": "What's the weather in San Francisco?"},
		},
		"tools": []map[string]any{
			{
				"name":        "get_weather",
				"description": "Get weather",
				"input_schema": map[string]any{
					"type": "object",
				},
			},
		},
	}
	raw1, _ := json.Marshal(body1)
	resp1, err := http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(raw1))
	if err != nil {
		t.Fatalf("turn 1 post: %v", err)
	}
	defer resp1.Body.Close()

	var res1 anthropicMessageResponse
	if err := json.NewDecoder(resp1.Body).Decode(&res1); err != nil {
		t.Fatalf("turn 1 decode: %v", err)
	}
	if res1.StopReason != "tool_use" {
		t.Fatalf("turn 1 stop_reason = %q, want tool_use", res1.StopReason)
	}
	var toolUseID string
	for _, b := range res1.Content {
		if b.Type == "tool_use" {
			toolUseID = b.ID
			break
		}
	}
	if toolUseID == "" {
		t.Fatalf("turn 1 did not return a tool_use block: %+v", res1.Content)
	}

	// Turn 2 with tool_result
	body2 := map[string]any{
		"model":      "qwen2.5-coder",
		"max_tokens": 512,
		"messages": []map[string]any{
			{"role": "user", "content": "What's the weather in San Francisco?"},
			{
				"role": "assistant",
				"content": []map[string]any{
					{"type": "text", "text": "Let me check the weather."},
					{"type": "tool_use", "id": toolUseID, "name": "get_weather", "input": map[string]any{"city": "San Francisco"}},
				},
			},
			{
				"role": "user",
				"content": []map[string]any{
					{
						"type":        "tool_result",
						"tool_use_id": toolUseID,
						"content":     "{\"temperature\": 68}",
					},
				},
			},
		},
		"tools": []map[string]any{
			{
				"name":        "get_weather",
				"description": "Get weather",
				"input_schema": map[string]any{
					"type": "object",
				},
			},
		},
	}
	raw2, _ := json.Marshal(body2)
	resp2, err := http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(raw2))
	if err != nil {
		t.Fatalf("turn 2 post: %v", err)
	}
	defer resp2.Body.Close()

	var res2 anthropicMessageResponse
	if err := json.NewDecoder(resp2.Body).Decode(&res2); err != nil {
		t.Fatalf("turn 2 decode: %v", err)
	}
	if res2.StopReason != "end_turn" {
		t.Errorf("turn 2 stop_reason = %q, want end_turn", res2.StopReason)
	}
	if len(res2.Content) == 0 || res2.Content[0].Text != "The weather in San Francisco is 68°F." {
		t.Errorf("turn 2 content wrong: %+v", res2.Content)
	}
}

// TestAnthropicMessagesAPI_CountTokens verifies the POST /v1/messages/count_tokens endpoint.
func TestAnthropicMessagesAPI_CountTokens(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.AnthropicMessagesHandler())
	defer ts.Close()

	body := map[string]any{
		"model": "qwen2.5-coder",
		"messages": []map[string]any{
			{"role": "user", "content": "How many tokens does this sentence have?"},
		},
		"tools": []map[string]any{
			{"name": "test_tool", "input_schema": map[string]any{"type": "object"}},
		},
	}
	raw, _ := json.Marshal(body)

	resp, err := http.Post(ts.URL+"/v1/messages/count_tokens", "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("post count_tokens: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result map[string]int
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode count_tokens response: %v", err)
	}

	inputTokens, ok := result["input_tokens"]
	if !ok || inputTokens <= 0 {
		t.Errorf("input_tokens = %d, want > 0", inputTokens)
	}
}

// TestAnthropicMessagesAPI_NonStreaming verifies non-streaming requests with tool calls.
func TestAnthropicMessagesAPI_NonStreaming(t *testing.T) {
	srv := newQwenTestServer(t)
	planner := &mockQwenPlanner{
		output: "Checking files.\n<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"main.go\"}}\n</tool_call>",
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.AnthropicMessagesHandler())
	defer ts.Close()

	body := map[string]any{
		"model":      "qwen2.5-coder",
		"max_tokens": 512,
		"stream":     false,
		"messages": []map[string]any{
			{"role": "user", "content": "Read main.go"},
		},
		"tools": []map[string]any{
			{"name": "read_file", "input_schema": map[string]any{"type": "object"}},
		},
	}
	raw, _ := json.Marshal(body)

	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, b)
	}

	var msg anthropicMessageResponse
	if err := json.NewDecoder(resp.Body).Decode(&msg); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if msg.Type != "message" || msg.Role != "assistant" {
		t.Errorf("envelope incorrect: %+v", msg)
	}
	if msg.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", msg.StopReason)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("content blocks count = %d, want 2: %+v", len(msg.Content), msg.Content)
	}

	// Block 0: text
	if msg.Content[0].Type != "text" || msg.Content[0].Text != "Checking files." {
		t.Errorf("block[0] = %+v, want text 'Checking files.'", msg.Content[0])
	}

	// Block 1: tool_use
	if msg.Content[1].Type != "tool_use" || msg.Content[1].Name != "read_file" {
		t.Errorf("block[1] = %+v, want tool_use 'read_file'", msg.Content[1])
	}
	if msg.Content[1].ID == "" {
		t.Error("tool_use id is empty")
	}
	if !strings.Contains(string(msg.Content[1].Input), "main.go") {
		t.Errorf("tool_use input = %s, want main.go", string(msg.Content[1].Input))
	}
}

// TestAnthropicMessagesAPI_QwenXMLSyntax verifies that Qwen XML syntax (<function=...><parameter=...>)
// is properly lifted into Anthropic tool_use blocks.
func TestAnthropicMessagesAPI_QwenXMLSyntax(t *testing.T) {
	raw := "I will inspect the file.\n<tool_call>\n<function=inspect_file>\n<parameter=path>\nmain.go\n</parameter>\n</function>\n</tool_call>"
	prose, calls := ParseQwenToolCalls(raw)

	if prose != "I will inspect the file." {
		t.Errorf("prose = %q, want 'I will inspect the file.'", prose)
	}
	if len(calls) != 1 {
		t.Fatalf("calls len = %d, want 1", len(calls))
	}
	if calls[0].Function.Name != "inspect_file" {
		t.Errorf("call name = %q, want inspect_file", calls[0].Function.Name)
	}
	if !strings.HasPrefix(calls[0].ID, "toolu_") {
		t.Errorf("call ID = %q, want prefix toolu_", calls[0].ID)
	}
	if !strings.Contains(calls[0].Function.Arguments, "main.go") {
		t.Errorf("arguments = %q, want to contain main.go", calls[0].Function.Arguments)
	}
}

// TestAnthropicMessagesAPI_QwenChatMLTranslation verifies ChatML formatting helper functions.
func TestAnthropicMessagesAPI_QwenChatMLTranslation(t *testing.T) {
	msgs := []agent.Message{
		{Role: agent.RoleUser, Content: "Hello"},
		{
			Role:    agent.RoleAssistant,
			Content: "Calling tool",
			ToolCalls: []agent.ToolCall{
				{
					ID:       "toolu_123",
					Function: agent.Func{Name: "get_weather", Arguments: `{"city":"SF"}`},
				},
			},
		},
		{
			Role:       agent.RoleTool,
			ToolCallID: "toolu_123",
			Name:       "get_weather",
			Content:    `{"temp":68}`,
		},
	}

	translated := TranslateToQwenChatML(msgs)
	if len(translated) != 3 {
		t.Fatalf("translated len = %d, want 3", len(translated))
	}

	// Assistant message has <tool_call>
	if !strings.Contains(translated[1].Content, "<tool_call>") || !strings.Contains(translated[1].Content, "get_weather") {
		t.Errorf("translated[1] content = %q, want to contain <tool_call>", translated[1].Content)
	}

	// Tool message converted to user role with <tool_response>
	if translated[2].Role != agent.RoleUser {
		t.Errorf("translated[2] role = %q, want user", translated[2].Role)
	}
	if !strings.Contains(translated[2].Content, "<tool_response>") || !strings.Contains(translated[2].Content, "68") {
		t.Errorf("translated[2] content = %q, want to contain <tool_response>", translated[2].Content)
	}
}

type scriptedPlannerFunc struct {
	completeFn func(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error)
}

func (s *scriptedPlannerFunc) Model() string { return "qwen2.5-coder" }

func (s *scriptedPlannerFunc) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	if s.completeFn != nil {
		return s.completeFn(ctx, messages, tools, opts...)
	}
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
	}, nil
}
