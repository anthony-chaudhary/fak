package anthropicbridge

// bridge_test.go witnesses the request/response translation surface: the
// Anthropic -> OpenAI request decode+marshal (TestTranslateMessagesRequest),
// the tool-use id round-trip (TestToolUseRoundTrip), and the stop_reason
// vocabulary mapping (TestTranslateChatResponseStopReasons).

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// translateBody is the Anthropic Messages body the decode+marshal test runs:
// system string, thinking + tool_use assistant blocks, a tool_result user turn,
// tools, sampling options, and a tool_choice.
const translateBody = `{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "system": "Be terse.",
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "List files"}]},
    {"role": "assistant", "content": [
      {"type": "thinking", "thinking": "plan", "signature": "sig123"},
      {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {}}
    ]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"}]}
  ],
  "tools": [{"name": "ls", "description": "list", "input_schema": {"type": "object"}}],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop_sequences": ["END"],
  "stream": true,
  "tool_choice": {"type": "any"}
}`

func TestTranslateMessagesRequest(t *testing.T) {
	chatBody, req, err := TranslateMessagesRequest([]byte(translateBody))
	if err != nil {
		t.Fatalf("TranslateMessagesRequest: %v", err)
	}

	// --- decoded canonical request ---
	if req.Model != "deepseek-ai/DeepSeek-V4.1-Flash" {
		t.Errorf("req.Model = %q, want model passthrough", req.Model)
	}
	if req.System != "Be terse." {
		t.Errorf("req.System = %q, want %q", req.System, "Be terse.")
	}
	if len(req.Messages) != 4 {
		t.Fatalf("len(req.Messages) = %d, want 4 (system + user + assistant + tool)", len(req.Messages))
	}
	if sys := req.Messages[0]; sys.Role != "system" || sys.Content != "Be terse." {
		t.Errorf("leading message = {%s %q}, want folded system message", sys.Role, sys.Content)
	}
	if req.Messages[1].Role != "user" || req.Messages[1].Content != "List files" {
		t.Errorf("user message = {%s %q}, want {user List files}", req.Messages[1].Role, req.Messages[1].Content)
	}
	asst := req.Messages[2]
	if asst.Role != "assistant" {
		t.Errorf("assistant role = %q", asst.Role)
	}
	if len(asst.ToolCalls) != 1 || asst.ToolCalls[0].ID != "toolu_1" {
		t.Errorf("assistant ToolCalls = %+v, want one call id toolu_1", asst.ToolCalls)
	}
	if asst.Thinking != "plan" || asst.ThinkingSignature != "sig123" {
		t.Errorf("assistant thinking = %q/%q, want plan/sig123", asst.Thinking, asst.ThinkingSignature)
	}
	tool := req.Messages[3]
	if tool.Role != "tool" || tool.ToolCallID != "toolu_1" {
		t.Errorf("tool message = {%s tool_call_id=%q}, want {tool toolu_1}", tool.Role, tool.ToolCallID)
	}
	if tool.Name != "ls" {
		t.Errorf("tool message Name = %q, want name re-bound from the matching assistant call", tool.Name)
	}
	if tool.Content != "a.txt" {
		t.Errorf("tool message Content = %q, want a.txt", tool.Content)
	}
	if len(req.Tools) != 1 {
		t.Fatalf("len(req.Tools) = %d, want 1", len(req.Tools))
	}
	if td := req.Tools[0]; td.Function.Name != "ls" || td.Function.Description != "list" {
		t.Errorf("tool def = %s/%s, want ls/list", td.Function.Name, td.Function.Description)
	} else if string(td.Function.Parameters) != `{"type": "object"}` {
		t.Errorf("tool def parameters = %s, want raw input_schema passthrough (verbatim bytes)", td.Function.Parameters)
	}
	if string(req.ToolChoice) != `{"type": "any"}` {
		t.Errorf("req.ToolChoice = %s, want the raw Anthropic tool_choice captured (verbatim bytes)", req.ToolChoice)
	}

	// --- encoded OpenAI chat body ---
	var body struct {
		Model    string `json:"model"`
		Messages []struct {
			Role       string `json:"role"`
			Content    string `json:"content"`
			ToolCallID string `json:"tool_call_id"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tool_calls"`
			Thinking          string `json:"thinking"`
			ThinkingSignature string `json:"thinking_signature"`
		} `json:"messages"`
		Tools []struct {
			Type     string `json:"type"`
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		} `json:"tools"`
		ToolChoice    json.RawMessage `json:"tool_choice"`
		MaxTokens     int             `json:"max_tokens"`
		Temperature   *float64        `json:"temperature"`
		TopP          *float64        `json:"top_p"`
		Stop          []string        `json:"stop"`
		Stream        bool            `json:"stream"`
		StreamOptions *struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	if err := json.Unmarshal(chatBody, &body); err != nil {
		t.Fatalf("chat body: %v", err)
	}
	if body.Model != "deepseek-ai/DeepSeek-V4.1-Flash" {
		t.Errorf("chat model = %q, want passthrough", body.Model)
	}
	if len(body.Messages) != 4 {
		t.Fatalf("chat messages = %d, want 4", len(body.Messages))
	}
	if body.Messages[0].Role != "system" || body.Messages[0].Content != "Be terse." {
		t.Errorf("chat messages[0] = {%s %q}, want leading system message", body.Messages[0].Role, body.Messages[0].Content)
	}
	chatAsst := body.Messages[2]
	if len(chatAsst.ToolCalls) != 1 || chatAsst.ToolCalls[0].ID != "toolu_1" || chatAsst.ToolCalls[0].Function.Name != "ls" {
		t.Errorf("chat assistant tool_calls = %+v, want toolu_1/ls", chatAsst.ToolCalls)
	}
	if chatAsst.Thinking != "plan" || chatAsst.ThinkingSignature != "sig123" {
		t.Errorf("chat assistant thinking = %q/%q, want the thinking block carried on the wire", chatAsst.Thinking, chatAsst.ThinkingSignature)
	}
	chatTool := body.Messages[3]
	if chatTool.Role != "tool" {
		t.Errorf("chat tool message role = %q, want tool", chatTool.Role)
	}
	if chatTool.ToolCallID != "toolu_1" {
		t.Errorf("chat tool message tool_call_id = %q, want toolu_1", chatTool.ToolCallID)
	}
	if len(body.Tools) != 1 || body.Tools[0].Type != "function" || body.Tools[0].Function.Name != "ls" {
		t.Errorf("chat tools = %+v, want one function tool ls", body.Tools)
	}
	if string(body.ToolChoice) != `"required"` {
		t.Errorf("chat tool_choice = %s, want \"required\" (any -> required)", body.ToolChoice)
	}
	if body.MaxTokens != 512 {
		t.Errorf("chat max_tokens = %d, want 512", body.MaxTokens)
	}
	if body.Temperature == nil || *body.Temperature != 0.7 {
		t.Errorf("chat temperature = %v, want 0.7", body.Temperature)
	}
	if body.TopP == nil || *body.TopP != 0.9 {
		t.Errorf("chat top_p = %v, want 0.9", body.TopP)
	}
	if len(body.Stop) != 1 || body.Stop[0] != "END" {
		t.Errorf("chat stop = %v, want [END]", body.Stop)
	}
	if !body.Stream {
		t.Errorf("chat stream = false, want true")
	}
	if body.StreamOptions == nil || !body.StreamOptions.IncludeUsage {
		t.Errorf("chat stream_options = %v, want include_usage (a streamed turn still reports usage)", body.StreamOptions)
	}
}

func TestToolUseRoundTrip(t *testing.T) {
	chatResp := `{"choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"ls","arguments":"{\"path\":\".\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`
	messagesBody, err := TranslateChatResponse([]byte(chatResp), ResponseOptions{ID: "msg_rt", Model: "m"})
	if err != nil {
		t.Fatalf("TranslateChatResponse: %v", err)
	}
	var resp struct {
		Content []struct {
			Type  string          `json:"type"`
			ID    string          `json:"id"`
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(messagesBody, &resp); err != nil {
		t.Fatalf("messages body: %v", err)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1 (no prose, so no text block)", len(resp.Content))
	}
	blk := resp.Content[0]
	if blk.Type != "tool_use" || blk.ID != "call_1" || blk.Name != "ls" {
		t.Errorf("content[0] = {%s id=%s name=%s}, want tool_use call_1 ls", blk.Type, blk.ID, blk.Name)
	}
	if strings.TrimSpace(string(blk.Input)) != `{"path":"."}` {
		t.Errorf("content[0].input = %s, want the parsed argument object", blk.Input)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage = %d/%d, want 10/5", resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}

	// The id round-trip: a follow-up request carrying the tool_result keyed by the
	// SAME tool_use id decodes to a role=tool chat message bound to call_1.
	followUp := `{"model":"m","max_tokens":64,"messages":[
		{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"ls","input":{"path":"."}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"files"}]}
	]}`
	chatBody, _, err := TranslateMessagesRequest([]byte(followUp))
	if err != nil {
		t.Fatalf("TranslateMessagesRequest (follow-up): %v", err)
	}
	var req2 struct {
		Messages []struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(chatBody, &req2); err != nil {
		t.Fatalf("follow-up chat body: %v", err)
	}
	found := false
	for _, m := range req2.Messages {
		if m.Role == "tool" {
			found = true
			if m.ToolCallID != "call_1" {
				t.Errorf("tool message tool_call_id = %q, want call_1 (the id round-trip)", m.ToolCallID)
			}
		}
	}
	if !found {
		t.Errorf("no role=tool message in the translated follow-up body: %s", chatBody)
	}
}

func TestTranslateChatResponseStopReasons(t *testing.T) {
	for _, tc := range []struct{ finish, want string }{
		{"length", "max_tokens"},
		{"stop", "end_turn"},
	} {
		chat := fmt.Sprintf(`{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":%q}]}`, tc.finish)
		body, err := TranslateChatResponse([]byte(chat), ResponseOptions{ID: "msg_sr", Model: "m"})
		if err != nil {
			t.Fatalf("finish %q: %v", tc.finish, err)
		}
		var resp struct {
			StopReason string `json:"stop_reason"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			t.Fatalf("finish %q: body: %v", tc.finish, err)
		}
		if resp.StopReason != tc.want {
			t.Errorf("finish %q -> stop_reason %q, want %q", tc.finish, resp.StopReason, tc.want)
		}
	}
}

// The full Anthropic tool_choice vocabulary maps onto the OpenAI Chat
// Completions one; absent/unknown keeps the encoder default (nil).
func TestToolChoiceForChat(t *testing.T) {
	for _, tc := range []struct{ raw, want string }{
		{`{"type":"auto"}`, `"auto"`},
		{`{"type":"any"}`, `"required"`},
		{`{"type":"tool","name":"ls"}`, `{"type":"function","function":{"name":"ls"}}`},
		{`"auto"`, `"auto"`},
		{`"any"`, `"required"`},
		{`"none"`, `"none"`},
		{``, ``},
		{`null`, ``},
		{`{"type":"weird"}`, ``},
	} {
		got := ToolChoiceForChat(json.RawMessage(tc.raw))
		if string(got) != tc.want {
			t.Errorf("ToolChoiceForChat(%s) = %s, want %s", tc.raw, got, tc.want)
		}
	}
}
