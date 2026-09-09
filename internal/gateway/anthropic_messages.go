package gateway

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// AnthropicMessagesAdapter is the native Anthropic Messages API protocol adapter
// for local Qwen ChatML serving. It translates between the Anthropic Messages wire
// protocol (as used by Claude Code and Anthropic SDKs) and local Qwen ChatML conventions.
type AnthropicMessagesAdapter struct {
	server  *Server
	planner agent.Planner
}

// NewAnthropicMessagesAdapter constructs an adapter wired to the provided Server.
func NewAnthropicMessagesAdapter(s *Server) *AnthropicMessagesAdapter {
	return &AnthropicMessagesAdapter{server: s}
}

// RegisterRoutes registers the Anthropic Messages API endpoints:
// - POST /v1/messages
// - POST /v1/messages/count_tokens
func (a *AnthropicMessagesAdapter) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/messages", a.HandleMessages)
	mux.HandleFunc("/v1/messages/count_tokens", a.HandleCountTokens)
}

// Handler returns an http.Handler that routes Anthropic Messages requests.
func (a *AnthropicMessagesAdapter) Handler() http.Handler {
	mux := http.NewServeMux()
	a.RegisterRoutes(mux)
	return mux
}

// FormatQwenToolCall formats a function call into Qwen ChatML <tool_call> syntax.
func FormatQwenToolCall(name string, args string) string {
	cleanArgs := strings.TrimSpace(args)
	if cleanArgs == "" {
		cleanArgs = "{}"
	}
	return fmt.Sprintf("<tool_call>\n{\"name\":\"%s\",\"arguments\":%s}\n</tool_call>", name, cleanArgs)
}

// FormatQwenToolResponse formats a tool result into Qwen ChatML <tool_response> syntax.
func FormatQwenToolResponse(content string) string {
	return fmt.Sprintf("<tool_response>\n%s\n</tool_response>", strings.TrimSpace(content))
}

// FormatQwenNamedToolResponse formats a named tool result into Qwen ChatML <tool_response> syntax.
func FormatQwenNamedToolResponse(name, content string) string {
	if name != "" {
		return fmt.Sprintf("<tool_response>\n%s: %s\n</tool_response>", name, strings.TrimSpace(content))
	}
	return FormatQwenToolResponse(content)
}

// TranslateToQwenChatML maps canonical transcript messages to Qwen ChatML conventions:
// - Prior assistant tool calls are represented with <tool_call> tags in assistant content.
// - Inbound tool results (RoleTool) are represented as <tool_response> blocks in user messages.
func TranslateToQwenChatML(messages []agent.Message) []agent.Message {
	out := make([]agent.Message, 0, len(messages))
	for _, m := range messages {
		switch m.Role {
		case agent.RoleAssistant:
			content := m.Content
			for _, tc := range m.ToolCalls {
				tcBlock := FormatQwenToolCall(tc.Function.Name, tc.Function.Arguments)
				if content != "" {
					content += "\n" + tcBlock
				} else {
					content = tcBlock
				}
			}
			out = append(out, agent.Message{
				Role:      agent.RoleAssistant,
				Content:   content,
				ToolCalls: m.ToolCalls,
			})
		case agent.RoleTool:
			respText := FormatQwenToolResponse(m.Content)
			out = append(out, agent.Message{
				Role:       agent.RoleUser,
				Content:    respText,
				Name:       m.Name,
				ToolCallID: m.ToolCallID,
			})
		default:
			out = append(out, m)
		}
	}
	return out
}

// ParseQwenToolCalls inspects model output text for Qwen ChatML <tool_call> syntax
// (both Hermes JSON format and Qwen XML <function=...><parameter=...> format).
// It converts any detected tool calls into structured ToolCalls with "toolu_" IDs,
// and strips the <tool_call> tags from the content text so only natural language prose remains.
func ParseQwenToolCalls(content string) (string, []agent.ToolCall) {
	msg := agent.LiftTextToolCalls(agent.Message{Role: agent.RoleAssistant, Content: content})
	prose := strings.TrimSpace(msg.Content)
	calls := make([]agent.ToolCall, len(msg.ToolCalls))
	for i, tc := range msg.ToolCalls {
		calls[i] = tc
		if !strings.HasPrefix(calls[i].ID, "toolu_") {
			calls[i].ID = mintAnthropicToolUseID()
		}
		args := strings.TrimSpace(calls[i].Function.Arguments)
		if args == "" {
			calls[i].Function.Arguments = "{}"
		} else {
			var testMap map[string]any
			if err := json.Unmarshal([]byte(args), &testMap); err != nil {
				var unquoted string
				if err2 := json.Unmarshal([]byte(args), &unquoted); err2 == nil {
					if err3 := json.Unmarshal([]byte(unquoted), &testMap); err3 == nil {
						calls[i].Function.Arguments = unquoted
					}
				}
			}
		}
	}
	return prose, calls
}

func mintAnthropicMessageID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "msg_" + hex.EncodeToString(b[:])
}

func mintAnthropicToolUseID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "toolu_" + hex.EncodeToString(b[:])
}

// HandleMessages implements POST /v1/messages.
func (a *AnthropicMessagesAdapter) HandleMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	raw, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "error reading request body", http.StatusBadRequest)
		return
	}

	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		http.Error(w, "invalid Anthropic Messages request: "+err.Error(), http.StatusBadRequest)
		return
	}

	planner := a.planner
	if planner == nil && a.server != nil {
		planner = a.server.planner
	}
	if planner == nil {
		http.Error(w, "no planner configured on server", http.StatusInternalServerError)
		return
	}

	model := req.Model
	if model == "" && a.server != nil && a.server.model != "" {
		model = a.server.model
	}
	if model == "" {
		model = "qwen2.5-coder"
	}

	trace := r.Header.Get("X-Trace-Id")
	if trace == "" && a.server != nil {
		trace = a.server.traceFor("")
	}
	if trace == "" {
		trace = "anthropic-trace"
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096
	}

	opts := []agent.SampleOpt{
		agent.WithMaxTokens(maxTokens),
		agent.WithModel(model),
	}
	if req.Temperature != 0 {
		opts = append(opts, agent.WithTemperature(&req.Temperature))
	}
	if req.TopP != nil {
		opts = append(opts, agent.WithTopP(req.TopP))
	}
	if req.TopK != nil {
		opts = append(opts, agent.WithTopK(req.TopK))
	}
	if len(req.StopSequences) > 0 {
		opts = append(opts, agent.WithStop(req.StopSequences))
	}

	inputTokens := agent.EstimateAnthropicTokens(req)
	if inputTokens <= 0 {
		if len(raw) > 0 {
			inputTokens = len(raw) / 4
		}
		if inputTokens <= 0 {
			inputTokens = 1
		}
	}

	if req.Stream {
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)

		send := anthropicSSESender(w, flusher)
		msgID := mintAnthropicMessageID()

		send("message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id":            msgID,
				"type":          "message",
				"role":          "assistant",
				"model":         model,
				"content":       []any{},
				"stop_reason":   nil,
				"stop_sequence": nil,
				"usage": map[string]any{
					"input_tokens":  inputTokens,
					"output_tokens": 0,
				},
			},
		})

		var comp *agent.Completion
		var streamErr error
		if sp, ok := planner.(agent.StreamingPlanner); ok && sp.StreamingSupported() {
			var buf strings.Builder
			sink := func(frag string) error {
				buf.WriteString(frag)
				return nil
			}
			comp, streamErr = sp.CompleteStream(r.Context(), sink, req.Messages, req.Tools, opts...)
			if comp == nil && streamErr == nil {
				comp = &agent.Completion{
					Message: agent.Message{
						Role:    agent.RoleAssistant,
						Content: buf.String(),
					},
					FinishReason: "stop",
				}
			} else if comp != nil && comp.Message.Content == "" && buf.Len() > 0 {
				comp.Message.Content = buf.String()
			}
		} else {
			comp, streamErr = planner.Complete(r.Context(), req.Messages, req.Tools, opts...)
		}

		if streamErr != nil {
			send("error", map[string]any{
				"type": "error",
				"error": map[string]any{
					"type":    "api_error",
					"message": streamErr.Error(),
				},
			})
			return
		}

		prose, toolCalls := ParseQwenToolCalls(comp.Message.Content)
		if len(toolCalls) == 0 && len(comp.Message.ToolCalls) > 0 {
			toolCalls = make([]agent.ToolCall, len(comp.Message.ToolCalls))
			for i, tc := range comp.Message.ToolCalls {
				toolCalls[i] = tc
				if !strings.HasPrefix(toolCalls[i].ID, "toolu_") {
					toolCalls[i].ID = mintAnthropicToolUseID()
				}
			}
		}

		if a.server != nil && a.server.k != nil && len(toolCalls) > 0 {
			kept, _, _, _, _, _ := a.server.adjudicateProposedTurn(r.Context(), agent.Message{
				Role:      agent.RoleAssistant,
				Content:   prose,
				ToolCalls: toolCalls,
			}, trace)
			toolCalls = kept
		}

		idx := 0
		if prose != "" {
			send("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": idx,
				"content_block": map[string]any{
					"type": "text",
					"text": "",
				},
			})
			send("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{
					"type": "text_delta",
					"text": prose,
				},
			})
			send("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": idx,
			})
			idx++
		}

		for _, tc := range toolCalls {
			args := tc.Function.Arguments
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			send("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": idx,
				"content_block": map[string]any{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Function.Name,
					"input": map[string]any{},
				},
			})
			send("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{
					"type":         "input_json_delta",
					"partial_json": args,
				},
			})
			send("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": idx,
			})
			idx++
		}

		stopReason := "end_turn"
		if len(toolCalls) > 0 {
			stopReason = "tool_use"
		} else if strings.ToLower(comp.FinishReason) == "length" {
			stopReason = "max_tokens"
		}

		outTokens := comp.Usage.CompletionTokens
		if outTokens <= 0 {
			outTokens = len(prose) / 4
			if outTokens <= 0 {
				outTokens = 1
			}
		}

		send("message_delta", map[string]any{
			"type": "message_delta",
			"delta": map[string]any{
				"stop_reason":   stopReason,
				"stop_sequence": nil,
			},
			"usage": map[string]any{
				"output_tokens": outTokens,
			},
		})
		send("message_stop", map[string]any{
			"type": "message_stop",
		})
		return
	}

	// Non-streaming path
	comp, err := planner.Complete(r.Context(), req.Messages, req.Tools, opts...)
	if err != nil {
		http.Error(w, "upstream model error: "+err.Error(), http.StatusBadGateway)
		return
	}

	prose, toolCalls := ParseQwenToolCalls(comp.Message.Content)
	if len(toolCalls) == 0 && len(comp.Message.ToolCalls) > 0 {
		toolCalls = make([]agent.ToolCall, len(comp.Message.ToolCalls))
		for i, tc := range comp.Message.ToolCalls {
			toolCalls[i] = tc
			if !strings.HasPrefix(toolCalls[i].ID, "toolu_") {
				toolCalls[i].ID = mintAnthropicToolUseID()
			}
		}
	}

	if a.server != nil && a.server.k != nil && len(toolCalls) > 0 {
		kept, _, _, _, _, _ := a.server.adjudicateProposedTurn(r.Context(), agent.Message{
			Role:      agent.RoleAssistant,
			Content:   prose,
			ToolCalls: toolCalls,
		}, trace)
		toolCalls = kept
	}

	var blocks []agent.AnthropicBlockOut
	if prose != "" {
		blocks = append(blocks, agent.AnthropicBlockOut{
			Type: "text",
			Text: prose,
		})
	}
	for _, tc := range toolCalls {
		args := tc.Function.Arguments
		if strings.TrimSpace(args) == "" {
			args = "{}"
		}
		blocks = append(blocks, agent.AnthropicBlockOut{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(args),
		})
	}

	stopReason := "end_turn"
	if len(toolCalls) > 0 {
		stopReason = "tool_use"
	} else if strings.ToLower(comp.FinishReason) == "length" {
		stopReason = "max_tokens"
	}

	outTokens := comp.Usage.CompletionTokens
	if outTokens <= 0 {
		outTokens = len(prose) / 4
		if outTokens <= 0 {
			outTokens = 1
		}
	}

	resp := anthropicMessageResponse{
		ID:         mintAnthropicMessageID(),
		Type:       "message",
		Role:       "assistant",
		Model:      model,
		Content:    blocks,
		StopReason: stopReason,
		Usage: anthropicUsage{
			InputTokens:  inputTokens,
			OutputTokens: outTokens,
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

// HandleCountTokens implements POST /v1/messages/count_tokens.
func (a *AnthropicMessagesAdapter) HandleCountTokens(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	raw, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "error reading request body", http.StatusBadRequest)
		return
	}

	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		http.Error(w, "invalid Anthropic Messages request: "+err.Error(), http.StatusBadRequest)
		return
	}

	tokens := agent.EstimateAnthropicTokens(req)
	if tokens <= 0 {
		if len(raw) > 0 {
			tokens = len(raw) / 4
		}
		if tokens <= 0 {
			tokens = 1
		}
	}

	writeJSON(w, http.StatusOK, map[string]int{
		"input_tokens": tokens,
	})
}
