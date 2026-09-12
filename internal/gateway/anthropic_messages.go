package gateway

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// AnthropicMessagesAdapter is the native Anthropic Messages API protocol adapter
// for local Qwen ChatML serving. It translates between the Anthropic Messages wire
// protocol (as used by Claude Code and Anthropic SDKs) and local Qwen ChatML conventions.
type AnthropicMessagesAdapter struct {
	server *Server
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

func mintAnthropicToolUseID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "toolu_" + hex.EncodeToString(b[:])
}

// HandleMessages implements POST /v1/messages through the canonical served-request
// boundary, preserving incremental streaming and whole-turn tool adjudication.
func (a *AnthropicMessagesAdapter) HandleMessages(w http.ResponseWriter, r *http.Request) {
	if a == nil || a.server == nil {
		http.Error(w, "no server configured on adapter", http.StatusInternalServerError)
		return
	}
	// Wrap at the exported handler boundary so direct calls and registered routes
	// receive the same authentication and metrics as Server.Handler.
	a.server.withMetrics(a.server.withAuth(http.HandlerFunc(a.server.handleAnthropicMessages))).ServeHTTP(w, r)
}

// HandleCountTokens implements POST /v1/messages/count_tokens through the same
// authenticated token-count handler used by Server.Handler.
func (a *AnthropicMessagesAdapter) HandleCountTokens(w http.ResponseWriter, r *http.Request) {
	if a == nil || a.server == nil {
		http.Error(w, "no server configured on adapter", http.StatusInternalServerError)
		return
	}
	a.server.withMetrics(a.server.withAuth(http.HandlerFunc(a.server.handleAnthropicCountTokens))).ServeHTTP(w, r)
}
