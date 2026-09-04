package openaiadapter

// Invariant: OpenAI adapter conversions are fail-closed and preserve request integrity.
// Guard: Requests with invalid app tokens, unmapped model aliases, or unsupported response formats
// are rejected deterministically before execution.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
)

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
type Message struct {
	Role      string     `json:"role"`
	Content   any        `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}
type Tool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		Parameters  map[string]any `json:"parameters,omitempty"`
	} `json:"function"`
}
type ResponseFormat struct {
	Type       string         `json:"type"`
	JSONSchema map[string]any `json:"json_schema,omitempty"`
}
type Request struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Stream         bool            `json:"stream,omitempty"`
	Tools          []Tool          `json:"tools,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}
type Result struct {
	Content   any
	ToolCalls []ToolCall
	Usage     Usage
}
type Execute func(context.Context, string, Request) (Result, error)
type Lifecycle interface {
	Readiness(context.Context) (any, error)
	Assets(context.Context) (any, error)
	Pressure(context.Context) (any, error)
	Handoff(context.Context) (any, error)
	Receipts(context.Context) (any, error)
}
type Server struct {
	Token     string
	Aliases   map[string]string
	Execute   Execute
	Lifecycle Lifecycle
}
type ErrorBody struct {
	Error APIError `json:"error"`
}
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

var ErrUnsupported = errors.New("openaiadapter: unsupported")

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.chat)
	return s.auth(mux)
}
func (s *Server) Listen(network, address string) (net.Listener, error) {
	if network != "unix" {
		host, _, e := net.SplitHostPort(address)
		if e != nil {
			return nil, e
		}
		ip := net.ParseIP(host)
		if ip == nil || !ip.IsLoopback() {
			return nil, errors.New("openaiadapter: only loopback or unix sockets are allowed")
		}
	}
	return net.Listen(network, address)
}
func (s *Server) auth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.Token == "" || r.Header.Get("Authorization") != "Bearer "+s.Token {
			writeErr(w, http.StatusUnauthorized, "authentication_error", "invalid_app_token", "invalid app token")
			return
		}
		next.ServeHTTP(w, r)
	})
}
func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	var req Request
	if e := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); e != nil {
		writeErr(w, 400, "invalid_request_error", "invalid_json", "invalid JSON request")
		return
	}
	model, ok := s.Aliases[req.Model]
	if !ok {
		writeErr(w, 400, "invalid_request_error", "model_not_found", "unknown app model alias")
		return
	}
	if req.ResponseFormat != nil && req.ResponseFormat.Type != "json_schema" && req.ResponseFormat.Type != "json_object" {
		writeErr(w, 400, "invalid_request_error", "unsupported_response_format", "response format is unsupported")
		return
	}
	result, e := s.Execute(r.Context(), model, req)
	if e != nil {
		if errors.Is(e, context.Canceled) {
			writeErr(w, 499, "cancelled_error", "request_cancelled", "request cancelled")
			return
		}
		if errors.Is(e, ErrUnsupported) {
			writeErr(w, 400, "invalid_request_error", "unsupported_feature", "feature is unsupported")
			return
		}
		writeErr(w, 500, "server_error", "execution_failed", "local execution failed")
		return
	}
	if req.Stream {
		s.stream(w, req, result)
		return
	}
	resp := map[string]any{"id": "fak-app-1", "object": "chat.completion", "model": req.Model, "choices": []any{map[string]any{"index": 0, "message": Message{Role: "assistant", Content: result.Content, ToolCalls: result.ToolCalls}, "finish_reason": finish(result)}}, "usage": result.Usage}
	writeJSON(w, 200, resp)
}
func finish(r Result) string {
	if len(r.ToolCalls) > 0 {
		return "tool_calls"
	}
	return "stop"
}
func (s *Server) stream(w http.ResponseWriter, req Request, res Result) {
	w.Header().Set("Content-Type", "text/event-stream")
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, 500, "server_error", "stream_unsupported", "streaming unavailable")
		return
	}
	chunks := []any{map[string]any{"id": "fak-app-1", "object": "chat.completion.chunk", "model": req.Model, "choices": []any{map[string]any{"index": 0, "delta": Message{Role: "assistant", Content: res.Content}}}}, map[string]any{"id": "fak-app-1", "object": "chat.completion.chunk", "model": req.Model, "choices": []any{map[string]any{"index": 0, "delta": Message{ToolCalls: res.ToolCalls}, "finish_reason": finish(res)}}, "usage": res.Usage}}
	for _, c := range chunks {
		b, _ := json.Marshal(c)
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
}
func writeErr(w http.ResponseWriter, status int, typ, code, msg string) {
	writeJSON(w, status, ErrorBody{Error: APIError{Message: msg, Type: typ, Code: code}})
}
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
func Compatibility() map[string]string {
	return map[string]string{"chat.completions": "supported", "streaming": "supported", "tool_calls": "supported", "response_format.json_schema": "supported", "cancellation": "supported", "usage": "supported", "images": "unsupported", "audio": "unsupported", "assistants": "unsupported", "fine_tuning": "unsupported"}
}

// FakKit exposes lifecycle truth separately instead of adding nonstandard OpenAI fields.
type FakKit struct{ Lifecycle Lifecycle }

func (f FakKit) Readiness(ctx context.Context) (any, error) { return f.Lifecycle.Readiness(ctx) }
func (f FakKit) Assets(ctx context.Context) (any, error)    { return f.Lifecycle.Assets(ctx) }
func (f FakKit) Pressure(ctx context.Context) (any, error)  { return f.Lifecycle.Pressure(ctx) }
func (f FakKit) Handoff(ctx context.Context) (any, error)   { return f.Lifecycle.Handoff(ctx) }
func (f FakKit) Receipts(ctx context.Context) (any, error)  { return f.Lifecycle.Receipts(ctx) }

var _ = strings.Builder{}
