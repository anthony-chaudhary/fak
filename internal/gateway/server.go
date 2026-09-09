package gateway

import (
	"net/http"
)

// AnthropicMessagesAdapter returns the native Anthropic Messages API protocol adapter
// configured for this Server.
func (s *Server) AnthropicMessagesAdapter() *AnthropicMessagesAdapter {
	return NewAnthropicMessagesAdapter(s)
}

// RegisterAnthropicMessagesRoutes registers the Anthropic Messages API endpoints
// (POST /v1/messages and POST /v1/messages/count_tokens) onto the provided ServeMux.
func (s *Server) RegisterAnthropicMessagesRoutes(mux *http.ServeMux) {
	s.AnthropicMessagesAdapter().RegisterRoutes(mux)
}

// HandleAnthropicMessages handles an Anthropic Messages request using the native adapter.
func (s *Server) HandleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	s.AnthropicMessagesAdapter().HandleMessages(w, r)
}

// HandleAnthropicCountTokens handles an Anthropic count_tokens request using the native adapter.
func (s *Server) HandleAnthropicCountTokens(w http.ResponseWriter, r *http.Request) {
	s.AnthropicMessagesAdapter().HandleCountTokens(w, r)
}

// AnthropicMessagesHandler returns an http.Handler that dispatches to the Anthropic Messages
// protocol adapter endpoints.
func (s *Server) AnthropicMessagesHandler() http.Handler {
	mux := http.NewServeMux()
	s.RegisterAnthropicMessagesRoutes(mux)
	return mux
}
