package gateway

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"net/http"
	"strings"
)

type mcpAuthenticatedLaunchKey struct{}

// handleAuthenticatedMCPHTTP authenticates the guard launch before the MCP
// dispatcher derives its request context. It grants provenance, not lease
// ownership; the admission provider must independently establish any lease.
func (s *Server) handleAuthenticatedMCPHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := s.authenticateMCPSession(r.Context(), r.Header.Get("X-Fak-MCP-Session"))
	s.handleMCPHTTP(w, r.WithContext(ctx))
}

// authenticateMCPSession accepts only the dedicated per-launch bearer emitted
// in the generated MCP config. Request trace and principal text are not proof
// of launch origin. The gateway's separate Authorization header remains intact.
func (s *Server) authenticateMCPSession(ctx context.Context, authorization string) context.Context {
	if s == nil || !s.mcpSessionAuthEnabled {
		return ctx
	}
	token, ok := strings.CutPrefix(authorization, "Bearer ")
	if !ok || token == "" {
		return ctx
	}
	got := sha256.Sum256([]byte(token))
	if subtle.ConstantTimeCompare(got[:], s.mcpSessionBearerDigest[:]) != 1 {
		return ctx
	}
	return context.WithValue(ctx, mcpAuthenticatedLaunchKey{}, true)
}

func authenticatedMCPLaunch(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	authenticated, _ := ctx.Value(mcpAuthenticatedLaunchKey{}).(bool)
	return authenticated
}
