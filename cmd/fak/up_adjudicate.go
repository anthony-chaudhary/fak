package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const (
	turnkeyAdjudicationUnavailable     = "tool-call adjudication is unavailable; refusing to skip adjudication"
	turnkeyAdjudicationUnavailableCode = "tool_call_adjudication_unavailable"
)

// newTurnkeyToolAdjudicator builds the gateway fak up uses only for the serve
// floor over proposed tool calls. gateway.New opens no listener and starts no
// loop (Serve does both), and no request is ever routed to it. fak up has no
// DOS workspace lease authority, so lease admission runs as serve's
// --workspace-admission-permissive: a known peer conflict is still refused.
func newTurnkeyToolAdjudicator(engineID string) (*gateway.Server, error) {
	return gateway.New(gateway.Config{
		EngineID:                     engineID,
		Logf:                         func(string, ...any) {},
		WorkspaceAdmissionPermissive: true,
	})
}

var (
	turnkeyDefaultAdjudicatorOnce sync.Once
	turnkeyDefaultAdjudicator     *gateway.Server
	turnkeyDefaultAdjudicatorErr  error
)

// toolAdjudicator returns the gateway that adjudicates this server's proposed
// tool calls. A server built outside startTurnkeyServer shares one process
// default, so no construction path returns a proposed call unadjudicated.
func (s *turnkeyServer) toolAdjudicator() (*gateway.Server, error) {
	if s.adjudicator != nil {
		return s.adjudicator, nil
	}
	turnkeyDefaultAdjudicatorOnce.Do(func() {
		turnkeyDefaultAdjudicator, turnkeyDefaultAdjudicatorErr = newTurnkeyToolAdjudicator("inkernel")
	})
	return turnkeyDefaultAdjudicator, turnkeyDefaultAdjudicatorErr
}

// turnkeyRequestTrace is the client's X-Trace-Id. Empty lets the gateway mint a
// fresh trace per call, as serve does for a headerless request.
func turnkeyRequestTrace(r *http.Request) string {
	return strings.TrimSpace(r.Header.Get("X-Trace-Id"))
}

func writeTurnkeyAdjudicationUnavailable(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusBadGateway)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
		"message": turnkeyAdjudicationUnavailable, "type": "server_error", "code": turnkeyAdjudicationUnavailableCode,
	}})
}
