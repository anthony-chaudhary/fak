package gateway

// sessions_list.go — the MULTI-session read side of the session-control surface:
// GET /v1/fak/sessions returns a point-in-time snapshot of EVERY live session's
// DRIVE state. It is the plural of handleFakSession's GET (one trace id): where the
// single-session route answers "what is THIS session doing", this answers "what is
// every session doing right now", turning the table's Snapshot (the scheduler's data
// structure) into a live operator surface — the read the dispatch loop reconstructs
// today from git commits + a process scan + a 0-byte-log heuristic
// (docs/dispatch-loop.md). The snapshot implementation is injected by cmd/fak so
// this package stays session-internals-blind, mirroring ObserveSession.

import (
	"context"
	"net/http"
)

// SessionListFunc is injected by the host CLI so the gateway can read a snapshot of
// every live session's DRIVE state without importing internal/session. It returns
// the sessions in the table's Snapshot order (by Priority ascending — lower yields
// first under contention — ties broken by most-recently-changed), so the wire order
// is already the order a scheduler/operator consumes. Nil disables GET
// /v1/fak/sessions (the same fail-closed posture as a nil ObserveSession).
type SessionListFunc func(context.Context) []SessionState

// SessionListResponse is the wire result of GET /v1/fak/sessions: a snapshot of every
// retained session's DRIVE state plus its count. Count is len(Sessions), surfaced so
// a client need not re-count and a "0 live sessions" reading is explicit rather than
// an empty array a reader might mistake for a transport error.
type SessionListResponse struct {
	Sessions []SessionState `json:"sessions"`
	Count    int            `json:"count"`
}

// handleFakSessions serves GET /v1/fak/sessions — the multi-session snapshot. It is
// GET-only (a read), and a nil listSessions injection ⇒ 404 (never a silent empty
// reading), the same fail-closed posture the trace/session routes take with no
// backing store. The exact "/v1/fak/sessions" path is registered distinctly from the
// "/v1/fak/session/" subtree, so a single-session request never lands here.
func (s *Server) handleFakSessions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeErr(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	if s.listSessions == nil {
		writeErr(w, http.StatusNotFound, "session list is not configured")
		return
	}
	sessions := s.listSessions(r.Context())
	writeJSON(w, http.StatusOK, SessionListResponse{Sessions: sessions, Count: len(sessions)})
}
