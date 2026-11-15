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
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	if s.listSessions == nil {
		if s.table != nil {
			snap := s.table.Snapshot()
			sessions := make([]SessionState, 0, len(snap))
			for _, st := range snap {
				row := toGatewaySessionState(st)
				row.Activity = s.activity.actionCounts(row.TraceID)
				sessions = append(sessions, row)
			}
			rows := s.sessionTenantView(r, sessions)
			writeJSON(w, http.StatusOK, SessionListResponse{Sessions: rows, Count: len(rows)})
			return
		}
		writeErr(w, http.StatusNotFound, "session list is not configured")
		return
	}
	sessions := append([]SessionState(nil), s.listSessions(r.Context())...)
	for i := range sessions {
		// Explicit coverage or nonzero legacy counters belong to the provider;
		// their observation window cannot safely be summed with this registry.
		if sessions[i].Activity.IsZero() {
			sessions[i].Activity = s.activity.actionCounts(sessions[i].TraceID)
		}
	}
	rows := s.sessionTenantView(r, sessions)
	writeJSON(w, http.StatusOK, SessionListResponse{Sessions: rows, Count: len(rows)})
}

// sessionTenantView projects a session snapshot down to the rows the caller's
// authenticated tenant may see (#14442): under an armed keyset a stamped tenant
// keeps only the sessions it drove (its own trace-owner binding); everyone else
// — the operator scope, or a single-tenant deployment — keeps the whole table,
// exactly as before. Unbound rows are invisible to a stamped tenant: a session
// no tenant claims is never another tenant's to read.
func (s *Server) sessionTenantView(r *http.Request, sessions []SessionState) []SessionState {
	caller, armed := s.tenantPrincipalOf(r)
	if !armed {
		return sessions
	}
	kept := sessions[:0]
	for _, row := range sessions {
		owner, _ := s.traceOwnerOf(row.TraceID)
		if s.sessionOwnerAdmitted(caller, owner) {
			kept = append(kept, row)
		}
	}
	return kept
}
