package gateway

import (
	"net/http"
)

// session_subscribe.go — the per-session subscribe / re-attach op (#2767, epic
// #2753): the A2A tasks/resubscribe equivalent for a served session. Where
// /v1/fak/session/changes tails EVERY session's drive-state revisions,
// GET /v1/fak/session/{trace_id}/subscribe drains ONE session's revisions after
// the client's ?since= cursor — so a controller that disconnected re-attaches to
// the exact live session by its trace handle and resumes the event stream where
// it left off, then issues control ops on the same handle (the existing POST
// /v1/fak/session/{trace_id}/{verb} surface; no new auth model).
//
// The trace handle is the STABLE EXTERNAL address this op keys on: a
// gateway-minted gw-<n> id is already exposed by /v1/fak/sessions and every
// change event, and subscribe is what turns it from a display id into a
// re-attach key. This is a LIVE re-attach — a cursor-resumed tail of the running
// session's event feed — not the cold resume (rehydrate) path; the two must not
// be conflated.
//
// Loss across the reconnect boundary is made legible, never silent: the feed's
// ring is bounded, so a lapsed client whose cursor predates the oldest retained
// event gets complete=false (some events fell off before it re-attached) and
// re-syncs to head, exactly like the global feed's Seq-gap contract.

// SessionSubscribeResponse is the wire result of the subscribe/re-attach drain:
// the one session's revision events after the cursor, the client's next cursor
// (the feed's GLOBAL Seq — the same drain cursor the /session/changes stream
// uses, so one cursor vocabulary serves both), and whether that tail is complete
// (no retained-window gap between the presented cursor and the events returned).
type SessionSubscribeResponse struct {
	TraceID string               `json:"trace_id"`
	Events  []SessionChangeEvent `json:"events"`
	Cursor  uint64               `json:"cursor"`
	// Complete reports the resume was lossless: every event since the client's
	// cursor is still retained. False means the bounded ring trimmed events the
	// client never saw — revisions may be missing and the client should re-read
	// the session's current state (GET /v1/fak/session/{trace_id}) to re-sync.
	Complete bool `json:"complete"`
}

// drainTrace returns the retained events for one trace with Seq > sinceSeq, the
// highest Seq now known (the next cursor — global, shared with drain), and
// whether the tail is complete: no event (any trace) with Seq > sinceSeq has
// been trimmed from the ring, so nothing the client missed is unaccounted for.
// Completeness is judged on the GLOBAL ring, not the filtered view — a trimmed
// window cannot say which traces its lost events carried, so the only honest
// verdict is "the window since your cursor is intact" or "it is not".
func (f *sessionFeed) drainTrace(trace string, sinceSeq uint64) (events []SessionChangeEvent, cursor uint64, complete bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	cursor = sinceSeq
	// oldestRetained is the Seq of the oldest event still in the ring; an empty
	// ring retains nothing, so the window is intact only for a client already at
	// head (sinceSeq >= f.seq).
	oldestRetained := f.seq + 1
	if len(f.ring) > 0 {
		oldestRetained = f.ring[0].Seq
	}
	complete = sinceSeq+1 >= oldestRetained
	for _, ev := range f.ring {
		if ev.Seq > sinceSeq && ev.TraceID == trace {
			events = append(events, ev)
		}
		if ev.Seq > cursor {
			cursor = ev.Seq
		}
	}
	return events, cursor, complete
}

// handleFakSessionSubscribe serves GET /v1/fak/session/{trace_id}/subscribe — the
// re-attach drain (#2767). ?since= is the Seq of the last event the client saw
// (0 = the whole retained tail), the same cursor grammar as /session/changes.
// A server built without the feed 404s (fail-closed, like an un-injected
// observe), never an empty clean-looking tail.
func (s *Server) handleFakSessionSubscribe(w http.ResponseWriter, r *http.Request, traceID string) {
	if s.sessionFeed == nil {
		writeErr(w, http.StatusNotFound, "session change feed is not configured")
		return
	}
	var since uint64
	if v := r.URL.Query().Get("since"); v != "" {
		var n uint64
		for _, c := range v {
			if c < '0' || c > '9' {
				writeErr(w, http.StatusBadRequest, "since must be a non-negative integer")
				return
			}
			n = n*10 + uint64(c-'0')
		}
		since = n
	}
	events, cursor, complete := s.sessionFeed.drainTrace(traceID, since)
	writeJSON(w, http.StatusOK, SessionSubscribeResponse{
		TraceID:  traceID,
		Events:   events,
		Cursor:   cursor,
		Complete: complete,
	})
}
