package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
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
//
// #4310 adds the HELD form of that same drain: with ?wait=<duration> the request
// parks until a revision for this session lands past the cursor, so a controller
// is PUSHED the next revision instead of re-polling for it. It is additive, not a
// second protocol — same route, same cursor grammar, same reply shape, and the
// one-shot drain (no ?wait=) stays the resume primitive. The two invariants that
// make it safe: the hold arms in the SAME lock hold as its drain (armTrace), so
// no revision can slip through the gap between "nothing yet" and "now waiting";
// and the hold is bounded, so an elapsed wait answers an empty tail at the head
// cursor rather than occupying a connection forever.

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
	return f.drainTraceLocked(trace, sinceSeq)
}

// drainTraceLocked is drainTrace's body with f.mu already held, so the held
// attach can drain and arm inside ONE critical section (see armTrace).
func (f *sessionFeed) drainTraceLocked(trace string, sinceSeq uint64) (events []SessionChangeEvent, cursor uint64, complete bool) {
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

// armTrace is the held-attach primitive (#4310): it drains the trace and, ONLY
// when that drain is empty, hands back the feed's broadcast channel — created
// under the SAME lock hold as the drain. That single critical section is what
// closes the race a poll loop cannot close: no revision can land between "I saw
// nothing" and "I am waiting", because a publisher must take this same mutex
// both to append and to swap the broadcast out. A non-empty drain returns a nil
// channel — there is already something to answer with, so there is nothing to
// wait for.
func (f *sessionFeed) armTrace(trace string, sinceSeq uint64) (events []SessionChangeEvent, cursor uint64, complete bool, wake <-chan struct{}) {
	f.mu.Lock()
	defer f.mu.Unlock()
	events, cursor, complete = f.drainTraceLocked(trace, sinceSeq)
	if len(events) > 0 {
		return events, cursor, complete, nil
	}
	if f.wake == nil {
		f.wake = make(chan struct{})
	}
	return nil, cursor, complete, f.wake
}

// maxSubscribeWait caps how long a held attach parks the request. The hold is
// BOUNDED on purpose: the reply's cursor makes the follow loop trivially
// resumable, so answering an empty tail early costs a client nothing, while an
// unbounded hold would outlive the proxies and load balancers in front of
// `fak serve`.
const maxSubscribeWait = 55 * time.Second

// handleFakSessionSubscribe serves GET /v1/fak/session/{trace_id}/subscribe — the
// re-attach drain (#2767) and its held/streaming form (#4310). ?since= is the Seq
// of the last event the client saw (0 = the whole retained tail), the same cursor
// grammar as /session/changes; ?wait= optionally HOLDS the request until a
// revision for this session lands, so a controller is pushed the next revision
// instead of re-polling for it. A server built without the feed 404s
// (fail-closed, like an un-injected observe), never an empty clean-looking tail.
func (s *Server) handleFakSessionSubscribe(w http.ResponseWriter, r *http.Request, traceID string) {
	if s.sessionFeed == nil {
		writeErr(w, http.StatusNotFound, "session change feed is not configured")
		return
	}
	since, ok := subscribeSince(w, r)
	if !ok {
		return
	}

	isSSE := strings.Contains(r.Header.Get("Accept"), "text/event-stream") ||
		r.URL.Query().Get("format") == "sse" ||
		r.URL.Query().Get("stream") == "sse" ||
		r.URL.Query().Get("stream") == "true"

	isNDJSON := strings.Contains(r.Header.Get("Accept"), "application/x-ndjson") ||
		r.URL.Query().Get("format") == "ndjson" ||
		r.URL.Query().Get("stream") == "ndjson"

	if isSSE || isNDJSON {
		if isSSE {
			w.Header().Set("Content-Type", "text/event-stream")
		} else {
			w.Header().Set("Content-Type", "application/x-ndjson")
		}
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)
		if flusher != nil {
			flusher.Flush()
		}

		emit := func(ev SessionChangeEvent) error {
			b, err := json.Marshal(ev)
			if err != nil {
				return err
			}
			if isSSE {
				if _, err := fmt.Fprintf(w, "event: transition\ndata: %s\n\n", b); err != nil {
					return err
				}
			} else {
				if _, err := fmt.Fprintf(w, "%s\n", b); err != nil {
					return err
				}
			}
			if flusher != nil {
				flusher.Flush()
			}
			return nil
		}

		cursor := since
		events, nextCursor, _ := s.sessionFeed.drainTrace(traceID, cursor)
		cursor = nextCursor
		for _, ev := range events {
			if err := emit(ev); err != nil {
				return
			}
		}

		for {
			events, nextCursor, _, wake := s.sessionFeed.armTrace(traceID, cursor)
			cursor = nextCursor
			for _, ev := range events {
				if err := emit(ev); err != nil {
					return
				}
			}
			if wake == nil {
				continue
			}
			select {
			case <-r.Context().Done():
				return
			case <-wake:
				events, nextCursor, _ := s.sessionFeed.drainTrace(traceID, cursor)
				cursor = nextCursor
				for _, ev := range events {
					if err := emit(ev); err != nil {
						return
					}
				}
			}
		}
	}

	wait, ok := subscribeWait(w, r)
	if !ok {
		return
	}
	if wait <= 0 {
		// The one-shot drain: byte-identical to the pre-#4310 wire shape. It stays
		// the resume primitive; the hold is strictly additive on the same cursor.
		events, cursor, complete := s.sessionFeed.drainTrace(traceID, since)
		writeSessionSubscribe(w, traceID, events, cursor, complete)
		return
	}
	s.holdSessionSubscribe(w, r, traceID, since, wait)
}

// holdSessionSubscribe serves the held (long-poll) attach: park until a revision
// for THIS trace lands past the cursor, the wait budget elapses, or the client
// hangs up. The reply shape is the drain's, unchanged — an elapsed hold answers
// an empty tail at the head cursor — so a follow loop is exactly "call again
// with the cursor you were handed", and a client that cannot hold keeps working
// against the same endpoint by omitting ?wait=.
func (s *Server) holdSessionSubscribe(w http.ResponseWriter, r *http.Request, traceID string, since uint64, wait time.Duration) {
	budget := time.NewTimer(wait)
	defer budget.Stop()
	for {
		events, cursor, complete, wake := s.sessionFeed.armTrace(traceID, since)
		if wake == nil {
			writeSessionSubscribe(w, traceID, events, cursor, complete)
			return
		}
		select {
		case <-wake:
			// A revision landed — but the broadcast is feed-global, so it may have
			// been another session's. Re-drain and, if it was not ours, re-arm and
			// keep holding until the budget runs out.
		case <-budget.C:
			// Nothing for this trace within the budget. Drain once more so a
			// revision that raced the timer is still delivered rather than
			// deferred to the client's next attach, then answer the tail.
			events, cursor, complete = s.sessionFeed.drainTrace(traceID, since)
			writeSessionSubscribe(w, traceID, events, cursor, complete)
			return
		case <-r.Context().Done():
			// The controller hung up mid-hold. Write nothing: its cursor never
			// advanced, so its next attach resumes from exactly where it was —
			// the disconnect costs no events.
			return
		}
	}
}

// subscribeSince reads the ?since= cursor: the Seq of the last event the client
// saw, 0 (or absent) meaning the whole retained tail. Digits only — a non-numeric
// cursor is a client bug, not a silent 0, so it 400s.
func subscribeSince(w http.ResponseWriter, r *http.Request) (uint64, bool) {
	v := r.URL.Query().Get("since")
	if v == "" {
		return 0, true
	}
	var n uint64
	for _, c := range v {
		if c < '0' || c > '9' {
			writeErr(w, http.StatusBadRequest, "since must be a non-negative integer")
			return 0, false
		}
		n = n*10 + uint64(c-'0')
	}
	return n, true
}

// subscribeWait reads the optional ?wait= hold budget as a Go duration ("30s",
// "250ms"). Absent, empty, or zero is the one-shot drain. A budget past
// maxSubscribeWait is CAPPED rather than refused — the client still gets a
// resumable answer — but an unparseable one 400s, because silently treating it
// as "no hold" would look like a server that never pushes.
func subscribeWait(w http.ResponseWriter, r *http.Request) (time.Duration, bool) {
	v := r.URL.Query().Get("wait")
	if v == "" {
		return 0, true
	}
	d, err := time.ParseDuration(v)
	if err != nil || d < 0 {
		writeErr(w, http.StatusBadRequest, "wait must be a non-negative Go duration, e.g. 30s")
		return 0, false
	}
	if d > maxSubscribeWait {
		d = maxSubscribeWait
	}
	return d, true
}

// writeSessionSubscribe answers one subscribe reply. Both the one-shot drain and
// the held attach go through it, which is what keeps their wire shapes identical.
func writeSessionSubscribe(w http.ResponseWriter, traceID string, events []SessionChangeEvent, cursor uint64, complete bool) {
	writeJSON(w, http.StatusOK, SessionSubscribeResponse{
		TraceID:  traceID,
		Events:   events,
		Cursor:   cursor,
		Complete: complete,
	})
}
