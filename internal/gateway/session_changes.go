package gateway

import (
	"net/http"
	"sync"
)

// session_changes.go — the drive-state revision stream (#630). It is the
// /v1/fak/changes-style feed for SESSION control state: where /v1/fak/changes
// streams the vDSO coherence bus (cache Mutations/Revocations), this streams every
// revision of the per-session DRIVE table (run-state/budget/priority/pace), so
// "what is every session doing right now" becomes a live tail and every scheduling
// preemption is a first-class, auditable wire event rather than hidden policy.
//
// The feed is PUSH-FED by the host: cmd/fak wires session.Table.WatchRevisions to
// PublishSessionRevision, so the gateway stays session-internals-blind (it ingests
// the already-projected SessionState, never internal/session). It carries its OWN
// monotone Seq — assigned at append, NOT the per-session Rev — because Rev is
// per-session and not globally monotone, so it cannot be a cross-session drain
// cursor. Each event still carries Rev (the per-session optimistic-concurrency
// cursor the design keys on); Seq is the feed's global drain cursor, exactly like
// the coherence feed's bus Seq. A client drains by Seq; a never-draining client
// cannot grow the ring without bound (the oldest events fall off and a lapsed
// client sees a Seq gap and re-syncs to head).

// SessionChangeEvent is one drive-state revision on the wire: a session's full
// drive snapshot AT the revision, tagged with the feed's monotone Seq (the drain
// cursor). The embedded SessionState carries Rev — the per-session cursor — so a
// consumer can both order the global stream (Seq) and dedupe per session (Rev).
type SessionChangeEvent struct {
	Seq          uint64 `json:"seq"` // feed-local monotone sequence (the drain cursor)
	SessionState        // the drive snapshot at this revision (carries trace_id, run, budget, …, rev)
}

// sessionFeed is a bounded ring of SessionChangeEvents fed by the host's
// every-revision observer. It is the server's sliding window onto the drive table;
// a client drains it by cursor. Bounded so a never-draining client cannot grow it
// without limit.
type sessionFeed struct {
	mu   sync.Mutex
	ring []SessionChangeEvent // chronological; oldest at front
	cap  int
	seq  uint64 // monotone; assigned to each event at append
}

// newSessionFeed builds the drive-state revision ring. capacity<=0 uses
// defaultFeedCap (shared with the coherence feed).
func newSessionFeed(capacity int) *sessionFeed {
	if capacity <= 0 {
		capacity = defaultFeedCap
	}
	return &sessionFeed{cap: capacity}
}

// add records one drive-state revision, assigning it the next monotone feed Seq and
// trimming the oldest event once the ring is full. It takes only its own cheap mutex
// (no session-table access), so it is safe to call from session.Table.putLocked
// under the table lock — the lock-held, in-Rev-order delivery a cursor feed needs.
func (f *sessionFeed) add(st SessionState) {
	f.mu.Lock()
	f.seq++
	f.ring = appendRingCapped(f.ring, SessionChangeEvent{Seq: f.seq, SessionState: st}, f.cap)
	f.mu.Unlock()
}

// drain returns every retained event with Seq > sinceSeq (sinceSeq==0 => all
// retained) plus the highest Seq now known (the client's next cursor). The cursor
// advances over the whole retained tail, so a lapsed client re-syncs to head rather
// than re-scanning forever.
func (f *sessionFeed) drain(sinceSeq uint64) ([]SessionChangeEvent, uint64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]SessionChangeEvent, 0, len(f.ring))
	cursor := sinceSeq
	for _, ev := range f.ring {
		if ev.Seq > sinceSeq {
			out = append(out, ev)
		}
		if ev.Seq > cursor {
			cursor = ev.Seq
		}
	}
	return out, cursor
}

// PublishSessionRevision records one drive-state revision onto the session change
// feed. The host wires it to session.Table.WatchRevisions so every Rev bump streams;
// a nil receiver (a server built without the feed, e.g. the zero Server a route test
// constructs) is a no-op, so wiring it is always safe.
func (s *Server) PublishSessionRevision(st SessionState) {
	if s == nil || s.sessionFeed == nil {
		return
	}
	s.sessionFeed.add(st)
}

// sessionChanges drains the drive-state revision feed for events after the client's
// cursor. A server built without the feed returns an empty tail at the same cursor.
func (s *Server) sessionChanges(sinceSeq uint64) ([]SessionChangeEvent, uint64) {
	if s.sessionFeed == nil {
		return nil, sinceSeq
	}
	return s.sessionFeed.drain(sinceSeq)
}

// SessionChangesResponse is the drained drive-state revision slice plus the client's
// next cursor (mirrors ChangesResponse for the coherence feed).
type SessionChangesResponse struct {
	Events []SessionChangeEvent `json:"events"`
	Cursor uint64               `json:"cursor"`
}

// handleFakSessionChanges drains the drive-state revision stream (#630) after the
// client's ?since= cursor — the Seq of the last revision it saw; 0 returns the whole
// retained tail. It is the /v1/fak/changes protocol applied to the per-session DRIVE
// table rather than the vDSO coherence bus. GET ?since=N or POST {"since":N}. Unlike
// the coherence feed it is not principal-scoped: it exposes exactly what the existing
// /v1/fak/sessions snapshot already does (every retained session's drive state), so
// it opens no new cross-tenant surface — it just makes that read a live tail.
func (s *Server) handleFakSessionChanges(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use GET or POST")
		return
	}
	var since uint64
	if r.Method == http.MethodPost {
		var req ChangesRequest
		if !decodeRequestBody(w, r, &req) {
			return
		}
		since = req.Since
	} else if v := r.URL.Query().Get("since"); v != "" {
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
	events, cursor := s.sessionChanges(since)
	writeJSON(w, http.StatusOK, SessionChangesResponse{Events: events, Cursor: cursor})
}
