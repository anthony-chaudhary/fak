package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// session_changes_test.go — the drive-state revision stream (#630). It covers the
// bounded ring's cursor/trim semantics, the host-pushed publish path, the GET/POST
// HTTP drain (incl. the bad-cursor and bad-method guards), and the ServeMux exact-
// match precedence that lets /v1/fak/session/changes win over the /v1/fak/session/
// control subtree.

func TestSessionFeedDrainCursor(t *testing.T) {
	f := newSessionFeed(0)
	f.add(SessionState{TraceID: "a", Run: "running"}) // seq 1
	f.add(SessionState{TraceID: "a", Run: "paused"})  // seq 2
	f.add(SessionState{TraceID: "b", Run: "running"}) // seq 3

	all, cur := f.drain(0)
	if len(all) != 3 || cur != 3 {
		t.Fatalf("drain(0): got %d events, cursor %d; want 3 events, cursor 3", len(all), cur)
	}
	if all[0].Seq != 1 || all[2].Seq != 3 {
		t.Fatalf("feed seqs not monotone from 1: %d..%d", all[0].Seq, all[2].Seq)
	}

	tail, cur := f.drain(2)
	if len(tail) != 1 || tail[0].Seq != 3 || tail[0].TraceID != "b" || cur != 3 {
		t.Fatalf("drain(2): got %+v cursor %d; want [seq3 b] cursor 3", tail, cur)
	}

	// A client at head sees nothing and stays at head.
	none, cur := f.drain(3)
	if len(none) != 0 || cur != 3 {
		t.Fatalf("drain(3): got %d events cursor %d; want 0 events cursor 3", len(none), cur)
	}
}

func TestSessionFeedBoundedRingDropsOldest(t *testing.T) {
	f := newSessionFeed(2) // cap 2
	f.add(SessionState{TraceID: "a"})
	f.add(SessionState{TraceID: "b"})
	f.add(SessionState{TraceID: "c"}) // evicts a

	all, cur := f.drain(0)
	if len(all) != 2 || all[0].TraceID != "b" || all[1].TraceID != "c" {
		t.Fatalf("bounded ring: got %+v; want [b c]", all)
	}
	// The cursor still reflects the highest seq ever assigned (3), so a lapsed
	// client that missed 'a' re-syncs to head rather than re-reading forever.
	if cur != 3 {
		t.Fatalf("cursor after eviction = %d, want 3", cur)
	}
}

func TestPublishSessionRevisionNilSafe(t *testing.T) {
	// A server with no feed (the zero Server a route test builds) must not panic.
	var s Server
	s.PublishSessionRevision(SessionState{TraceID: "x"}) // no-op, must not panic
	if ev, cur := s.sessionChanges(0); len(ev) != 0 || cur != 0 {
		t.Fatalf("nil-feed sessionChanges = %v,%d; want empty,0", ev, cur)
	}
}

func TestHandleFakSessionChangesHTTP(t *testing.T) {
	s := &Server{sessionFeed: newSessionFeed(0)}
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "running", Rev: 1})
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "paused", Rev: 2})

	// GET full tail.
	rr := httptest.NewRecorder()
	s.handleFakSessionChanges(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/session/changes", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("GET status = %d, want 200", rr.Code)
	}
	var got SessionChangesResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got.Events) != 2 || got.Cursor != 2 {
		t.Fatalf("GET drained %+v cursor %d; want 2 events cursor 2", got.Events, got.Cursor)
	}
	if got.Events[1].Run != "paused" || got.Events[1].Rev != 2 || got.Events[1].Seq != 2 {
		t.Fatalf("event carries seq+rev+run: %+v", got.Events[1])
	}

	// GET ?since= drains only newer.
	rr = httptest.NewRecorder()
	s.handleFakSessionChanges(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/session/changes?since=1", nil))
	got = SessionChangesResponse{}
	_ = json.Unmarshal(rr.Body.Bytes(), &got)
	if len(got.Events) != 1 || got.Events[0].Seq != 2 {
		t.Fatalf("GET ?since=1 = %+v; want only seq 2", got.Events)
	}

	// POST {since}.
	rr = httptest.NewRecorder()
	s.handleFakSessionChanges(rr, httptest.NewRequest(http.MethodPost, "/v1/fak/session/changes", strings.NewReader(`{"since":1}`)))
	got = SessionChangesResponse{}
	_ = json.Unmarshal(rr.Body.Bytes(), &got)
	if len(got.Events) != 1 || got.Events[0].Seq != 2 {
		t.Fatalf("POST since=1 = %+v; want only seq 2", got.Events)
	}

	// Bad cursor → 400.
	rr = httptest.NewRecorder()
	s.handleFakSessionChanges(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/session/changes?since=abc", nil))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("GET ?since=abc status = %d, want 400", rr.Code)
	}

	// Wrong method → 405.
	rr = httptest.NewRecorder()
	s.handleFakSessionChanges(rr, httptest.NewRequest(http.MethodDelete, "/v1/fak/session/changes", nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("DELETE status = %d, want 405", rr.Code)
	}
}

// TestSessionChangesRouteWinsOverSubtree proves the exact /v1/fak/session/changes
// pattern is matched ahead of the /v1/fak/session/ control subtree by net/http's
// ServeMux (the longer, exact pattern wins) — so the feed is reachable and a session
// id is never mistaken for the feed path.
func TestSessionChangesRouteWinsOverSubtree(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/fak/session/changes", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("changes")) })
	mux.HandleFunc("/v1/fak/session/", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("subtree")) })

	for path, want := range map[string]string{
		"/v1/fak/session/changes": "changes",
		"/v1/fak/session/gw-1":    "subtree",
	} {
		rr := httptest.NewRecorder()
		mux.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, path, nil))
		if got := rr.Body.String(); got != want {
			t.Errorf("%s routed to %q, want %q", path, got, want)
		}
	}
}
