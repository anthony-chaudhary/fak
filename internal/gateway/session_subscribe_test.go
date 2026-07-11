package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// session_subscribe_test.go — the subscribe/re-attach op (#2767). The headline
// test is the issue's witness: a controller drains a session's event stream,
// disconnects, re-attaches by trace handle with its saved cursor, and resumes
// with NO revision lost across the reconnect boundary — then issues a control op
// on the same handle. The companion tests pin the honest-loss contract (a lapsed
// cursor reads complete=false, never a silently truncated clean tail) and the
// fail-closed feedless 404.

// TestSessionSubscribeReconnectResumesCursor is the #2767 acceptance witness:
// cursor continuity over the per-session stream across a disconnect, then a
// control op issued after the re-attach.
func TestSessionSubscribeReconnectResumesCursor(t *testing.T) {
	s := &Server{sessionFeed: newSessionFeed(0)}
	s.logf = func(string, ...any) {} // the control success path logs; a bare test Server has no sink
	// A control stub so the re-attached controller can issue an op on the same
	// handle (the POST /v1/fak/session/{id}/{verb} surface).
	var controlled struct {
		trace, verb string
	}
	s.controlSession = func(_ context.Context, trace, verb string, req SessionControlRequest) (SessionState, bool, error) {
		controlled.trace, controlled.verb = trace, verb
		return SessionState{TraceID: trace, Run: "running", Priority: *req.Priority, Rev: 6}, true, nil
	}

	// The session under subscription (gw-1) interleaved with another session's
	// noise (gw-2): the per-trace drain must carry gw-1's revisions only.
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "running", Rev: 1})
	s.PublishSessionRevision(SessionState{TraceID: "gw-2", Run: "running", Rev: 1})
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "throttled", Rev: 2})

	subscribe := func(target string) SessionSubscribeResponse {
		t.Helper()
		rr := httptest.NewRecorder()
		s.handleFakSession(rr, httptest.NewRequest(http.MethodGet, target, nil))
		if rr.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d, want 200 (body %s)", target, rr.Code, rr.Body.String())
		}
		var got SessionSubscribeResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
			t.Fatalf("decode: %v", err)
		}
		return got
	}

	// First attach: the whole retained tail for gw-1.
	first := subscribe("/v1/fak/session/gw-1/subscribe")
	if len(first.Events) != 2 || first.TraceID != "gw-1" || !first.Complete {
		t.Fatalf("first attach = %+v; want 2 gw-1 events, complete", first)
	}
	if first.Events[0].Rev != 1 || first.Events[1].Rev != 2 {
		t.Fatalf("first attach revs = %d,%d; want 1,2", first.Events[0].Rev, first.Events[1].Rev)
	}
	// The cursor is the feed's global Seq (3 events published), not the filtered count.
	if first.Cursor != 3 {
		t.Fatalf("first cursor = %d, want 3", first.Cursor)
	}

	// Controller disconnects; the session keeps revising (plus more noise).
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "running", Rev: 3})
	s.PublishSessionRevision(SessionState{TraceID: "gw-2", Run: "paused", Rev: 2})
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "paused", Rev: 4})

	// Re-attach by the same trace handle with the saved cursor: the resumed tail
	// starts exactly after the last event seen — no gap, no replay.
	second := subscribe("/v1/fak/session/gw-1/subscribe?since=3")
	if len(second.Events) != 2 || !second.Complete {
		t.Fatalf("re-attach = %+v; want 2 new gw-1 events, complete", second)
	}
	if second.Events[0].Rev != 3 || second.Events[1].Rev != 4 {
		t.Fatalf("re-attach revs = %d,%d; want 3,4", second.Events[0].Rev, second.Events[1].Rev)
	}

	// Continuity across the reconnect boundary: the two drains together carry
	// every revision 1..4 exactly once — nothing lost, nothing duplicated.
	seen := map[uint64]bool{}
	for _, ev := range append(first.Events, second.Events...) {
		if seen[ev.Rev] {
			t.Fatalf("revision %d delivered twice across the reconnect", ev.Rev)
		}
		seen[ev.Rev] = true
	}
	for rev := uint64(1); rev <= 4; rev++ {
		if !seen[rev] {
			t.Fatalf("revision %d lost across the reconnect", rev)
		}
	}

	// The re-attached controller issues a control op on the same handle.
	rr := httptest.NewRecorder()
	s.handleFakSession(rr, httptest.NewRequest(http.MethodPost, "/v1/fak/session/gw-1/priority", strings.NewReader(`{"priority":5}`)))
	if rr.Code != http.StatusOK {
		t.Fatalf("post-re-attach control status = %d, want 200 (body %s)", rr.Code, rr.Body.String())
	}
	if controlled.trace != "gw-1" || controlled.verb != "priority" {
		t.Fatalf("control landed on %s/%s, want gw-1/priority", controlled.trace, controlled.verb)
	}
}

// TestSessionSubscribeLapsedCursorReadsIncomplete pins the honest-loss contract:
// a client whose cursor predates the bounded ring's oldest retained event gets
// complete=false and a head cursor to re-sync to — never a clean-looking tail
// that silently skipped trimmed revisions.
func TestSessionSubscribeLapsedCursorReadsIncomplete(t *testing.T) {
	s := &Server{sessionFeed: newSessionFeed(2)}                                    // tiny ring: trims fast
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "running", Rev: 1}) // seq 1
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "paused", Rev: 2})  // seq 2
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "running", Rev: 3}) // seq 3, evicts seq 1

	rr := httptest.NewRecorder()
	s.handleFakSession(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/session/gw-1/subscribe", nil))
	var got SessionSubscribeResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	// since=0 while seq 1 is trimmed: the tail is real but NOT complete.
	if got.Complete {
		t.Fatalf("lapsed cursor read complete=true over a trimmed window: %+v", got)
	}
	if len(got.Events) != 2 || got.Cursor != 3 {
		t.Fatalf("lapsed drain = %+v; want the 2 retained events, head cursor 3", got)
	}

	// A client at the retained-window edge (since=1: everything after seq 1 is
	// retained) resumes complete.
	rr = httptest.NewRecorder()
	s.handleFakSession(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/session/gw-1/subscribe?since=1", nil))
	got = SessionSubscribeResponse{}
	_ = json.Unmarshal(rr.Body.Bytes(), &got)
	if !got.Complete || len(got.Events) != 2 {
		t.Fatalf("edge cursor = %+v; want complete with 2 events", got)
	}
}

// TestSessionSubscribeGuards pins the request-shape and fail-closed edges: a bad
// cursor 400s, a feedless server 404s (never a silent empty tail), and a GET
// with a non-subscribe verb keeps the pre-#2767 405.
func TestSessionSubscribeGuards(t *testing.T) {
	s := &Server{sessionFeed: newSessionFeed(0)}

	rr := httptest.NewRecorder()
	s.handleFakSession(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/session/gw-1/subscribe?since=abc", nil))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("bad cursor status = %d, want 400", rr.Code)
	}

	feedless := &Server{}
	rr = httptest.NewRecorder()
	feedless.handleFakSession(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/session/gw-1/subscribe", nil))
	if rr.Code != http.StatusNotFound {
		t.Fatalf("feedless subscribe status = %d, want 404", rr.Code)
	}

	rr = httptest.NewRecorder()
	s.handleFakSession(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/session/gw-1/observe", nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("non-subscribe GET verb status = %d, want 405", rr.Code)
	}
}
