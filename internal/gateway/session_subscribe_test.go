package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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

	// An unparseable hold budget 400s rather than silently degrading to the
	// one-shot drain — a server that quietly never pushes is the worse failure.
	rr = httptest.NewRecorder()
	s.handleFakSession(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/session/gw-1/subscribe?wait=abc", nil))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("bad wait status = %d, want 400", rr.Code)
	}
}

// armedHold reports whether a held attach is currently parked on the feed's
// broadcast. It is the deterministic "the request is waiting" signal the #4310
// witness needs: without it a test could only sleep and hope, and a publish that
// raced ahead of the hold would silently turn the push test into a drain test.
func (f *sessionFeed) armedHold() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.wake != nil
}

// waitArmed blocks until a held attach has armed on the feed, failing the test
// rather than hanging if it never does.
func waitArmed(t *testing.T, f *sessionFeed) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if f.armedHold() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("no held attach armed on the feed within 5s")
}

// TestSessionSubscribeHeldRequestCompletesOnPublish is the #4310 acceptance
// witness: a caught-up controller attaches with ?wait=, the request PARKS (it is
// provably armed and has not answered), a revision is published AFTER it parked,
// and the held request completes carrying exactly that revision and the advanced
// cursor. Another session's revision in between must not satisfy the hold — the
// push is per-trace, like the drain it extends.
func TestSessionSubscribeHeldRequestCompletesOnPublish(t *testing.T) {
	s := &Server{sessionFeed: newSessionFeed(0)}
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "running", Rev: 1}) // seq 1

	// The controller is caught up at cursor 1: there is nothing to drain, so the
	// held attach has no choice but to wait.
	done := make(chan SessionSubscribeResponse, 1)
	go func() {
		rr := httptest.NewRecorder()
		s.handleFakSession(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/session/gw-1/subscribe?since=1&wait=30s", nil))
		var got SessionSubscribeResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
			t.Errorf("decode held reply (status %d, body %s): %v", rr.Code, rr.Body.String(), err)
		}
		if rr.Code != http.StatusOK {
			t.Errorf("held subscribe status = %d, want 200 (body %s)", rr.Code, rr.Body.String())
		}
		done <- got
	}()

	waitArmed(t, s.sessionFeed)
	select {
	case got := <-done:
		t.Fatalf("held request answered before any publish: %+v", got)
	default:
	}

	// Noise for a different session wakes the feed-global broadcast but must NOT
	// complete this hold: the subscriber re-drains, finds nothing of its own, and
	// re-arms.
	s.PublishSessionRevision(SessionState{TraceID: "gw-2", Run: "running", Rev: 1}) // seq 2
	waitArmed(t, s.sessionFeed)
	select {
	case got := <-done:
		t.Fatalf("another session's revision completed the hold: %+v", got)
	default:
	}

	// The revision this controller is subscribed to — published while it is held.
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "paused", Rev: 2}) // seq 3

	select {
	case got := <-done:
		if got.TraceID != "gw-1" || len(got.Events) != 1 {
			t.Fatalf("held reply = %+v; want exactly 1 gw-1 event", got)
		}
		if got.Events[0].Rev != 2 || got.Events[0].Seq != 3 {
			t.Fatalf("held event = rev %d seq %d; want the revision published while held (rev 2, seq 3)", got.Events[0].Rev, got.Events[0].Seq)
		}
		if got.Cursor != 3 {
			t.Fatalf("held cursor = %d, want 3 (advanced past the delivered event)", got.Cursor)
		}
		if !got.Complete {
			t.Fatalf("held reply = %+v; want complete (nothing was trimmed)", got)
		}
	// The patience here is deliberately far SHORTER than the ?wait= budget above:
	// that is what makes this a push test rather than an "eventually" test. If the
	// publish did not release the hold, the only other way out is the budget
	// expiring at 30s — so a reply must arrive well inside 5s or the push is dead.
	case <-time.After(5 * time.Second):
		t.Fatal("held request never completed after the revision it was waiting for was published (the publish did not release the hold)")
	}
}

// TestSessionSubscribeHeldRequestBoundedByWait pins the hold's bound: with no
// revision to deliver, an elapsed budget answers the SAME shape as the one-shot
// drain — an empty tail at the head cursor — so a follow loop just calls again
// with the cursor it was handed, and a connection is never held forever.
func TestSessionSubscribeHeldRequestBoundedByWait(t *testing.T) {
	s := &Server{sessionFeed: newSessionFeed(0)}
	s.PublishSessionRevision(SessionState{TraceID: "gw-2", Run: "running", Rev: 1}) // seq 1: not ours

	rr := httptest.NewRecorder()
	s.handleFakSession(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/session/gw-1/subscribe?since=1&wait=20ms", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("elapsed hold status = %d, want 200 (body %s)", rr.Code, rr.Body.String())
	}
	var got SessionSubscribeResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got.Events) != 0 || got.TraceID != "gw-1" || !got.Complete {
		t.Fatalf("elapsed hold = %+v; want an empty complete tail for gw-1", got)
	}
	if got.Cursor != 1 {
		t.Fatalf("elapsed hold cursor = %d, want the head cursor 1", got.Cursor)
	}
}

// TestSessionSubscribeIsDocumentedInOpenAPISpec is the spec half of #4310: the
// subscribe verb lives on the /v1/fak/session/ subtree, which the route-table
// drift gate can only see as /v1/fak/session/{trace_id} — so the verb needs its
// own assertion or it stays invisible to every generated SDK.
func TestSessionSubscribeIsDocumentedInOpenAPISpec(t *testing.T) {
	raw, err := os.ReadFile(filepath.FromSlash(openAPISpecPath))
	if err != nil {
		t.Fatalf("read %s: %v", openAPISpecPath, err)
	}
	spec := string(raw)

	const path = "/v1/fak/session/{trace_id}/subscribe"
	if !specHasPathKey(spec, path) {
		t.Errorf("%s does not document the subscribe verb (expected an OpenAPI path key %q)", openAPISpecPath, path)
	}
	// The held form is the reason the verb is worth documenting: a client cannot
	// discover the push without the parameter that arms it.
	if !strings.Contains(spec, "name: wait") {
		t.Errorf("%s documents the subscribe path but not its ?wait= hold budget (#4310)", openAPISpecPath)
	}
}

func TestSessionSubscribeStreamingSSE(t *testing.T) {
	s := &Server{sessionFeed: newSessionFeed(0)}
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "running", Rev: 1})

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.handleFakSession(w, r)
	}))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/v1/fak/session/gw-1/subscribe?format=sse", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}

	dec := json.NewDecoder(resp.Body)
	_ = dec
	// Read first event (already published)
	buf := make([]byte, 1024)
	n, err := resp.Body.Read(buf)
	if err != nil && n == 0 {
		t.Fatal(err)
	}
	data := string(buf[:n])
	if !strings.Contains(data, "gw-1") || !strings.Contains(data, "running") {
		t.Fatalf("first event output: %s", data)
	}

	// Publish live event
	go func() {
		time.Sleep(20 * time.Millisecond)
		s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "paused", Rev: 2})
	}()

	n2, err := resp.Body.Read(buf)
	if err != nil && n2 == 0 {
		t.Fatal(err)
	}
	data2 := string(buf[:n2])
	if !strings.Contains(data2, "paused") {
		t.Fatalf("second event output: %s", data2)
	}
	cancel()
}

func TestSessionSubscribeStreamingNDJSON(t *testing.T) {
	s := &Server{sessionFeed: newSessionFeed(0)}
	s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "running", Rev: 1})

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.handleFakSession(w, r)
	}))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/v1/fak/session/gw-1/subscribe?format=ndjson", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); ct != "application/x-ndjson" {
		t.Fatalf("Content-Type = %q, want application/x-ndjson", ct)
	}

	buf := make([]byte, 1024)
	n, err := resp.Body.Read(buf)
	if err != nil && n == 0 {
		t.Fatal(err)
	}
	data := string(buf[:n])
	if !strings.Contains(data, "gw-1") || !strings.Contains(data, "running") {
		t.Fatalf("first event output: %s", data)
	}

	go func() {
		time.Sleep(20 * time.Millisecond)
		s.PublishSessionRevision(SessionState{TraceID: "gw-1", Run: "throttled", Rev: 2})
	}()

	n2, err := resp.Body.Read(buf)
	if err != nil && n2 == 0 {
		t.Fatal(err)
	}
	data2 := string(buf[:n2])
	if !strings.Contains(data2, "throttled") {
		t.Fatalf("second event output: %s", data2)
	}
	cancel()
}
