package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// denyEvent builds a journal-producing DENY event (rowFromEvent only mints a Row
// for DECIDE/DENY/QUARANTINE/VDSO_HIT kinds).
func denyEvent(tool, trace string) abi.Event {
	return abi.Event{
		Kind: abi.EvDeny,
		Call: &abi.ToolCall{
			Tool:    tool,
			TraceID: trace,
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
		},
		Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "test"},
	}
}

// vdsoHitEvent builds a journal-producing VDSO_HIT/ALLOW event — a row whose Kind AND
// Verdict both differ from denyEvent's, so a filter test can tell the two columns apart
// instead of accidentally passing on a single conflated one.
func vdsoHitEvent(tool, trace string) abi.Event {
	return abi.Event{
		Kind: abi.EvVDSOHit,
		Call: &abi.ToolCall{
			Tool:    tool,
			TraceID: trace,
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
		},
		Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "test"},
	}
}

func TestHandleFakEvents404WhenJournalDisabled(t *testing.T) {
	prev := activeJournal
	activeJournal = func() *journal.Journal { return nil }
	defer func() { activeJournal = prev }()

	rec := httptest.NewRecorder()
	(&Server{}).handleFakEvents(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/events", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("want 404 when no journal configured, got %d (%s)", rec.Code, rec.Body.String())
	}
}

func TestHandleFakEventsDrainsTailAndAdvancesCursor(t *testing.T) {
	j := journal.OpenMemory()
	j.Emit(denyEvent("send_email", "trace-a"))
	j.Emit(denyEvent("Bash", "trace-b"))
	j.Emit(denyEvent("fetch_url", "trace-c"))

	prev := activeJournal
	activeJournal = func() *journal.Journal { return j }
	defer func() { activeJournal = prev }()

	s := &Server{}

	// since=0 drains the whole retained tail in order, cursor = highest Seq.
	rec := httptest.NewRecorder()
	s.handleFakEvents(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/events", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("want 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var resp EventsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Events) != 3 {
		t.Fatalf("want 3 drained events, got %d", len(resp.Events))
	}
	if resp.Events[0].Tool != "send_email" || resp.Events[2].Tool != "fetch_url" {
		t.Fatalf("rows out of order: %q .. %q", resp.Events[0].Tool, resp.Events[2].Tool)
	}
	if resp.Cursor != 3 {
		t.Fatalf("want cursor 3, got %d", resp.Cursor)
	}
	// Rows must carry the hash-chain fields so an auditor can VerifyRows them.
	if resp.Events[0].Hash == "" || resp.Events[0].Verdict != "DENY" {
		t.Fatalf("row missing chain/verdict fields: %+v", resp.Events[0])
	}

	// since=2 returns only the tail after the cursor; cursor stays at the head.
	rec = httptest.NewRecorder()
	s.handleFakEvents(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/events?since=2", nil))
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Events) != 1 || resp.Events[0].Seq != 3 {
		t.Fatalf("want only Seq 3 after since=2, got %d rows", len(resp.Events))
	}
	if resp.Cursor != 3 {
		t.Fatalf("want cursor 3, got %d", resp.Cursor)
	}

	// A non-numeric cursor is a 400.
	rec = httptest.NewRecorder()
	s.handleFakEvents(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/events?since=abc", nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400 for non-numeric since, got %d", rec.Code)
	}

	// POST {"since":N} is honored too.
	rec = httptest.NewRecorder()
	s.handleFakEvents(rec, httptest.NewRequest(http.MethodPost, "/v1/fak/events", strings.NewReader(`{"since":1}`)))
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Events) != 2 {
		t.Fatalf("want 2 rows after POST since=1, got %d", len(resp.Events))
	}
}

// filterJournal seeds a journal whose four rows differ on every filterable column, so a
// predicate that silently ignores its field cannot pass by returning "everything".
//
//	Seq 1  DENY / send_email / trace-a
//	Seq 2  DENY / Bash       / trace-b
//	Seq 3  VDSO_HIT ALLOW / Bash / trace-a
//	Seq 4  DENY / send_email / trace-b
func filterJournal(t *testing.T) *journal.Journal {
	t.Helper()
	j := journal.OpenMemory()
	j.Emit(denyEvent("send_email", "trace-a"))
	j.Emit(denyEvent("Bash", "trace-b"))
	j.Emit(vdsoHitEvent("Bash", "trace-a"))
	j.Emit(denyEvent("send_email", "trace-b"))
	prev := activeJournal
	activeJournal = func() *journal.Journal { return j }
	t.Cleanup(func() { activeJournal = prev })
	return j
}

// drainEvents runs one request through the handler and returns the decoded response,
// failing on any non-200 so a filter assertion never reads a zero value off an error body.
func drainEvents(t *testing.T, r *http.Request) EventsResponse {
	t.Helper()
	rec := httptest.NewRecorder()
	(&Server{}).handleFakEvents(rec, r)
	if rec.Code != http.StatusOK {
		t.Fatalf("want 200 for %s %s, got %d (%s)", r.Method, r.URL, rec.Code, rec.Body.String())
	}
	var resp EventsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return resp
}

// seqsOf renders the drained rows' Seq line, the assertion currency for "exactly these
// rows, in this order".
func seqsOf(resp EventsResponse) []uint64 {
	got := make([]uint64, 0, len(resp.Events))
	for _, row := range resp.Events {
		got = append(got, row.Seq)
	}
	return got
}

func sameSeqs(got, want []uint64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// TestHandleFakEventsRowFilters covers the #3979 predicates on GET and POST: each
// column narrows on its own, the predicates AND together, and an unknown filter key is
// a 400 on both verbs rather than a silently unfiltered full-tail drain.
func TestHandleFakEventsRowFilters(t *testing.T) {
	filterJournal(t)

	// Each column narrows independently, and the cursor stays at the journal head (4)
	// regardless of which rows survived the predicate.
	for _, tc := range []struct {
		name  string
		query string
		body  string
		want  []uint64
	}{
		{"verdict", "?verdict=DENY", `{"verdict":"DENY"}`, []uint64{1, 2, 4}},
		{"kind", "?kind=VDSO_HIT", `{"kind":"VDSO_HIT"}`, []uint64{3}},
		{"tool", "?tool=Bash", `{"tool":"Bash"}`, []uint64{2, 3}},
		{"trace_id", "?trace_id=trace-a", `{"trace_id":"trace-a"}`, []uint64{1, 3}},
		// Combinable: tool AND trace_id together select strictly less than either alone.
		{"tool+trace", "?tool=Bash&trace_id=trace-b", `{"tool":"Bash","trace_id":"trace-b"}`, []uint64{2}},
		// Combinable across kind and verdict, which a conflated implementation would mix up.
		{"kind+verdict", "?kind=DENY&verdict=DENY", `{"kind":"DENY","verdict":"DENY"}`, []uint64{1, 2, 4}},
		// A filter that matches nothing still answers 200 with the ADVANCED cursor.
		{"matches nothing", "?tool=no_such_tool", `{"tool":"no_such_tool"}`, nil},
		// A filter combined with the cursor narrows only the post-cursor window.
		{"since+verdict", "?since=2&verdict=DENY", `{"since":2,"verdict":"DENY"}`, []uint64{4}},
	} {
		t.Run("GET "+tc.name, func(t *testing.T) {
			resp := drainEvents(t, httptest.NewRequest(http.MethodGet, "/v1/fak/events"+tc.query, nil))
			if got := seqsOf(resp); !sameSeqs(got, tc.want) {
				t.Fatalf("GET %s: want seqs %v, got %v", tc.query, tc.want, got)
			}
			if resp.Cursor != 4 {
				t.Fatalf("GET %s: want cursor advanced to 4, got %d", tc.query, resp.Cursor)
			}
		})
		t.Run("POST "+tc.name, func(t *testing.T) {
			resp := drainEvents(t, httptest.NewRequest(http.MethodPost, "/v1/fak/events", strings.NewReader(tc.body)))
			if got := seqsOf(resp); !sameSeqs(got, tc.want) {
				t.Fatalf("POST %s: want seqs %v, got %v", tc.body, tc.want, got)
			}
			if resp.Cursor != 4 {
				t.Fatalf("POST %s: want cursor advanced to 4, got %d", tc.body, resp.Cursor)
			}
		})
	}

	// An unknown filter key is a 400 on GET (typo'd predicate) and on POST (unknown body
	// field). Ignoring it would drain the whole tail and read as "matched everything".
	for _, tc := range []struct {
		name   string
		method string
		target string
		body   string
	}{
		{"GET typo", http.MethodGet, "/v1/fak/events?verdit=DENY", ""},
		{"GET unsupported column", http.MethodGet, "/v1/fak/events?reason=POLICY_BLOCK", ""},
		{"GET valid plus unknown", http.MethodGet, "/v1/fak/events?verdict=DENY&nope=1", ""},
		{"POST typo", http.MethodPost, "/v1/fak/events", `{"verdit":"DENY"}`},
		{"POST valid plus unknown", http.MethodPost, "/v1/fak/events", `{"verdict":"DENY","nope":1}`},
	} {
		t.Run("400 "+tc.name, func(t *testing.T) {
			var body io.Reader
			if tc.body != "" {
				body = strings.NewReader(tc.body)
			}
			rec := httptest.NewRecorder()
			(&Server{}).handleFakEvents(rec, httptest.NewRequest(tc.method, tc.target, body))
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("want 400 for unknown filter key (%s), got %d (%s)", tc.name, rec.Code, rec.Body.String())
			}
		})
	}
}

// TestHandleFakEventsFilteredCursorDoesNotRedeliverOrSkip is the cursor-semantics
// guarantee the filters must not break (#3979): polling with a predicate must advance
// past rows the filter DROPPED, so the next poll neither re-delivers them nor skips a
// matching row that landed behind them. The bug this pins is the natural
// implementation — advancing the cursor only over DELIVERED rows — which pins a
// narrow filter's cursor at the last match and re-walks the same tail forever.
func TestHandleFakEventsFilteredCursorDoesNotRedeliverOrSkip(t *testing.T) {
	j := filterJournal(t)

	// Poll 1: only trace-a's VDSO_HIT (Seq 3) matches. The cursor must still advance to
	// 4 — past the non-matching Seq 4 the drain walked — not stall at 3.
	resp := drainEvents(t, httptest.NewRequest(http.MethodGet, "/v1/fak/events?kind=VDSO_HIT", nil))
	if got := seqsOf(resp); !sameSeqs(got, []uint64{3}) {
		t.Fatalf("poll 1: want seq [3], got %v", got)
	}
	if resp.Cursor != 4 {
		t.Fatalf("poll 1: cursor must advance past the filtered-out head row: want 4, got %d", resp.Cursor)
	}

	// Poll 2 at that cursor with no new rows: nothing re-delivered, cursor held.
	next := drainEvents(t, httptest.NewRequest(http.MethodGet,
		"/v1/fak/events?kind=VDSO_HIT&since="+strconv.FormatUint(resp.Cursor, 10), nil))
	if len(next.Events) != 0 {
		t.Fatalf("poll 2: filtered rows must not be re-delivered, got %v", seqsOf(next))
	}
	if next.Cursor != 4 {
		t.Fatalf("poll 2: want cursor held at 4, got %d", next.Cursor)
	}

	// A new matching row lands behind rows the filter drops. Polling from the advanced
	// cursor must see it — the head was not skipped by the earlier advance.
	j.Emit(denyEvent("Bash", "trace-c"))       // Seq 5, filtered out
	j.Emit(vdsoHitEvent("fetch_url", "trace-c")) // Seq 6, matches
	after := drainEvents(t, httptest.NewRequest(http.MethodGet,
		"/v1/fak/events?kind=VDSO_HIT&since="+strconv.FormatUint(next.Cursor, 10), nil))
	if got := seqsOf(after); !sameSeqs(got, []uint64{6}) {
		t.Fatalf("poll 3: want the new matching seq [6], got %v", got)
	}
	if after.Cursor != 6 {
		t.Fatalf("poll 3: want cursor 6, got %d", after.Cursor)
	}
}

// TestHandleFakEventsUnfilteredDrainUnchanged pins the back-compat floor: with no
// predicate the endpoint keeps its exact pre-#3979 behavior.
func TestHandleFakEventsUnfilteredDrainUnchanged(t *testing.T) {
	filterJournal(t)
	resp := drainEvents(t, httptest.NewRequest(http.MethodGet, "/v1/fak/events", nil))
	if got := seqsOf(resp); !sameSeqs(got, []uint64{1, 2, 3, 4}) {
		t.Fatalf("unfiltered drain must return the whole tail, got %v", got)
	}
	if resp.Cursor != 4 {
		t.Fatalf("want cursor 4, got %d", resp.Cursor)
	}
}
