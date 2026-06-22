package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
