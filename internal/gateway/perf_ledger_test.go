package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

func getPerfRecent(t *testing.T, srv *Server, query string) *httptest.ResponseRecorder {
	t.Helper()
	path := "/v1/fak/perf/recent"
	if query != "" {
		path += "?" + query
	}
	req := httptest.NewRequest(http.MethodGet, path, nil)
	req.RemoteAddr = "127.0.0.1:50101"
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	return rec
}

func decodePerfReport(t *testing.T, rec *httptest.ResponseRecorder) perfledger.Report {
	t.Helper()
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var rep perfledger.Report
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("decode report: %v", err)
	}
	return rep
}

// fak-test:runtime fast est=500ms lane=default
func TestPerfLedgerServedTurnEmitsOneRowAndSurvivesRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gateway-perf.jsonl")

	srv := newTestServer(t)
	seed, capped := perfledger.ReadTail(path)
	w := perfledger.OpenWriter(path, perfledger.DefaultMaxBytes)
	srv.SetPerfLedger(w, seed, capped)

	// The same fold a served turn reaches (messages_stream_passthrough.go).
	srv.metrics.observeInferenceServedTimed(localitySelfHosted, "", 1000, 201, 500, 0, "end_turn", 5*time.Second, time.Second)

	if err := w.Close(); err != nil {
		t.Fatalf("writer close: %v", err)
	}
	if w.Written() != 1 || w.Dropped() != 0 {
		t.Fatalf("written/dropped = %d/%d, want 1/0", w.Written(), w.Dropped())
	}
	recs, _ := perfledger.ReadTail(path)
	if len(recs) != 1 {
		t.Fatalf("ledger rows = %d, want exactly 1", len(recs))
	}
	got := recs[0]
	if got.Schema != perfledger.Schema || got.Locality != perfledger.LocalitySelfHosted || got.FinishReason != "end_turn" {
		t.Fatalf("row identity = %q/%q/%q", got.Schema, got.Locality, got.FinishReason)
	}
	if got.PromptTokens != 1000 || got.CompletionTokens != 201 || got.CachedTokens != 500 {
		t.Fatalf("row tokens = %d/%d/%d", got.PromptTokens, got.CompletionTokens, got.CachedTokens)
	}
	if got.E2EMS != 5000 || got.TTFTMS != 1000 || got.PrefillTPS != 1000 || got.DecodeTPS != 50 {
		t.Fatalf("row timings = e2e %v ttft %v prefill %v decode %v", got.E2EMS, got.TTFTMS, got.PrefillTPS, got.DecodeTPS)
	}

	// Restart: a fresh server seeded from the file serves the prior row.
	srv2 := newTestServer(t)
	seed2, capped2 := perfledger.ReadTail(path)
	srv2.SetPerfLedger(nil, seed2, capped2)

	rep := decodePerfReport(t, getPerfRecent(t, srv2, "n=10&format=json"))
	if rep.Schema != perfledger.ReportSchema {
		t.Fatalf("report schema = %q", rep.Schema)
	}
	if rep.Window.Requests != 1 || rep.Window.RetainedCap != perfledger.RingCap || rep.Window.Capped {
		t.Fatalf("window = %+v", rep.Window)
	}
	if len(rep.Records) != 1 || rep.Records[0] != got {
		t.Fatalf("restart records = %+v, want [%+v]", rep.Records, got)
	}
	if rep.Summary.Count != 1 || rep.Summary.TTFTMeasured != 1 || rep.Summary.TTFTP50MS != 1000 {
		t.Fatalf("summary = %+v", rep.Summary)
	}

	compact := getPerfRecent(t, srv2, "format=compact")
	if compact.Code != http.StatusOK || !strings.HasPrefix(compact.Body.String(), "PERF n=1") {
		t.Fatalf("compact = %d %q, want 200 with prefix PERF n=1", compact.Code, compact.Body.String())
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestPerfLedgerHTTPChatTurnRecordsExactlyOneRow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gateway-perf.jsonl")
	srv := newTestServer(t)
	// A mock turn can finish inside one tick of Windows' coarse wall clock, which
	// measures e2e as exactly 0; hold the turn long enough to be observable.
	srv.planner = perfDelayedPlanner{inner: srv.planner, d: 20 * time.Millisecond}
	w := perfledger.OpenWriter(path, perfledger.DefaultMaxBytes)
	srv.SetPerfLedger(w, nil, false)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	var resp ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{Model: "test-model", Messages: []agent.Message{{Role: "user", Content: "hello"}}}, &resp)
	if code != http.StatusOK {
		t.Fatalf("chat status = %d, want 200", code)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("writer close: %v", err)
	}
	recs, _ := perfledger.ReadTail(path)
	if len(recs) != 1 {
		t.Fatalf("ledger rows after one served chat turn = %d, want 1", len(recs))
	}
	if recs[0].Schema != perfledger.Schema || recs[0].E2EMS <= 0 {
		t.Fatalf("row = %+v", recs[0])
	}
	rep := decodePerfReport(t, getPerfRecent(t, srv, ""))
	if rep.Window.Requests != 1 || len(rep.Records) != 1 {
		t.Fatalf("live window = %+v records=%d, want 1", rep.Window, len(rep.Records))
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestPerfLedgerRecentRejectsBadParams(t *testing.T) {
	srv := newTestServer(t)
	for _, q := range []string{"n=0", "n=abc", "format=xml"} {
		if rec := getPerfRecent(t, srv, q); rec.Code != http.StatusBadRequest {
			t.Fatalf("%s: status = %d, want 400", q, rec.Code)
		}
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/fak/perf/recent", nil)
	req.RemoteAddr = "127.0.0.1:50102"
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST status = %d, want 405", rec.Code)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestPerfLedgerRingCapsAndSeedPrecedesLive(t *testing.T) {
	srv := newTestServer(t)
	seed := make([]perfledger.Record, perfledger.RingCap)
	for i := range seed {
		seed[i] = perfledger.Record{Schema: perfledger.Schema, UnixMS: int64(i), E2EMS: 1}
	}
	srv.SetPerfLedger(nil, seed, false)
	srv.metrics.observeInferenceServedTimed(localityVendor, "", 10, 1, 0, 0, "stop", time.Second, 0)

	rep := srv.PerfReport(perfledger.RingCap)
	if rep.Window.Requests != perfledger.RingCap || !rep.Window.Capped {
		t.Fatalf("window = %+v, want %d capped", rep.Window, perfledger.RingCap)
	}
	if rep.Records[0].UnixMS != 1 {
		t.Fatalf("oldest retained unix_ms = %d, want 1 (seed row 0 evicted)", rep.Records[0].UnixMS)
	}
	last := rep.Records[len(rep.Records)-1]
	if last.Locality != perfledger.LocalityVendor || last.TTFTMS != 0 {
		t.Fatalf("newest row = %+v, want the live vendor turn with ttft unmeasured", last)
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestPerfLedgerRowCarriesServingIdentity(t *testing.T) {
	srv := newTestServer(t)
	srv.metrics.setPerfIdentity("default-model", perfledger.Identity{Planner: "inkernel", Backend: "cpu-ref", Host: "h", Version: "v"})

	srv.metrics.observeInferenceServedTimed(localitySelfHosted, "routed-model", 10, 5, 0, 0, "stop", time.Second, 100*time.Millisecond)
	srv.metrics.observeInferenceServedTimed(localitySelfHosted, "", 10, 5, 0, 0, "stop", time.Second, 100*time.Millisecond)

	rep := decodePerfReport(t, getPerfRecent(t, srv, "n=10"))
	if len(rep.Records) != 2 {
		t.Fatalf("records = %d, want 2", len(rep.Records))
	}
	want := perfledger.Identity{Planner: "inkernel", Backend: "cpu-ref", Host: "h", Version: "v"}
	if got := rep.Records[0]; got.Model != "routed-model" || got.Identity != want {
		t.Fatalf("routed row = %q %+v", got.Model, got.Identity)
	}
	if got := rep.Records[1].Model; got != "default-model" {
		t.Fatalf("unrouted row model = %q, want server default", got)
	}
	if rep.Summary.Identities != 2 || rep.Summary.ServedBy == nil || rep.Summary.ServedBy.Model != "default-model" {
		t.Fatalf("summary served_by = %+v (%d), want newest default-model (2)", rep.Summary.ServedBy, rep.Summary.Identities)
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestNewServerStampsPerfIdentity(t *testing.T) {
	srv := newTestServer(t)
	sb := srv.metrics.perfServedBy.Load()
	if sb == nil || sb.Planner == "" || sb.Version == "" {
		t.Fatalf("server perf identity = %+v, want planner and version resolved at New", sb)
	}
}
