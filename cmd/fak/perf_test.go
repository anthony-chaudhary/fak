package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

func writePerfLedger(t *testing.T, recs ...perfledger.Record) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "gateway-perf.jsonl")
	w := perfledger.OpenWriter(path, 0)
	for _, r := range recs {
		w.Offer(r)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close ledger: %v", err)
	}
	return path
}

func perfFixture() []perfledger.Record {
	return []perfledger.Record{
		perfledger.NewRecord(time.UnixMilli(1), "stop", perfledger.LocalitySelfHosted, 1000, 200, 500, 5*time.Second, time.Second),
		perfledger.NewRecord(time.UnixMilli(2), "stop", perfledger.LocalityVendor, 100, 10, 0, 2*time.Second, 0),
		perfledger.NewRecord(time.UnixMilli(3), "end_turn", perfledger.LocalitySelfHosted, 400, 40, 0, 3*time.Second, 500*time.Millisecond),
	}
}

// fak-test:runtime fast est=50ms
func TestPerfVerbJSONReadsLedger(t *testing.T) {
	path := writePerfLedger(t, perfFixture()...)
	var out, errOut bytes.Buffer
	if rc := runPerf(&out, &errOut, []string{"--ledger", path, "--json", "--n", "2"}); rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	var rep perfledger.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if rep.Schema != perfledger.ReportSchema || rep.Source != path {
		t.Fatalf("schema/source = %q/%q", rep.Schema, rep.Source)
	}
	if rep.Window.Requests != 2 || len(rep.Records) != 2 || rep.Records[0].UnixMS != 2 || rep.Records[1].UnixMS != 3 {
		t.Fatalf("window/records = %+v / %+v, want the newest 2 oldest-first", rep.Window, rep.Records)
	}
	if rep.Summary.Count != 2 || rep.Summary.TTFTMeasured != 1 || rep.Summary.TTFTP50MS != 500 {
		t.Fatalf("summary = %+v", rep.Summary)
	}
}

// fak-test:runtime fast est=50ms
func TestPerfVerbCompactReadsLedger(t *testing.T) {
	path := writePerfLedger(t, perfFixture()...)
	var out, errOut bytes.Buffer
	if rc := runPerf(&out, &errOut, []string{"--ledger", path}); rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	line := out.String()
	if !strings.HasPrefix(line, "PERF n=3") || strings.Count(line, "\n") != 1 {
		t.Fatalf("compact = %q, want one line starting PERF n=3", line)
	}
}

// fak-test:runtime fast est=50ms
func TestPerfVerbEmptyLedgerAndBadFlags(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "absent.jsonl")
	var out, errOut bytes.Buffer
	if rc := runPerf(&out, &errOut, []string{"--ledger", missing}); rc != 0 {
		t.Fatalf("missing ledger rc = %d, want 0", rc)
	}
	if !strings.HasPrefix(out.String(), "PERF n=0") {
		t.Fatalf("missing ledger compact = %q", out.String())
	}
	out.Reset()
	if rc := runPerf(&out, &errOut, []string{"--ledger", missing, "--n", "0"}); rc != 2 {
		t.Fatalf("--n 0 rc = %d, want 2", rc)
	}
}

// fak-test:runtime fast est=100ms
func TestPerfVerbURLReadsLiveEndpoint(t *testing.T) {
	want := perfledger.BuildReport(perfFixture(), 50, false, 0)
	var gotPath, gotN string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotN = r.URL.Path, r.URL.Query().Get("n")
		_ = json.NewEncoder(w).Encode(want)
	}))
	defer ts.Close()
	var out, errOut bytes.Buffer
	if rc := runPerf(&out, &errOut, []string{"--url", ts.URL, "--n", "7", "--json"}); rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	if gotPath != "/v1/fak/perf/recent" || gotN != "7" {
		t.Fatalf("request path/n = %q/%q", gotPath, gotN)
	}
	var rep perfledger.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if rep.Window.Requests != 3 || len(rep.Records) != 3 {
		t.Fatalf("window = %+v", rep.Window)
	}
}
