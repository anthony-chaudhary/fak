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

// fak-test:runtime fast est=100ms
// fak-test:justify why=contract when=changed:cmd/fak/perf.go
func TestPerfTokenComparison(t *testing.T) {
	full := strings.Repeat("a", 64)
	makeRecords := func() []perfledger.Record {
		records := perfFixture()[:2]
		for i, id := range []string{"left", "right"} {
			records[i].RequestID, records[i].Status = id, 200
			records[i].BenchEvidence = &perfledger.BenchEvidence{Complete: true, TokenSHA256: full, TokenSHA: full[:12]}
		}
		return records
	}
	tests := []struct {
		name    string
		change  func([]perfledger.Record) []perfledger.Record
		n       string
		outcome perfledger.BenchTokenOutcome
		reason  perfledger.BenchTokenReason
	}{
		{"equal", func(r []perfledger.Record) []perfledger.Record { return r }, "2", perfledger.BenchTokensEqual, perfledger.BenchTokenVerified},
		{"different", func(r []perfledger.Record) []perfledger.Record {
			r[1].BenchEvidence.TokenSHA256 = full[:12] + strings.Repeat("b", 52)
			return r
		}, "2", perfledger.BenchTokensDifferent, perfledger.BenchTokenVerified},
		{"incomplete", func(r []perfledger.Record) []perfledger.Record { r[1].BenchEvidence.Complete = false; return r }, "2", perfledger.BenchTokensUnverified, perfledger.BenchTokenIncomplete},
		{"failed", func(r []perfledger.Record) []perfledger.Record { r[1].Error = perfledger.ErrorUpstreamStatus; return r }, "2", perfledger.BenchTokensUnverified, perfledger.BenchTokenFailedRequest},
		{"missing_digest", func(r []perfledger.Record) []perfledger.Record { r[1].BenchEvidence.TokenSHA256 = ""; return r }, "2", perfledger.BenchTokensUnverified, perfledger.BenchTokenInvalidDigest},
		{"outside_window", func(r []perfledger.Record) []perfledger.Record { return r }, "1", perfledger.BenchTokensUnverified, perfledger.BenchTokenMissingRequest},
		{"duplicate", func(r []perfledger.Record) []perfledger.Record { return append(r, r[0]) }, "3", perfledger.BenchTokensUnverified, perfledger.BenchTokenAmbiguousRequest},
		{"empty", func(r []perfledger.Record) []perfledger.Record { return nil }, "2", perfledger.BenchTokensUnverified, perfledger.BenchTokenMissingRequest},
	}
	for _, tc := range tests {
		for _, live := range []bool{false, true} {
			mode := "ledger"
			if live {
				mode = "http"
			}
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				records := tc.change(makeRecords())
				args := []string{"--compare-left", "left", "--compare-right", "right", "--json", "--n", tc.n}
				var source string
				if live {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						if r.URL.Path != "/v1/fak/perf/recent" || r.URL.Query().Get("n") != tc.n {
							t.Error("comparison changed HTTP read request")
						}
						rep := perfledger.BuildReport(records, 1024, true, 7)
						rep.Source = "untrusted-server-label"
						_ = json.NewEncoder(w).Encode(rep)
					}))
					defer server.Close()
					args = append(args, "--url", server.URL)
					source = server.URL + "/v1/fak/perf/recent"
				} else {
					source = writePerfLedger(t, records...)
					args = append(args, "--ledger", source)
				}
				var out, errOut bytes.Buffer
				rc := runPerf(&out, &errOut, args)
				wantRC := 1
				if tc.outcome == perfledger.BenchTokensEqual {
					wantRC = 0
				}
				if rc != wantRC {
					t.Fatalf("exit=%d want=%d stderr=%s", rc, wantRC, errOut.String())
				}
				var got perfledger.BenchTokenComparison
				if err := json.Unmarshal(out.Bytes(), &got); err != nil {
					t.Fatal(err)
				}
				if got.Outcome != tc.outcome || got.Reason != tc.reason || got.Source != source {
					t.Fatalf("outcome/reason/source=%s/%s/%s", got.Outcome, got.Reason, got.Source)
				}
				if tc.name == "outside_window" && got.Window.Requests != 1 {
					t.Fatal("comparison escaped selected window")
				}
				if live && (!got.Window.Capped || got.Window.DroppedWrites != 7) {
					t.Fatal("lost bounded-reader uncertainty")
				}
			})
		}
	}
	for _, args := range [][]string{
		{"--compare-left", "left"}, {"--compare-right", "right"},
		{"--compare-left", "", "--compare-right", "right"},
		{"--compare-left", "left", "--compare-right", " "},
		{"--compare-left", "left", "--compare-right", "left"},
		{"--compare-left", "left", "--compare-right", "right", "extra"},
	} {
		var out, errOut bytes.Buffer
		if rc := runPerf(&out, &errOut, args); rc != 2 || out.Len() != 0 {
			t.Fatalf("invalid selection rc=%d stdout=%s", rc, out.String())
		}
	}
	var out, errOut bytes.Buffer
	if rc := runPerf(&out, &errOut, []string{"--ledger", writePerfLedger(t, makeRecords()...), "--compare-left", "left", "--compare-right", "right"}); rc != 0 || !strings.Contains(out.String(), "token_identity=equal") || !strings.Contains(out.String(), "window=2") {
		t.Fatalf("compact comparison exit=%d stdout=%s", rc, out.String())
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
