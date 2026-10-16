package gateway

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// fak-test:runtime fast est=10ms lane=default
func TestPerfRowCarriesInboundRequestID(t *testing.T) {
	s := newTestServer(t)
	var rows []perfledger.Record
	h := s.withMetrics(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		ctx, _, _, _, _ := s.beginServedRequest(w, r)
		var row perfledger.Record
		perfSourceFrom(ctx).apply(&row)
		rows = append(rows, row)
	}))
	cases := []struct {
		name, requestID, traceparent, want string
	}{
		{"router id", "rt-0123456789abcdef", "", "rt-0123456789abcdef"},
		{"traceparent fallback", "", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", "4bf92f3577b34da6a3ce929d0e0e4736"},
		{"unsafe id falls back", "bad id\x7f", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", "4bf92f3577b34da6a3ce929d0e0e4736"},
		{"oversized id dropped", strings.Repeat("a", maxInboundRequestIDLen+1), "", ""},
		{"none", "", "", ""},
	}
	for _, tc := range cases {
		r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("{}"))
		if tc.requestID != "" {
			r.Header.Set("X-Request-ID", tc.requestID)
		}
		if tc.traceparent != "" {
			r.Header.Set("traceparent", tc.traceparent)
		}
		h.ServeHTTP(httptest.NewRecorder(), r)
		row := rows[len(rows)-1]
		if row.InboundRequestID != tc.want {
			t.Errorf("%s: inbound_request_id = %q, want %q", tc.name, row.InboundRequestID, tc.want)
		}
		if row.RequestID == "" || row.RequestID == tc.requestID {
			t.Errorf("%s: fak request_id must stay fak-minted, got %q", tc.name, row.RequestID)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestReadGPUBusyAbsentNotZero(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	if g := readGPUBusy(write("ok", "37\n")); g == nil || *g != 37 {
		t.Fatalf("readable counter = %v, want 37", g)
	}
	if g := readGPUBusy(write("zero", "0\n")); g == nil || *g != 0 {
		t.Fatalf("idle counter = %v, want measured 0", g)
	}
	for _, p := range []string{"", filepath.Join(dir, "missing"), write("junk", "busy"), write("range", "140")} {
		if g := readGPUBusy(p); g != nil {
			t.Fatalf("readGPUBusy(%q) = %d, want absent", p, *g)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCommitPerfSamplesGPUBusy(t *testing.T) {
	prev := sampleGPUBusy
	t.Cleanup(func() { sampleGPUBusy = prev })
	s := newTestServer(t)

	busy := 61
	sampleGPUBusy = func() *int { v := busy; return &v }
	s.metrics.commitPerf(perfledger.Record{E2EMS: 10})
	sampleGPUBusy = func() *int { return nil }
	s.metrics.commitPerf(perfledger.Record{E2EMS: 11})

	rows, _, _ := s.metrics.perfRecordsSnapshot()
	if len(rows) != 2 || rows[0].GPUBusyPercent == nil || *rows[0].GPUBusyPercent != busy || rows[1].GPUBusyPercent != nil {
		t.Fatalf("gpu busy not recorded as sampled/absent: %+v", rows)
	}
}
