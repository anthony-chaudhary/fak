package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// startupReportStub serves /debug/vars carrying the given startup_report — the shape a
// live `fak guard` gateway exposes after SetStartupReport at boot.
func startupReportStub(t *testing.T, report string) *httptest.Server {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"gateway":        map[string]any{"uptime_seconds": 1.0},
		"startup_report": report,
	})
	if err != nil {
		t.Fatalf("marshal stub: %v", err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/debug/vars" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestRunInfoStartupPrintsRecordedReport is the read-side witness for the compact-banner
// feature: the full startup report an attended `fak guard` launch kept off the terminal
// is one `fak info --startup` away, verbatim, for the session's whole life.
func TestRunInfoStartupPrintsRecordedReport(t *testing.T) {
	const report = "fak guard 9.9.9 — kernel-adjudicated: claude\n  floor      : built-in guard floor\n  every tool call the agent proposes crosses the capability floor before it runs.\n"
	srv := startupReportStub(t, report)

	var stdout, stderr bytes.Buffer
	code := runInfo(&stdout, &stderr, []string{"--gateway-url", srv.URL, "--startup"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	if stdout.String() != report {
		t.Fatalf("stdout = %q, want the recorded report verbatim %q", stdout.String(), report)
	}
}

// TestRunInfoStartupWithoutReportFailsActionably: a gateway that answers but recorded no
// report (fak serve, an older guard) exits non-zero with a next step, never an empty page
// that reads as "the report was blank".
func TestRunInfoStartupWithoutReportFailsActionably(t *testing.T) {
	srv := startupReportStub(t, "")

	var stdout, stderr bytes.Buffer
	code := runInfo(&stdout, &stderr, []string{"--gateway-url", srv.URL, "--startup"})
	if code == 0 {
		t.Fatalf("exit = 0 for a gateway with no recorded report, want non-zero")
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want empty (nothing to print)", stdout.String())
	}
	for _, want := range []string{"no startup report", "--banner=full"} {
		if !bytes.Contains(stderr.Bytes(), []byte(want)) {
			t.Errorf("stderr missing %q: %s", want, stderr.String())
		}
	}
}
