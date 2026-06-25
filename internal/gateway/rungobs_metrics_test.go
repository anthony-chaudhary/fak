package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// AC #5 (issue #693) — /metrics exposes the rung-decision distribution as a labeled
// prometheus counter fak_kernel_decisions_total{rung,kind,reason}. The gateway
// registers a passive rungobs Emitter in New; one adjudicated call is enough to give
// it a decision to bucket, and the metric family (HELP + TYPE + one series) must
// appear on the scrape. Mirrors the substring style of the existing metrics test.
func TestMetricsExposesRungDecisionDistribution(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Drive one adjudicated allow so the observer has a structural decision to bucket
	// (allow_read reaches the engine, so it is NOT a vDSO hit).
	var resp SyscallResponse
	if code := postJSON(t, ts.URL+"/v1/fak/syscall", SyscallRequest{
		Tool:      "allow_read",
		Arguments: json.RawMessage(`{"x":1}`),
		ReadOnly:  true,
	}, &resp); code != http.StatusOK || resp.Verdict.Kind != "ALLOW" {
		t.Fatalf("syscall allow_read: status=%d verdict=%+v (want 200/ALLOW)", code, resp.Verdict)
	}

	text := getMetrics(t, ts.URL+"/metrics", "")
	for _, want := range []string{
		"# HELP fak_kernel_decisions_total ",
		"# TYPE fak_kernel_decisions_total counter",
		`fak_kernel_decisions_total{rung="`,
		`,kind="ALLOW"`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("metrics missing %q\n--- metrics (decisions) ---\n%s", want,
				extractDecisions(text))
		}
	}
}

// extractDecisions returns just the fak_kernel_decisions_total lines for a readable
// failure message.
func extractDecisions(text string) string {
	var out strings.Builder
	for _, line := range strings.Split(text, "\n") {
		if strings.Contains(line, "fak_kernel_decisions_total") {
			out.WriteString(line)
			out.WriteByte('\n')
		}
	}
	return out.String()
}
