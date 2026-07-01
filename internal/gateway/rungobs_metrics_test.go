package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/fusedturn"
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

func TestMetricsExposesFusedTurnObserver(t *testing.T) {
	srv := newTestServer(t)
	ctx := context.Background()
	for _, c := range []*abi.ToolCall{
		taggedMetricsCall("turn-fused", fusedturn.ClassClassical),
		taggedMetricsCall("turn-fused", fusedturn.ClassWeight),
		taggedMetricsCall("turn-classical", fusedturn.ClassClassical),
	} {
		srv.k.Syscall(ctx, c)
	}

	text := srv.renderMetrics()
	for _, want := range []string{
		"# HELP fak_fused_turns_total ",
		"# TYPE fak_fused_turns_total counter",
		"fak_fused_turns_total 1",
		"# HELP fak_turn_ops_total ",
		`fak_turn_ops_total{family="classical"} 2`,
		`fak_turn_ops_total{family="weight"} 1`,
		"fak_turns_total 2",
		"# TYPE fak_fused_turn_rate gauge",
		"fak_fused_turn_rate 0.5",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("metrics missing %q\n--- fused metrics ---\n%s", want,
				extractFusedTurnMetrics(text))
		}
	}
}

func taggedMetricsCall(trace string, class fusedturn.OpClass) *abi.ToolCall {
	return fusedturn.Tag(&abi.ToolCall{
		Tool:    "allow_read",
		TraceID: trace,
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
	}, class)
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

func extractFusedTurnMetrics(text string) string {
	var out strings.Builder
	for _, line := range strings.Split(text, "\n") {
		if strings.Contains(line, "fak_fused_turn") || strings.Contains(line, "fak_turn") {
			out.WriteString(line)
			out.WriteByte('\n')
		}
	}
	return out.String()
}
