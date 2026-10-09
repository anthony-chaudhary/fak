package gateway

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

func scrapePerfMetrics(t *testing.T, srv *Server) string {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	req.RemoteAddr = "127.0.0.1:50102"
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("/metrics status = %d", rec.Code)
	}
	return rec.Body.String()
}

func perfMetricLine(body, prefix string) string {
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, prefix+" ") {
			return line
		}
	}
	return ""
}

// fak-test:runtime fast est=500ms lane=default
func TestPerfRowCarriesClientAndSyntheticCanaryStaysOutOfLatency(t *testing.T) {
	srv := newTestServer(t)
	srv.SetPerfLedger(nil, nil, false)

	canaryReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	canaryReq.Header.Set(perfledger.ProbeHeader, "halo-canary")
	agentReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	agentReq.Header.Set("User-Agent", "opencode/1.2.3 ai-sdk/5")

	// The witnessed Halo canary: a cold 25-token prompt the size heuristic misses.
	srv.metrics.observeInferenceServedTimedFrom(perfSourceFromRequest(canaryReq), localitySelfHosted, "", 25, 21, 0, 0, "stop", 300*time.Millisecond, 100*time.Millisecond)
	srv.metrics.observeInferenceServedTimedFrom(perfSourceFromRequest(agentReq), localitySelfHosted, "", 4000, 60, 16000, 0, "stop", 9*time.Second, 7*time.Second)

	rep := decodePerfReport(t, getPerfRecent(t, srv, "n=10&format=json"))
	if len(rep.Records) != 2 {
		t.Fatalf("rows = %d, want 2", len(rep.Records))
	}
	canary, served := rep.Records[0], rep.Records[1]
	if !canary.Synthetic || canary.Client != perfledger.ClientProbe {
		t.Fatalf("canary row client/synthetic = %q/%v, want probe/true", canary.Client, canary.Synthetic)
	}
	if served.Synthetic || served.Client != perfledger.ClientOpenCode {
		t.Fatalf("served row client/synthetic = %q/%v, want opencode/false", served.Client, served.Synthetic)
	}
	if rep.Summary.Count != 1 || rep.Summary.Probes != 1 {
		t.Fatalf("summary count/probes = %d/%d, want 1/1", rep.Summary.Count, rep.Summary.Probes)
	}
	if got := rep.Summary.ByRegime["cold"].Count; got != 0 {
		t.Fatalf("cold regime count = %d, want 0 (the canary is synthetic)", got)
	}

	body := scrapePerfMetrics(t, srv)
	if line := perfMetricLine(body, "fak_gateway_inference_synthetic_turns_total"); line != "fak_gateway_inference_synthetic_turns_total 1" {
		t.Fatalf("synthetic counter line = %q, want 1", line)
	}
	if line := perfMetricLine(body, "fak_gateway_inference_ttft_seconds_count"); line != "fak_gateway_inference_ttft_seconds_count 1" {
		t.Fatalf("ttft histogram count line = %q, want only the served turn", line)
	}
	if line := perfMetricLine(body, `fak_gateway_inference_ttft_by_regime_seconds_count{regime="cold"}`); line != "" {
		t.Fatalf("cold regime row present = %q, want absent", line)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestPerfSourceIsUnrecordedOffTheHTTPEdge(t *testing.T) {
	if src := perfSourceFrom(t.Context()); src != (perfSource{}) {
		t.Fatalf("source off the edge = %+v, want zero", src)
	}
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	ctx := withPerfSource(req.Context(), perfSourceFromRequest(req))
	if src := perfSourceFrom(ctx); src.client != perfledger.ClientUnknown || src.synthetic {
		t.Fatalf("source with no headers = %+v, want unknown/false", src)
	}
}
