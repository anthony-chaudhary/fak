package gateway

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

func metricLines(text string) map[string]bool {
	lines := map[string]bool{}
	for _, l := range strings.Split(text, "\n") {
		lines[strings.TrimSpace(l)] = true
	}
	return lines
}

// TestFusionCacheAdvantageRatioAgainstBaselines pins the advantage arithmetic: saved and
// ratio versus no reuse and versus a same-session-only cache, with an unreported turn
// excluded from both sides rather than counted as a miss.
//
// fak-test:runtime fast est=50ms
func TestFusionCacheAdvantageRatioAgainstBaselines(t *testing.T) {
	var a prefixReuseAttribution
	a.observe("", 1, 100, 0, 10)             // coordinator cold, footprint 110
	a.observe("", 2, 100, 50, 10)            // sibling: 50 cross
	a.observe("", 1, 200, 100, 10)           // coordinator: 100 same
	a.observeTurn("", 1, 1000, 0, 10, false) // unreported: excluded
	var b strings.Builder
	writeFusionCacheMetrics(&b, a.snapshot())
	lines := metricLines(b.String())
	for _, want := range []string{
		`fak_gateway_fusion_cache_prompt_tokens_total 400`,
		`fak_gateway_fusion_cache_prefilled_tokens_total 250`,
		`fak_gateway_fusion_cache_saved_tokens_total{baseline="no_reuse"} 150`,
		`fak_gateway_fusion_cache_saved_tokens_total{baseline="same_session_only"} 50`,
		`fak_gateway_fusion_cache_advantage_ratio{baseline="no_reuse"} 1.6`,
		`fak_gateway_fusion_cache_advantage_ratio{baseline="same_session_only"} 1.2`,
	} {
		if !lines[want] {
			t.Errorf("missing %q\n%s", want, b.String())
		}
	}

	var empty strings.Builder
	writeFusionCacheMetrics(&empty, prefixReuseSnapshot{})
	if strings.Contains(empty.String(), "advantage_ratio") {
		t.Errorf("ratio emitted with no prefilled tokens:\n%s", empty.String())
	}
}

// TestFusionCacheSeriesOnOperatorMetrics is the today-gap witness: /metrics carries the
// fak_gateway_fusion_cache_* family by default.
//
// fak-test:runtime fast est=2s
func TestFusionCacheSeriesOnOperatorMetrics(t *testing.T) {
	ts := principalTestServer(t, &scopeCapturePlanner{usage: [][2]int{{100, 0}, {100, 60}}}, "tenant-a")
	body := `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`
	postPrefixReuse(t, ts.URL+"/v1/chat/completions", body, nil)
	postPrefixReuse(t, ts.URL+"/v1/chat/completions", body, nil)
	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	lines := metricLines(string(raw))
	for _, want := range []string{
		`fak_gateway_fusion_cache_prompt_tokens_total 200`,
		`fak_gateway_fusion_cache_prefilled_tokens_total 140`,
		`fak_gateway_fusion_cache_saved_tokens_total{baseline="no_reuse"} 60`,
	} {
		if !lines[want] {
			t.Errorf("/metrics missing %q", want)
		}
	}
}
