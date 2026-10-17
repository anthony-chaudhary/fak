package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestPrefillFallbackCounterReachesMetricsSurface pins the wire between the
// model's process-wide prefill-fallback counter and the gateway /metrics scrape,
// so a silent drop to scalar token replay is visible on the operator surface.
func TestPrefillFallbackCounterReachesMetricsSurface(t *testing.T) {
	model.DeclarePrefillFallbackReason(model.Qwen35SequencePrefillDeclineEmbeddingCap)
	model.CountPrefillFallback(model.Qwen35SequencePrefillDeclineEmbeddingCap)

	out := newTestServer(t).renderMetrics()
	for _, want := range []string{
		"# HELP fak_prefill_fallback_total ",
		"# TYPE fak_prefill_fallback_total counter",
		`fak_prefill_fallback_total{reason="` + model.Qwen35SequencePrefillDeclineEmbeddingCap + `"} `,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("/metrics missing %q", want)
		}
	}
}
