package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// TestCacheBootSummary pins the startup cache-state banner line (epic #1072): idle at boot
// (no served turn) names where reuse will appear and the metric family to scrape; once turns
// accumulate it reports the WITNESSED realized reuse ratio, the absolute tokens saved, and the
// per-regime split (#1076).
func TestCacheBootSummary(t *testing.T) {
	idle := cacheBootSummary(cacheobs.Stats{})
	for _, want := range []string{"idle", "fak_gateway_kv_prefix_"} {
		if !strings.Contains(idle, want) {
			t.Errorf("idle summary = %q; missing %q", idle, want)
		}
	}
	active := cacheBootSummary(cacheobs.Stats{Turns: 4, PromptTokens: 8000, ReusedTokens: 7000, ReuseRatio: 0.875, FrozenTurns: 3, PartialTurns: 1})
	for _, want := range []string{"reuse 88%", "saved=7000 tok", "4 turns", "frozen=3", "partial=1", "by=vdso"} {
		if !strings.Contains(active, want) {
			t.Errorf("active summary = %q; missing %q", active, want)
		}
	}
}
