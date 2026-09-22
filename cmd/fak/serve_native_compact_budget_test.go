package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestServePlannerWiresNativeCompactBudget is the S1/S3 production-seam witness:
// the planner `fak serve` actually builds must carry a compaction shed-line
// derived from the resolved window. Before this wiring the field was left zero by
// every production constructor, so ApplyPromptShrink was a no-op on the native
// path and an over-window transcript was refused with HTTP 400
// context_length_exceeded instead of being compacted.
func TestServePlannerWiresNativeCompactBudget(t *testing.T) {
	newFlags := func(t *testing.T, argv ...string) *serveFlags {
		t.Helper()
		fs, sf := newServeFlagSet()
		if err := fs.Parse(argv); err != nil {
			t.Fatalf("parse %v: %v", argv, err)
		}
		return sf
	}

	// Auto: derived from the resolved window.
	t.Run("derived from resolved window", func(t *testing.T) {
		sf := newFlags(t)
		cfg := serveNativePlannerConfigWithContext(sf, 150_000)
		if cfg.ContextTokens != 150_000 {
			t.Fatalf("ContextTokens = %d, want 150000", cfg.ContextTokens)
		}
		want := agent.DeriveCompactHistoryBudget(150_000, 0)
		if want <= 0 {
			t.Fatalf("derivation produced %d; the test premise is that a 150k window yields a positive budget", want)
		}
		if cfg.CompactHistoryBudget != want {
			t.Fatalf("CompactHistoryBudget = %d, want %d (derived from the resolved window)", cfg.CompactHistoryBudget, want)
		}
		if cfg.CompactHistoryBudget <= gateway.DefaultCompactHistoryBudget {
			t.Fatalf("native budget %d is at or below the Anthropic passthrough default %d; a 150k native window must not be flattened to the provider-shaped default",
				cfg.CompactHistoryBudget, gateway.DefaultCompactHistoryBudget)
		}
	})

	// Explicit flag wins over the derivation.
	t.Run("explicit flag wins", func(t *testing.T) {
		sf := newFlags(t, "--native-compact-history-budget", "12345")
		if got := serveNativePlannerConfigWithContext(sf, 150_000).CompactHistoryBudget; got != 12345 {
			t.Fatalf("CompactHistoryBudget = %d, want the explicit 12345", got)
		}
	})

	// No resolved window => no honest shed-line; keep the historical no-op so the
	// change is inert for a caller that cannot state a bound.
	t.Run("unresolved window stays inert", func(t *testing.T) {
		sf := newFlags(t)
		if got := serveNativePlannerConfigWithContext(sf, 0).CompactHistoryBudget; got != 0 {
			t.Fatalf("CompactHistoryBudget = %d for an unresolved window, want 0 (no-op preserved)", got)
		}
	})
}
