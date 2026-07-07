package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
)

// TestFireGateDefaultsPinnedToCacheprice is the fire-gate half of the #2798 acceptance:
// the head-anchored burst gate's default cache-read / cache-write multipliers
// (defaultCacheReadMult / defaultCacheWriteMult in anthropic_compact.go) MUST equal the
// canonical cacheprice values that the gateway pricing model, the resume planner, the
// net-true ledger and the Track-2 report all read. That equality is what makes the fire
// gate and the report value an identical compaction-shed token identically — the single
// source of truth #2798 requires.
//
// The constants themselves still live as literals in anthropic_compact.go rather than
// reading cacheprice.* directly: that file carries concurrent work in this shared tree, so
// folding the two consts to `= cacheprice.ReadMultiplier` / `= cacheprice.Write5mMultiplier`
// is the tracked follow-on. Until then this real cross-package symbol pin — not a bare-0.1
// comment — is what fails the moment the fire gate drifts from the canonical.
func TestFireGateDefaultsPinnedToCacheprice(t *testing.T) {
	if defaultCacheReadMult != cacheprice.ReadMultiplier {
		t.Fatalf("fire-gate defaultCacheReadMult %.3f drifted from canonical cacheprice.ReadMultiplier %.3f (#2798) — fold the const to read cacheprice.ReadMultiplier",
			defaultCacheReadMult, cacheprice.ReadMultiplier)
	}
	if defaultCacheWriteMult != cacheprice.Write5mMultiplier {
		t.Fatalf("fire-gate defaultCacheWriteMult %.3f drifted from canonical cacheprice.Write5mMultiplier %.3f (#2798)",
			defaultCacheWriteMult, cacheprice.Write5mMultiplier)
	}
}
