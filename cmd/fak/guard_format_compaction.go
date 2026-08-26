package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func formatCompactionSummary(sum gateway.AdjudicationSummary) string {
	var section strings.Builder
	if sum.CompactionFired > 0 || sum.CompactionBailed > 0 || sum.CompactionOff > 0 {
		// WITNESSED half only: what fak attempted and removed. The OBSERVED post-fire cache_read
		// is a provider counter (it lives on /metrics) and is noise here — a low value with no
		// prefix_mismatch bail is a provider-side miss fak does not control, not a fak failure.
		// Lead with whether the lever is ENABLED so "0 fired" can't read as "disabled": budget>0
		// with all-under_budget bails is compaction ON and correctly idle (nothing sprawled past
		// the cut), the opposite of OFF.
		// The state VALUE stays short and scannable on the row; the WHY (when it needs one)
		// drops to a demoted note so a tighter-budget misread can't hide, without bloating the
		// row into a paragraph.
		status := fmt.Sprintf("ENABLED, budget %d tok", sum.CompactionBudget)
		statusNote := ""
		if sum.CompactionBudget <= 0 {
			status = "DISABLED (budget 0; body forwarded byte-for-byte)"
		} else if sum.CompactionFired == 0 && sum.CompactionShedTokens == 0 {
			status = fmt.Sprintf("ENABLED but idle, budget %d tok — nothing sprawled past the cut", sum.CompactionBudget)
			// An idle that is NOT a short session: the cache_control anchor protected a prefix
			// larger than the budget, so the lever could not fire no matter the session length.
			// This is the dormant-on-real-Claude-Code-traffic pathology (#1407), the opposite of
			// "nothing sprawled" — call it out so a tighter budget isn't misread as the fix.
			if sum.CompactionAnchorStarved > 0 {
				status = fmt.Sprintf("ENABLED but ANCHOR-STARVED, budget %d tok — cannot fire", sum.CompactionBudget)
				statusNote = "the cache_control anchor protects MORE than the budget so it cannot fire (NOT a short session; --compact-anchor-head is default-on, so either it was disabled or the traffic carries no stable system/tools breakpoint to re-anchor on, #1407)"
			}
		}
		section.WriteString(guardSection("compaction"))
		section.WriteString(guardRow("state", status))
		if statusNote != "" {
			section.WriteString(guardNote(statusNote))
		}
		section.WriteString(guardRow("fired/bailed/off",
			fmt.Sprintf("%d fired, %d bailed, %d off; shed %d token(s)",
				sum.CompactionFired, sum.CompactionBailed, sum.CompactionOff, sum.CompactionShedTokens)))
		// Break the bailed lump out by reason (same shape as the deny "blocked:" loop below):
		// without this, N bailed conflates under_budget (benign, working-as-designed) with
		// no_breakpoint (can't fire) and prefix_mismatch (the ONLY fak-fault cache signal — call
		// it out explicitly when nonzero so a real regression can never hide in the lump).
		if len(sum.CompactionBailReasons) > 0 {
			for _, r := range sortedMapKeys(sum.CompactionBailReasons) {
				section.WriteString(guardRow("  bailed: "+r, fmt.Sprintf("x%d", sum.CompactionBailReasons[r])))
				if r == "prefix_mismatch" || r == "splice_failed" || r == "redecode_failed" {
					section.WriteString(guardNote("⚠ fak-fault: a fired rewrite would have burst the cache — must stay 0"))
				}
			}
		}
		// Anchor-starved is a SUBSET of the under_budget bails above, surfaced apart because it is
		// operationally opposite: a plain under_budget is a benign short session, an anchor-starved
		// one means the anchor swallowed the conversation so no budget tightening can ever make it
		// fire — only a re-anchor (#1407 / opt-in head-anchored firing #1408) can.
		if sum.CompactionAnchorStarved > 0 {
			section.WriteString(guardRow("  ⚠ anchor-starved", fmt.Sprintf("x%d", sum.CompactionAnchorStarved)))
			section.WriteString(guardNote(fmt.Sprintf("protected prefix exceeds the %d-tok budget; compaction cannot fire on this traffic regardless of session length (a re-anchor is the fix, not a tighter budget: --compact-anchor-head is default-on, so either it was disabled or the traffic carries no stable system/tools breakpoint — #1407)", sum.CompactionBudget)))
		}
		// Solvency-forced is a SUBSET of the fires above, surfaced apart because it is economically
		// opposite: an ordinary fire repays in cache dollars, a forced one is a burst knowingly taken
		// at a LOSS to keep the session inside its context window (--compact-solvency-floor). Not a
		// fault — the override doing its job — but it must never be read as a cache win, and a run
		// where forced fires dominate is telling the operator the window, not the cache, is what
		// binds this workload.
		if sum.CompactionSolvencyForced > 0 {
			section.WriteString(guardRow("  solvency-forced", fmt.Sprintf("x%d of %d fired", sum.CompactionSolvencyForced, sum.CompactionFired)))
			section.WriteString(guardNote("the burst economics REFUSED these and --compact-solvency-floor fired them anyway: deliberately unprofitable sheds bought to stay inside the context window. Count them as survival, not savings; if they dominate, the window is the binding constraint on this workload"))
		}
	}

	return section.String()
}
