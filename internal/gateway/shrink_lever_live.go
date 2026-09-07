package gateway

// shrink_lever_live.go — the LIVE half of the prompt-shrink-lever wire gap (#5493).
//
// The three largest prompt-shrink levers this gateway ships are each gated, inside the
// gateway, on the Anthropic passthrough decision (anthropicPassthroughFor):
//
//   - --compact-history-budget → compactAnthropicRawWithReason  (messages.go)
//   - --elide-stale-reads      → maybeElideStaleReads           (messages.go)
//   - --defer-cold-tools       → maybeDeferColdTools            (messages_tooldefer.go)
//
// All three mutate req.Raw and nothing else, and req.Raw is forwarded verbatim ONLY on that
// passthrough — the generic planner path rebuilds the upstream request from the decoded
// req.Messages and never reads req.Raw at all. So off the passthrough the levers are not
// merely unused, they are structurally unreachable: the bytes they edit are discarded.
//
// cmd/fak/shrink_lever_wire.go closes the STARTUP half of that gap — it refuses a boot whose
// operator explicitly enabled an inert lever and prints a named notice for the default-on
// ones. This file closes the half that outlives the boot line. A process is read long after
// stderr has scrolled: by an A/B harness, by `fak info`, by a scrape of /debug/vars. Until
// this block, none of those could tell "the lever ran and saved nothing" from "the lever was
// never on this wire in the first place" — and the second-order harm in the issue is exactly
// that confusion being written up as a verdict on the kernel.
//
// WHY THE EXISTING COUNTERS CANNOT ANSWER IT. The neighbouring #3621 watchdog
// (defer_inert.go) looks like it should: it raises DEFER_ENABLED_BUT_INERT when the defer
// lever armed and never bit. But its stand-down rows accrue exclusively PAST
// maybeDeferColdTools' eligibility gate, and that gate includes the passthrough test — so on
// a non-Anthropic wire the transform returns before booking anything and the watchdog is
// silent BY CONSTRUCTION (defer_inert.go says so in as many words). Every defer counter is a
// pure numerator; a foreign wire renders a flat zero that is indistinguishable from a
// lever-off session. The missing fact is not per-turn, it is per-WIRE, and that is what this
// block carries.

import (
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

// The wire shape lives in internal/guardvars beside CacheAttributionVars and ManagedCacheVars
// so the `fak info` pane and any scraper decode one definition rather than a hand-synced copy.
type debugShrinkLeverVars = guardvars.ShrinkLeverVars

// wireRunsShrinkLevers reports whether this server's wire runs prompt-shrink levers.
// True for the Anthropic passthrough and for native in-kernel serving (InKernelPlanner).
func (s *Server) wireRunsShrinkLevers() bool {
	if s == nil {
		return false
	}
	if s.anthropicPassthrough() {
		return true
	}
	_, inKernel := s.planner.(*agent.InKernelPlanner)
	return inKernel
}

// shrinkLeverVars builds the /debug/vars prompt-shrink-lever posture block from the resolved
// wire (wireRunsLevers, indicating whether the wire runs levers; dualLocal; provider) and the
// three configured lever values as gateway.Config carries them.
//
// It returns nil — omitting the block — when no lever is configured ON at all, so a session
// that opted out of all three stays quiet instead of rendering an empty object. Every other
// case renders, including the healthy all-live passthrough or in-kernel wire: "which levers are live"
// is only a useful answer if the surface is also present when the answer is a good one, otherwise the
// block's presence alone becomes the finding and an operator learns to ignore it.
//
// Pure: flags in, block out. No planner call, no lock, no I/O.
func shrinkLeverVars(wireRunsLevers, dualLocal bool, provider string, compactHistoryBudget int, elideStaleReads, deferColdTools bool) *debugShrinkLeverVars {
	// The order is fixed (compaction, stale-read elision, cold-tool defer) to match the
	// startup admission's table, so the two surfaces name the levers in the same sequence.
	configured := []struct {
		token string
		on    bool
	}{
		{guardvars.ShrinkLeverCompactHistoryBudget, compactHistoryBudget > 0},
		{guardvars.ShrinkLeverElideStaleReads, elideStaleReads},
		{guardvars.ShrinkLeverDeferColdTools, deferColdTools},
	}
	block := &debugShrinkLeverVars{WireRunsLevers: wireRunsLevers, Wire: provider, DualLocalRouting: dualLocal}
	for _, l := range configured {
		if !l.on {
			continue
		}
		if wireRunsLevers {
			block.LiveOnWire = append(block.LiveOnWire, l.token)
		} else {
			block.InertOnWire = append(block.InertOnWire, l.token)
		}
	}
	if len(block.LiveOnWire) == 0 && len(block.InertOnWire) == 0 {
		return nil
	}
	if len(block.InertOnWire) > 0 {
		block.Finding = guardvars.FindingShrinkLeverInertOnWire
	}
	return block
}

// dualRoutesLocalModels reports whether this server's planner is the DUAL planner, i.e. an
// API upstream running alongside in-kernel weights. There the passthrough decision is
// per-REQUEST (anthropicPassthroughFor consults RoutesLocal on the request's model id), so
// the process-wide posture above is true for the proxy-bound turns and wrong for the
// locally-served ones. Surfacing the fact that such routing exists is honest where claiming a
// single answer would not be; naming WHICH model ids route local would put the operator's
// model names on a debug surface for no diagnostic gain.
func (s *Server) dualRoutesLocalModels() bool {
	if s == nil {
		return false
	}
	_, dual := s.planner.(*DualPlanner)
	return dual
}
