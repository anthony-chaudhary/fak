package gateway

import "github.com/anthony-chaudhary/fak/internal/metrics"

// The ANCHOR_REFUSED_RISING live watchdog (#3622, cache-verify epic #3569): the offensive
// cache-breakpoint placer (the star-anchor) splices a cache_control breakpoint onto a stable
// system[]/tools[] head, and `fak_gateway_cache_breakpoint_placement_total{outcome}` already
// records every attempt over the closed agent.BreakpointReason* vocabulary. That family is a
// pure CUMULATIVE tally, and a cumulative tally is a value nobody reads: when a session's head
// turns volatile across turns (every cacheable head span now carries a per-request token) the
// anchor quietly stops EARNING caching, the "placed" bucket simply stops rising, and the surface
// is indistinguishable from a session that went idle. Nothing fires.
//
// internal/metrics/anchor_refusal.go is the pure fold that watches the mix — a rolling refused
// fraction over DECISIVE turns, edge-triggered so a long volatile stretch raises once rather than
// becoming a stuck horn. This file is the gateway's half: metrics_observe.go now feeds every
// observePlacement outcome through the monitor under the lock it already held, and the two
// predicates below lower the session's verdict onto the operator surface (the guard exit banner,
// cmd/fak formatAuditSummary) — the same producer/consumer split the sibling
// DEFER_ENABLED_BUT_INERT watchdog (defer_inert.go) uses, so an operator reads the two alike.
//
// The false-positive guard lives in the fold, not here, and is the whole design: `already_set` —
// the Claude-Code shape, where the client authored its own cache_control and the turn IS cached —
// is DEFERRED, never refused. A Claude-Code-shaped session is ~100% already_set by construction,
// so pricing that as a refusal would alarm every healthy client on earth, which is how a monitor
// becomes a thing operators mute. Only earned + refused move the fraction.

// AnchorRefusedRising reports the #3622 finding for the session: the rolling refused fraction
// crossed the threshold at least once, so the star-anchor stopped earning caching for a stretch
// long enough to be worth an operator's attention.
//
// It keys on the CROSSING COUNT, not the end state. A session that turned volatile and later
// recovered ends with Alarmed=false, but it still spent turns paying uncached — an exit artifact
// that reported only the final instant would hide exactly the mid-session degradation this
// watchdog exists to name. AnchorRefusalAlarmed below is the end-state twin for a live pane.
//
// A session that never attempted a placement (non-Anthropic wire, cold process, a body the anchor
// never applied to) carries no report at all and can never raise it.
func (s AdjudicationSummary) AnchorRefusedRising() bool {
	return s.AnchorPlacement != nil && s.AnchorPlacement.Findings > 0
}

// AnchorRefusalAlarmed reports whether the session ENDED in the raised state — the instantaneous
// twin of AnchorRefusedRising, for a live pane that should clear when the anchor recovers.
func (s AdjudicationSummary) AnchorRefusalAlarmed() bool {
	return s.AnchorPlacement != nil && s.AnchorPlacement.Alarmed
}

// AnchorRefusalBanner is the session's one-line operator row, or "" when the session never
// attempted a placement. The line itself is rendered by the pure fold (metrics.AnchorRefusalReport
// .BannerRow), so the guard exit banner and any other consumer can never word the same verdict two
// different ways.
func (s AdjudicationSummary) AnchorRefusalBanner() string {
	if s.AnchorPlacement == nil {
		return ""
	}
	return s.AnchorPlacement.BannerRow()
}

// AnchorRefusalOutcomes returns just the REFUSED buckets of the session's placement mix, in the
// report's own deterministic order (turns desc, then outcome). This is the drilldown the banner
// prints under the finding — the operator's first question after "the anchor stopped earning" is
// which bail it was, since volatile_head (the caller's head carries a per-request token) and
// no_stable_head (there is no system[]/tools[] block at all) point at different fixes.
func (s AdjudicationSummary) AnchorRefusalOutcomes() []metrics.AnchorOutcomeTally {
	if s.AnchorPlacement == nil {
		return nil
	}
	var out []metrics.AnchorOutcomeTally
	for _, t := range s.AnchorPlacement.ByOutcome {
		if t.Class == metrics.AnchorRefused {
			out = append(out, t)
		}
	}
	return out
}
