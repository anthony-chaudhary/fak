package gateway

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

func (m *gatewayMetrics) writeCompactionMetrics(b *strings.Builder) {
	snap := m.compactionSnapshotData()

	writeHelpType(b, "fak_gateway_compaction_attempts_total",
		"WITNESSED (fak authored): Anthropic history-compaction attempts by outcome: fired (body rewritten, protected prefix shipped byte-identical), bailed (returned identity), off (budget unset).", "counter")
	for _, o := range []string{"fired", "bailed", "off"} { // stable order; emit at 0 so the panel exists pre-first-fire
		fmt.Fprintf(b, "fak_gateway_compaction_attempts_total{outcome=%q} %d\n", o, snap.attempts[o])
	}

	// The closed-set claim is DERIVED from agent.CompactBailReasons(), never re-typed here.
	// Hand-spelling it drifted twice: the string declared 9 members while internal/agent
	// emitted 13, so decode_failed (#5387) and three others were invisible to anyone who
	// trusted the declaration and built an alert over the listed labels (#5441). The emitter
	// owns the vocabulary; this rendering reads it, and agent's own registration test fails if
	// a CompactReason* constant is added without joining it.
	writeHelpType(b, "fak_gateway_compaction_bail_reason_total",
		"WITNESSED (fak authored): why a compaction attempt bailed to identity (closed set, derived from the emitting package's registered vocabulary: "+
			strings.Join(agent.CompactBailReasons(), "|")+"). prefix_mismatch>0 is the ONLY fak-fault cache signal and must stay 0; splice_failed/redecode_failed are splice bugs and must stay 0 too.", "counter")
	rs := make([]string, 0, len(snap.bailReasons))
	for r := range snap.bailReasons {
		rs = append(rs, r)
	}
	sort.Strings(rs)
	for _, r := range rs {
		fmt.Fprintf(b, "fak_gateway_compaction_bail_reason_total{reason=%q} %d\n", promQuote(r), snap.bailReasons[r])
	}

	// The ALERTABLE compaction-health pair (#5443). attempts{bailed} above counts every
	// non-fire, including the requests that were never compaction candidates at all — a
	// two-message subagent ping cannot compact at any setting — and on mixed fleet traffic
	// those dominate, pinning bails/(fires+bails) near 1.0 whatever the compactor does. These
	// two series are added ALONGSIDE the unchanged counters rather than redefining them: the
	// held-out population is published as its own counter so it is visible, not silently
	// subtracted, and the rate that divides only over eligible attempts is published beside it.
	// The same partition drives internal/gatewayusageledger's NonCandidateBails /
	// CandidateBailRate, so the live scrape and the durable ledger fold agree.
	nonCandidateBails, candidateBails := compactBailPartition(snap.bailReasons)
	writeCounter(b, "fak_gateway_compaction_non_candidate_bails_total",
		"WITNESSED (fak authored): the SUBSET of compaction bails the compactor decided before any compactible span existed (non_json, no_messages_key, decode_failed, too_few_msgs — see the bail_reason vocabulary). These requests were never in the running, so they say nothing about compaction health; they are held out of fak_gateway_compaction_candidate_bail_rate and published here so the held-out population stays visible. A cell whose bails are almost all these is a normal short-request stream, not a sick compactor. Subtract it from attempts{outcome=\"bailed\"} to recover the eligible bails.",
		int64(nonCandidateBails))
	writeHelpType(b, "fak_gateway_compaction_candidate_bail_rate",
		"WITNESSED (fak authored): eligible bails / (fires + eligible bails) — compaction declines over attempts that actually HAD a compactible span. This is the alertable rate; bails/(fires+bails) from attempts{} is not, because it counts requests that were never compactible and therefore reads near 1.0 on healthy and broken traffic alike. 0 when no attempt was ever eligible (an honest zero, not a fabricated ratio). A reason the emitting package has not registered counts as ELIGIBLE, so this can only read conservatively high — it never silently understates a problem.",
		"gauge")
	fmt.Fprintf(b, "fak_gateway_compaction_candidate_bail_rate %s\n", promFloat(compactCandidateBailRate(snap.attempts["fired"], candidateBails)))

	writeCounter(b, "fak_gateway_compaction_anchor_starved_total", "WITNESSED (fak authored): under_budget bails whose protected prefix ALREADY exceeded the budget — the cache_control anchor swallowed the conversation, so compaction structurally cannot fire no matter how long the session grows. A SUBSET of bail_reason{reason=\"under_budget\"}, broken out because the two are opposite: plain under_budget is a benign short session, anchor-starved is the dormant-on-real-Claude-Code-traffic pathology (issue #1407) that no budget tightening fixes.", int64(snap.anchorStarved))
	writeCounter(b, "fak_gateway_compaction_thrash_sessions_total", "WITNESSED (fak authored): sessions that hit the closed verdict COMPACTION_THRASH (#2424) — the context window refilled to the compaction limit on 3 consecutive turns, so the lever fired every turn and bought no lasting headroom. Counted ONCE per thrashing stretch, so this is sessions-that-thrashed, not turns-spent-thrashing. It is NOT a bail (compaction FIRED each time), which is why it is published here instead of under bail_reason_total, whose label set stays the compactor's own closed vocabulary. Nonzero means the binding constraint is the traffic, not the budget: tightening the budget compacts more often and changes nothing. Detection is always on; the session STOP that acts on it is opt-in behind FAK_COMPACTION_THRASH_STOP.", int64(snap.thrashSessions))
	writeCounter(b, "fak_gateway_compaction_dropped_turns_total", "WITNESSED (fak authored): whole messages stubbed out across all fires.", int64(snap.dropped))
	writeCounter(b, "fak_gateway_compaction_shed_tokens_total", "WITNESSED (fak authored): estimated tokens fak removed from the outbound body by history compaction fires plus uncached-tail trim (same ~4ch/token currency as the budget and provider input_tokens). What fak SENT — not a claim about what the provider billed.", int64(snap.shed))
	writeCounter(b, "fak_gateway_compaction_cache_read_tokens_total", "OBSERVED (provider-reported, relayed verbatim): cumulative cache_read_input_tokens on compacted turns. Pair with shed_tokens to see the net effect; attribute nothing to fak from it alone — fak only guarantees the prefix it shipped was byte-identical (see attempts{fired} with prefix_mismatch=0).", int64(snap.cacheReads))
	writeHelpType(b, "fak_gateway_compaction_post_fire_cache_read_tokens",
		"OBSERVED (provider-reported): cache_read_input_tokens on the MOST RECENT compacted turn. If this craters while fires climb, the prefix fak shipped was still byte-identical (witnessed by fired with prefix_mismatch=0), so the provider did not reuse it for a reason fak does NOT control: cache TTL expiry, eviction, or the client moving its own breakpoint. Only bail_reason{reason=\"prefix_mismatch\"}>0 is fak's bug.", "gauge")
	fmt.Fprintf(b, "fak_gateway_compaction_post_fire_cache_read_tokens %s\n", promFloat(snap.lastCacheRd))
	writeCounter(b, "fak_gateway_uncached_trim_results_total", "WITNESSED (fak authored): oversized old tool_result bodies shrunk in the uncached post-breakpoint region. The transform keeps the protected cache prefix byte-identical and leaves recent/cache_control-bearing results intact.", int64(snap.uncachedTrimResults))
	writeCounter(b, "fak_gateway_uncached_trim_shed_tokens_total", "WITNESSED (fak authored): estimated tokens removed by uncached-tail oversized-result trim, also folded into fak_gateway_compaction_shed_tokens_total for the owner=\"fak\" cache-savings attribution.", int64(snap.uncachedTrimShed))

	// The managed-cache 1h TTL upgrade family (--managed-cache, epic #1844 C6). WITNESSED:
	// fak spliced ttl:"1h" into an existing stable-head cache_control (or bailed for the
	// named reason). Whether the provider then HONORS the longer TTL across an idle gap is
	// OBSERVED on the cache_read counters, never claimed here. The "upgraded" row is emitted
	// even at 0 so an active lever with zero eligible heads is visible, not silent.
	writeHelpType(b, "fak_gateway_cache_ttl_upgrade_total",
		"WITNESSED (fak authored): managed-cache 1h TTL upgrade attempts on the outbound Anthropic wire, by outcome: upgraded (ttl:\"1h\" spliced into an existing stable system/tools-head cache_control), placed_and_upgraded (no cache_control existed; fak placed the stable-head breakpoint AND upgraded it in one turn, #2175), or the bail reason (no_stable_breakpoint|already_1h|ttl_already_set|volatile_head|non_json|splice_failed|redecode_failed). Rows exist only while --managed-cache is ACTIVE. splice_failed/redecode_failed are fak bugs and must stay 0. The provider honoring the 1h tier across an idle gap is OBSERVED via cache_read, not claimed by this counter.", "counter")
	fmt.Fprintf(b, "fak_gateway_cache_ttl_upgrade_total{outcome=%q} %d\n", "upgraded", snap.ttlUpgrades["upgraded"])
	ttlReasons := make([]string, 0, len(snap.ttlUpgrades))
	for r := range snap.ttlUpgrades {
		if r != "upgraded" {
			ttlReasons = append(ttlReasons, r)
		}
	}
	sort.Strings(ttlReasons)
	for _, r := range ttlReasons {
		fmt.Fprintf(b, "fak_gateway_cache_ttl_upgrade_total{outcome=%q} %d\n", promQuote(r), snap.ttlUpgrades[r])
	}

	// WITNESSED (fak authored): the OFFENSIVE cache-breakpoint placement family (#806). "placed"
	// counts turns where fak spliced a cache_control breakpoint onto the stable head of a caller that
	// sent none — the fak-UNLOCKED slice, since a no-breakpoint caller earns 0 provider cache without
	// it. "already_set" counts the Claude-Code shape fak leaves to the client's own cache (NOT fak's).
	// The "placed" row is emitted even at 0 so a passthrough that never placed is visible, not silent.
	// splice_failed/redecode_failed are fak bugs and must stay 0.
	writeHelpType(b, "fak_gateway_cache_breakpoint_placement_total",
		"WITNESSED (fak authored): offensive cache-breakpoint placements on the outbound Anthropic wire, by outcome: placed (a cache_control breakpoint spliced onto the stable system/tools head of a caller that sent none) or the bail reason (already_set|no_stable_head|volatile_head|non_json|splice_failed|redecode_failed). placed is the fak-unlocked slice — the provider cache_read those turns earn would be 0 without it; already_set is the client's own cache, not fak's. The provider serving turns 2..N from that cache is OBSERVED via cache_read, not claimed by this counter.", "counter")
	fmt.Fprintf(b, "fak_gateway_cache_breakpoint_placement_total{outcome=%q} %d\n", "placed", snap.placementAttempts["placed"])
	placementReasons := make([]string, 0, len(snap.placementAttempts))
	for r := range snap.placementAttempts {
		if r != "placed" {
			placementReasons = append(placementReasons, r)
		}
	}
	sort.Strings(placementReasons)
	for _, r := range placementReasons {
		fmt.Fprintf(b, "fak_gateway_cache_breakpoint_placement_total{outcome=%q} %d\n", promQuote(r), snap.placementAttempts[r])
	}

	// The INBOUND tool-floor prune family (the tools[] twin of the compaction shed above).
	// WITNESSED: how many unreachable tool DEFINITIONS fak dropped from the advertised surface,
	// a pure uncached-token saving the pruner makes only after the cache_control breakpoint so it
	// never bursts the cache. Both stay 0 on the dominant Claude Code path (its single breakpoint
	// sits on the LAST tool, so nothing is droppable) — which, before these rows existed, was the
	// invisible fact: the prune result was discarded with no counter.
	pruneTurns, pruneCount := m.inboundToolPruneSnapshot()
	writeCounter(b, "fak_gateway_inbound_tools_pruned_total", "WITNESSED (fak authored): cumulative unreachable tool DEFINITIONS dropped from the outbound tools[] across the session. A pure uncached-token saving — the pruner drops only tools after the cache_control breakpoint and re-proves the protected prefix is byte-identical, so a counted prune never bursts the provider-side upstream cache.", int64(pruneCount))
	writeCounter(b, "fak_gateway_inbound_tools_prune_turns_total", "WITNESSED (fak authored): turns on which at least one unreachable tool def was pruned from tools[]. Zero on a harness (e.g. Claude Code) whose single cache_control breakpoint sits on the LAST tool, since nothing is then droppable.", int64(pruneTurns))
	writeCounter(b, "fak_gateway_inbound_tools_pruned_then_proposed_total", "WITNESSED (fak authored): pruned tool definition names that the model later proposed anyway, counted once per trace/tool. Nonzero means the advertised floor and observed model behavior drifted; call-time adjudication still default-denies the proposal.", int64(m.inboundPrunedToolProposalSnapshot()))

	// The OUTBOUND cold-tool DEFERRAL family (the 10x floor lever, --defer-cold-tools #3232 under
	// epic #3229) — the OUTBOUND twin of the inbound prune family above, and the render surface the
	// #3232 follow-up called for. WITNESSED: how many cold tool DEFINITIONS fak marked
	// defer_loading and handed to the provider's tool-search fault-in. Unlike the prune, this does
	// NOT shrink request bytes — the reduction is provider-side (only the hot core loads into
	// context), so these counters witness the deferral fak DROVE; the actual token drop is OBSERVED
	// via input_tokens/cache_read, not claimed here. Both stay 0 when the lever is off (its DEFAULT)
	// or when no turn had a cold tool tail to defer.
	deferTurns, deferCold := m.toolDeferSnapshot()
	writeCounter(b, "fak_gateway_tool_defer_cold_total", "WITNESSED (fak authored): cumulative cold tool DEFINITIONS marked defer_loading:true on the outbound Anthropic body across the session (the 10x floor lever, --defer-cold-tools #3232). Cache-safe by construction (deterministic, byte-stable tools[] turn-over-turn), so a counted deferral never bursts the upstream prompt cache. The provider-side context/token drop it drives is OBSERVED via input_tokens/cache_read, not claimed by this counter.", int64(deferCold))
	writeCounter(b, "fak_gateway_tool_defer_turns_total", "WITNESSED (fak authored): turns on which fak deferred the cold tool tail (marked >=1 cold def defer_loading and injected a tool_search_tool). Zero when --defer-cold-tools is off (its default) or when every advertised tool was hot; nonzero means the lever fired that turn.", int64(deferTurns))
	// The DENOMINATOR of the two counters above (#3621). Both of them are pure numerators, so a
	// flat zero reads identically whether the lever was never armed or was armed and bit on
	// nothing — the silent-identity failure mode. This counter accrues only PAST the eligibility
	// gate (lever on, Anthropic passthrough wire, ablation arm off), so `_cold_total == 0 AND
	// _standdown_turns_total >= 3` is the scrape-side form of the DEFER_ENABLED_BUT_INERT finding
	// the guard banner and /debug/vars cache_attribution.fak_defer_finding raise.
	standDownTurns, _ := m.toolDeferStandDownSnapshot()
	writeCounter(b, "fak_gateway_tool_defer_standdown_turns_total", "WITNESSED (fak authored): turns on which the cold-tool-defer transform RAN — lever on, Anthropic passthrough wire, ablation arm off — and stood down to byte-identity anyway (no cold tools, a client body already deferred, or a splice fak could not prove). The denominator for fak_gateway_tool_defer_cold_total: a session with cold_total==0 and this counter climbing is the lever armed-but-inert (#3621), NOT a lever that was left off.", int64(standDownTurns))

	// The tool_reference SANITIZE family (a CORRECTNESS transform, not a cache saving). WITNESSED:
	// how many Claude-Code-internal tool_reference blocks fak rewrote into wire-valid text blocks so
	// the body was not 400'd upstream as malformed. Nonzero means fak repaired a body the provider
	// would otherwise have rejected outright (witnessed defect: session b98cf818, killed by a
	// ToolSearch tool_result carrying tool_reference blocks).
	refTurns, refConverted := m.toolRefSanitizeSnapshot()
	writeCounter(b, "fak_gateway_tool_reference_converted_total", "WITNESSED (fak authored): cumulative Claude-Code-internal tool_reference content blocks rewritten into wire-valid text blocks across the session. tool_reference is not a valid Anthropic tool_result.content type; converting it in place keeps a ToolSearch-bearing body from being 400'd by the API as malformed.", int64(refConverted))
	writeCounter(b, "fak_gateway_tool_reference_sanitize_turns_total", "WITNESSED (fak authored): turns on which at least one tool_reference block was converted. Zero on a harness that never replays tool-discovery results into a tool_result; nonzero means fak repaired a body the provider would otherwise have 400'd as malformed.", int64(refTurns))

	// The general-form EMPTY-CONTENT GATE family (#3118): the residual backstop to the
	// tool_reference sanitizer above. WITNESSED: how many tool_result.content arrays fak found
	// EMPTY (for ANY reason) and backfilled with a placeholder text block, since an empty content
	// array is itself a 400. Nonzero means fak repaired a body a future client-internal block type
	// (or a genuinely empty source result) would otherwise have gotten rejected upstream.
	emptyTurns, emptyRepaired := m.emptyContentRepairSnapshot()
	writeCounter(b, "fak_gateway_empty_tool_result_repaired_total", "WITNESSED (fak authored): cumulative empty tool_result.content arrays backfilled with a placeholder text block across the session. The general form of the tool_reference sanitizer — it catches any content array that ended up empty for ANY reason, since an empty array itself 400s as malformed.", int64(emptyRepaired))
	writeCounter(b, "fak_gateway_empty_tool_result_repair_turns_total", "WITNESSED (fak authored): turns on which at least one empty tool_result.content array was repaired. Zero on the common turn; nonzero means fak repaired a body the provider would otherwise have 400'd as empty content.", int64(emptyTurns))
}

// writeResetShadowMetrics renders the per-session resetScore SHADOW family (#792). The reset
// policy recommends cut-vs-reset on the compaction crossover; in SHADOW mode it acts on NOTHING,
// so this surface is purely "what it WOULD recommend" — recommendations_total is the count of
// turns the verdict said reset, and it stays a recommendation until shadow evidence supports
// enabling reset. The verdict is WITNESSED (fak's policy); the cache ratios it scored are OBSERVED
// (provider-reported), the same split the compaction family keeps. Every reason bucket is emitted
// at 0 so the panel exists before the first compacted turn.
func (m *gatewayMetrics) writeResetShadowMetrics(b *strings.Builder) {
	snap := m.resetShadowSnapshotData()
	writeHelpType(b, "fak_gateway_compaction_reset_shadow_total",
		"WITNESSED (fak policy, recommend-only): compacted turns scored by the resetScore SHADOW policy, by reason (healthy_cache|stale_prefix|cache_decay|cooldown|unknown_provider). SHADOW mode acts on nothing — this is what the cut-vs-reset crossover WOULD recommend, not a reset that happened. The cache ratios scored are OBSERVED (provider-reported).", "counter")
	for _, reason := range []string{
		string(ResetReasonHealthy), string(ResetReasonStalePrefix), string(ResetReasonDecay),
		string(ResetReasonCooldown), string(ResetReasonUnknown),
	} { // stable order; emit at 0 so the panel exists pre-first-turn
		fmt.Fprintf(b, "fak_gateway_compaction_reset_shadow_total{reason=%q} %d\n", reason, snap.reasons[reason])
	}
	writeCounter(b, "fak_gateway_compaction_reset_recommendations_total",
		"WITNESSED (fak policy, recommend-only): compacted turns whose resetScore SHADOW verdict was ShouldReset. In SHADOW mode NONE were acted on — the count is the reset pressure the policy held back, cut-by-default.", int64(snap.recommend))
	writeHelpType(b, "fak_gateway_compaction_reset_score",
		"The most recent compacted turn's 0..1 resetScore reset-pressure magnitude (0 = clearly keep cutting, 1 = clearly reset). Reported even when the cooldown holds the recommendation, so the building pressure is visible. Advisory: nothing acts on it in SHADOW mode.", "gauge")
	fmt.Fprintf(b, "fak_gateway_compaction_reset_score %s\n", promFloat(snap.lastScore))
}

// writeCacheBreakMetrics renders the per-session cache-break family (#2916): two counter
// families — event count and cold-rebuild token cost — each labeled by the closed cause
// vocabulary (toolset_change/altered_turn/rebuilt_prompt/provider_quirk/unknown) and emitted
// in canonical cause order. Both families ALWAYS declare their HELP/TYPE so a regression gate
// and a dashboard panel exist from the first scrape; a session with no witnessed break renders
// the declarations with NO per-cause sample (a clean zero), which is exactly the empty state a
// gate reads as "no regression". The scrape and the guard exit summary fold the SAME
// cacheBreakReport witnesses, so the two views can never disagree.
func (m *gatewayMetrics) writeCacheBreakMetrics(b *strings.Builder) {
	report := m.cacheBreakReport()
	writeHelpType(b, metrics.CacheBreakEventsMetric,
		"Cache-break events this session, labeled by closed cause (toolset_change/altered_turn/rebuilt_prompt/provider_quirk/unknown). A rise means the warm prompt prefix broke more often mid-conversation.", "counter")
	for _, t := range report.ByCause {
		fmt.Fprintf(b, "%s{cause=\"%s\"} %d\n", metrics.CacheBreakEventsMetric, promQuote(string(t.Cause)), t.Events)
	}
	writeHelpType(b, metrics.CacheBreakCostMetric,
		"Cold-rebuild token cost of cache breaks this session, labeled by closed cause. Each break's cost is the warm prompt prefix that had to be re-prefilled.", "counter")
	for _, t := range report.ByCause {
		fmt.Fprintf(b, "%s{cause=\"%s\"} %d\n", metrics.CacheBreakCostMetric, promQuote(string(t.Cause)), t.CostTokens)
	}
}
