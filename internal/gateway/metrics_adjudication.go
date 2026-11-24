package gateway

import (
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// AdjudicationSummary is a verdict roll-up over every kernel decision a gateway has
// made — the tally `fak guard` prints when the wrapped agent exits, so an operator
// sees what the kernel allowed vs blocked without scraping /metrics. It folds the
// per-(operation, verdict, reason) operation counters into one record; every count is
// the SAME number the fak_gateway_operations_total scrape would report, so the exit
// line can never disagree with the metrics.
type AdjudicationSummary struct {
	Total       uint64 `json:"total"`
	Allowed     uint64 `json:"allowed"`
	Denied      uint64 `json:"denied"`
	Transformed uint64 `json:"transformed"`
	Quarantined uint64 `json:"quarantined"`
	// Deferred counts DEFER verdicts: a non-blocking admit (e.g. an inbound tool
	// result the kernel let through while raising the session's taint watermark).
	// It is NOT an error — the old default-bucket fold reported it under Errored,
	// which made a perfectly healthy proxy_admit read as a failure in the exit
	// summary `fak guard` prints (a tool-bearing turn always admits its result).
	Deferred uint64 `json:"deferred"`
	// Escalated counts REQUIRE_WITNESS verdicts: a call HELD pending a witness /
	// human approval rather than allowed or denied outright. Also not an error.
	Escalated uint64 `json:"escalated"`
	// Errored counts genuine ERROR verdicts (and any unknown future kind) — a real
	// adjudication failure, never silently dropped.
	Errored uint64 `json:"errored"`
	// ByReason maps a deny/quarantine reason code to its count (the forensic "why").
	ByReason map[string]uint64 `json:"by_reason,omitempty"`
	// CachedPromptTokens is the cumulative count of prompt (input) tokens the upstream
	// PROVIDER served from its OWN prompt cache (cache_read) across this session's
	// turns — the provider-side reuse `fak guard` preserves byte-for-byte by forwarding
	// the client's cache_control prefix unchanged through the kernel hop. CachedTurns
	// counts the turns that got such a hit. Surfaced in the guard exit summary so the
	// operator SEES the cache reuse rather than having to scrape /metrics.
	CachedPromptTokens uint64 `json:"cached_prompt_tokens"`
	CachedTurns        uint64 `json:"cached_turns"`
	// DeadlineCacheCredit retains the historical cache-credit metrics; served
	// routes now use owned residency or stay cold, never historical credit.
	// Previously this counted chat admissions priced with a reported cached
	// prefix and the tokens credited (deadline_cache_credit.go).
	DeadlineCacheCredit *DeadlineCacheCreditStats `json:"deadline_cache_credit,omitempty"`
	// InputTokens (the uncached input remainder), OutputTokens, and CacheCreationTokens
	// (the cache WRITE axis) are retained alongside CachedPromptTokens (the READ axis) so
	// the summary can price the NET realized provider-cache saving — read rebate MINUS
	// write premium — via ProviderCacheNetSavings and the Track-2 savings ledger. All are
	// OBSERVED (provider-relayed).
	InputTokens         uint64 `json:"input_tokens"`
	OutputTokens        uint64 `json:"output_tokens"`
	CacheCreationTokens uint64 `json:"cache_creation_tokens"`
	// SelfHosted*/Vendor* split InputTokens and OutputTokens by WHO SERVED the turn
	// (epic #5416): tokens generated in-kernel on hardware we host, versus tokens
	// bought from a third-party API. Both are strict subsets of the unsplit totals —
	// a turn whose side servedLocality could not resolve is counted in the totals and
	// in neither group — so the shortfall between them is the unclassified volume,
	// and a reader can tell a measured self-hosted zero (Vendor turns present, no
	// local ones) from an unmeasured one (no classified turn at all). All omitempty:
	// a summary from a build or deployment that classified nothing must not serialize
	// six zeros that read as "we self-host nothing".
	SelfHostedTurns        uint64 `json:"self_hosted_turns,omitempty"`
	SelfHostedInputTokens  uint64 `json:"self_hosted_input_tokens,omitempty"`
	SelfHostedOutputTokens uint64 `json:"self_hosted_output_tokens,omitempty"`
	VendorTurns            uint64 `json:"vendor_turns,omitempty"`
	VendorInputTokens      uint64 `json:"vendor_input_tokens,omitempty"`
	VendorOutputTokens     uint64 `json:"vendor_output_tokens,omitempty"`
	// CacheCreationTokensUpgraded is the subset of CacheCreationTokens attributed to the
	// managed-cache 1h TTL tier (GATEWAY_ATTRIBUTED — fak's own per-turn upgrade witness,
	// since the provider wire never splits 5m vs 1h creation tokens; see
	// inferCacheCreationTokensUpgraded). 0 means either the lever was off or every write
	// stayed on the 5m tier; MechanismSavings/ProviderCacheNetSavings price the remainder
	// (CacheCreationTokens - CacheCreationTokensUpgraded) at the 5m tier as before (#2179).
	CacheCreationTokensUpgraded      uint64 `json:"cache_creation_tokens_upgraded,omitempty"`
	CacheCreationTokensTierObserved  uint64 `json:"cache_creation_tokens_tier_observed,omitempty"`
	CacheCreationTokensHeadOnly      uint64 `json:"cache_creation_tokens_head_only,omitempty"`
	CacheCreationTokensMessagePrefix uint64 `json:"cache_creation_tokens_message_prefix,omitempty"`

	// Compaction* folds the Anthropic history-compaction visibility into the same guard exit
	// summary, split WITNESSED (what fak authored) vs OBSERVED (what the provider reported):
	// Fired/Bailed/Off and DroppedTurns/ShedTokens are WITNESSED — what fak attempted and what
	// it removed (a turn only counts Fired when the prefix it shipped was byte-identical).
	// CompactionCacheReadTokens / LastCompactionCacheRead are OBSERVED — the provider's
	// cache_read_input_tokens, relayed verbatim. They are NOT proof fak preserved the cache (the
	// byte-identity is); a low value with no prefix_mismatch is a provider-side miss fak does not
	// control (TTL/eviction/client breakpoint move).
	CompactionFired           uint64  `json:"compaction_fired"`
	CompactionBailed          uint64  `json:"compaction_bailed"`
	CompactionOff             uint64  `json:"compaction_off"`
	CompactionDroppedTurns    uint64  `json:"compaction_dropped_turns"`
	CompactionRestoredTurns   uint64  `json:"compaction_restored_turns"`
	CompactionShedTokens      uint64  `json:"compaction_shed_tokens"`
	CompactionCacheReadTokens uint64  `json:"compaction_cache_read_tokens"`
	LastCompactionCacheRead   float64 `json:"last_compaction_cache_read"`
	// CompactionBailReasons is the WITNESSED per-reason breakdown of the CompactionBailed
	// lump (CompactReason* -> count). Without it, "N bailed" is uninterpretable: a session
	// that bailed N times to under_budget (the compactible suffix already fit — benign,
	// working-as-designed) is indistinguishable from one that bailed to prefix_mismatch (the
	// only fak-fault cache signal, must stay 0) or no_breakpoint (no anchor — can't fire).
	// Already surfaced on /metrics + /debug/vars; folded here so the guard exit summary can
	// render it the same way ByReason renders deny reasons.
	CompactionBailReasons map[string]uint64 `json:"compaction_bail_reasons,omitempty"`
	// CompactionAnchorStarved is the count of under_budget bails whose protected prefix ALREADY
	// exceeded the budget — the cache_control anchor swallowed the conversation, so compaction
	// structurally cannot fire no matter how long the session grows. It is a SUBSET of
	// CompactionBailReasons["under_budget"], broken out because the two are operationally opposite:
	// a plain under_budget is a benign short session (nothing to shed), an anchor-starved one is the
	// dormant-on-real-Claude-Code-traffic pathology (#1407) that no budget tightening can fix.
	CompactionAnchorStarved uint64 `json:"compaction_anchor_starved"`
	// CompactionSolvencyForced is the count of fires the head-anchored burst economics REFUSED
	// and the context-solvency floor forced anyway (Config.CompactSolvencyFloorTokens). It is a
	// SUBSET of CompactionFired, broken out because the two are economically opposite: an
	// ordinary fire is a burst that repays in cache dollars, a forced one is a burst knowingly
	// taken at a loss to keep the session inside its context window. A nonzero count is not a
	// fault — it is the override doing its job — but the cache-value ledger must not book these
	// as savings, and a count that dominates CompactionFired means the window, not the cache, is
	// the binding constraint on this workload.
	CompactionSolvencyForced uint64 `json:"compaction_solvency_forced"`
	// CompactionBudget is the resident-token threshold the history rewrite fires past
	// (Config.CompactHistoryBudget; 0 means the lever is OFF, body forwarded byte-for-byte).
	// Surfaced so the exit line can say whether compaction is ENABLED and merely idle
	// (budget>0, nothing exceeded it) vs DISABLED — the two readings of "0 fired" the bare
	// counters cannot tell apart.
	CompactionBudget int `json:"compaction_budget"`

	// KVPrefix* folds the local in-kernel RadixAttention prefix-cache reuse tap into
	// the same attribution frame as provider prompt-cache and compaction. These are
	// WITNESSED fak-authored savings: prompt prefill work the in-kernel planner did
	// not redo. They are distinct from CachedPromptTokens, which are OBSERVED
	// provider cache_read tokens.
	KVPrefixPromptTokens uint64 `json:"kv_prefix_prompt_tokens"`
	KVPrefixReusedTokens uint64 `json:"kv_prefix_reused_tokens"`
	// KVPrefixColdCliff is the LIVE frozen-trajectory cache-cliff finding (#3623): present
	// (PREFIX_COLD_CLIFF) only when the in-kernel KV-prefix reuse for this process has
	// collapsed — a cold-dominated turn distribution or an aggregate reuse ratio below the
	// floor. Absent (omitempty) on a healthy frozen/partial session, so its mere PRESENCE on
	// /debug/vars is the alarm. WITNESSED (fak's own kernel reuse), like the KVPrefix* counts.
	KVPrefixColdCliff *cacheobs.ColdCliffVerdict `json:"kv_prefix_cold_cliff,omitempty"`

	// ToolPrune* folds the INBOUND tool-definition prune lever into the same exit summary,
	// WITNESSED (fak authored): ToolPruneTurns is the count of turns that dropped at least one
	// unreachable tool def from the outbound tools[], and ToolPruneCount the total defs removed.
	// The pruner only drops tools strictly AFTER the cache_control breakpoint and re-proves the
	// protected prefix is byte-identical, so a counted prune is a pure uncached-token saving that
	// never bursts the upstream cache. Zero on the dominant Claude Code path (its single breakpoint
	// sits on the LAST tool, so nothing is droppable) — which is exactly the fact the operator
	// could not see before, since the prune result was discarded with no metric.
	ToolPruneTurns uint64 `json:"tool_prune_turns"`
	ToolPruneCount uint64 `json:"tool_prune_count"`

	// DeferCold* folds the OUTBOUND cold-tool DEFERRAL lever (--defer-cold-tools, #3232, the 10x
	// floor lever under epic #3229) into the same exit summary, WITNESSED (fak authored the splice):
	// DeferColdTurns is the count of turns on which fak deferred the cold tool tail (marked the
	// allowed-but-cold custom tools defer_loading:true and injected a tool_search_tool), and
	// DeferColdCount the total cold defs deferred across them. Unlike the prune lever this does NOT
	// shrink the request bytes — the reduction is provider-side (the provider loads only the hot
	// core into context, faulting a cold schema in on demand), so it lands in the OBSERVED usage
	// relay (input_tokens / cache_read), never in the ESTIMATED byte footprint. The transform is
	// deterministic + cache-safe, so a counted deferral never bursts the upstream prompt cache.
	// Zero when the lever is off (its DEFAULT) or when a turn had no cold tools to defer.
	DeferColdTurns uint64 `json:"defer_cold_turns"`
	DeferColdCount uint64 `json:"defer_cold_count"`
	// DeferColdToolNames is the DISTINCT set of custom tool names fak deferred across those
	// turns (#3647) — surfaced so the operator pane can name WHICH tools were made cold, not
	// just how many. Sorted for a stable render; empty/absent when the lever never fired.
	DeferColdToolNames []string `json:"defer_cold_tool_names,omitempty"`
	// DeferStandDownTurns / DeferStandDownReasons are the missing DENOMINATOR of the DeferCold*
	// numerator above (#3621): turns on which the lever was armed and the transform ran, yet it
	// stood down to identity, plus the per-reason breakdown (no_cold_tools, already_deferred,
	// decode_failed, …). They are what makes "the lever is off" distinguishable from "the lever
	// is on and never bit" — the silent-identity failure mode the DEFER_ENABLED_BUT_INERT
	// watchdog (defer_inert.go) keys on. Absent (omitempty) when nothing ever stood down.
	DeferStandDownTurns   uint64            `json:"defer_stand_down_turns,omitempty"`
	DeferStandDownReasons map[string]uint64 `json:"defer_stand_down_reasons,omitempty"`

	// ToolRefTurns / ToolRefConverted witness the tool_reference SANITIZE correctness transform:
	// the client's INTERNAL tool_reference blocks (emitted inside a ToolSearch tool_result) are not
	// a valid Anthropic tool_result.content type, so fak rewrites each into a wire-valid text block
	// before relay to keep the body from being 400'd upstream. ToolRefTurns counts turns with >=1
	// conversion; ToolRefConverted the total blocks converted. Zero on a harness that never surfaces
	// tool-discovery results back into a tool_result — nonzero means fak repaired a body the provider
	// would otherwise have rejected as malformed.
	ToolRefTurns     uint64 `json:"tool_ref_turns"`
	ToolRefConverted uint64 `json:"tool_ref_converted"`

	// DenyAllStops is how many served turns this session had EVERY proposed tool call
	// refused by the floor, which the wire reports as end_turn — a STOP the agent did not
	// choose (it wanted to act, the floor blocked all of it). Surfaced in the guard exit
	// summary so the operator SEES how often fak ended a turn, the otherwise-invisible
	// false-stop the --deny-all-continue Stop-hook auto-resumes the agent past.
	DenyAllStops uint64 `json:"deny_all_stops"`
	// ToolFeedbackTurns is the sibling non-terminal count: every proposed tool call was
	// rejected as retryable/model-fixable feedback, so the guard may continue the turn,
	// but this is NOT a session-stop/give-up policy input.
	ToolFeedbackTurns uint64 `json:"tool_feedback_turns"`

	// CacheTTLUpgrade* folds the managed-cache 1h TTL-upgrade lever (--managed-cache,
	// epic #1844 C6) into the same exit frame, WITNESSED (fak authored the cache_control
	// splice, and the upgrader re-proves the body redecodes before returning changed
	// bytes): CacheTTLUpgraded counts turns whose stable-head breakpoint fak actually
	// extended to the 1h tier — BOTH authoring outcomes, an existing breakpoint upgraded
	// ("upgraded") and a breakpoint placed-and-upgraded in one composed turn (#2175);
	// CacheTTLUpgradeReasons is the per-refusal breakdown over the closed
	// agent.TTLUpgradeReason* vocabulary (an authoring outcome is never a refusal, so it
	// never appears here). The family is recorded only while
	// the lever is on, so zero/absent means OFF, while a zero upgraded count WITH reason
	// rows means ON-but-ineligible (every head refused) — visible, not silent.
	CacheTTLUpgraded       uint64            `json:"cache_ttl_upgrades_upgraded"`
	CacheTTLUpgradeReasons map[string]uint64 `json:"cache_ttl_upgrade_reasons,omitempty"`

	// E2ELatency* fold the session's OBSERVED end-to-end per-turn latency distribution
	// (inferE2EHist, fed by observeInferenceTimed for every served turn) into the summary
	// so the guard exit line can price WITNESSED turns-saved into wall-clock at the
	// session's OWN measured per-turn cost — never a fabricated tokens/sec constant.
	// Sum is the total observed seconds across Count timed turns; mean = Sum/Count (see
	// MeanTurnLatencySeconds). Both are OBSERVED (provider round-trip timing), and 0 when
	// no turn was timed (a replayed/offline summary, or a session that served nothing) —
	// in which case the exit line prints turns only and omits the wall-clock dual.
	E2ELatencySumSeconds float64 `json:"e2e_latency_sum_seconds,omitempty"`
	E2ELatencyCount      uint64  `json:"e2e_latency_count,omitempty"`

	// AnchorPlacement is the session's breakpoint-placement outcome MIX and the rolling
	// volatile-head verdict over it (#3622) — the live-loop half of the placement counters the
	// Prometheus surface already renders. The counters are cumulative and unwatched: a session
	// whose head turns volatile mid-conversation simply stops incrementing "placed", which reads
	// identically to a session that went quiet. This carries the fraction, the dominant refusal
	// outcome, and how many times ANCHOR_REFUSED_RISING was raised, so the guard exit banner can
	// say the anchor stopped earning caching instead of leaving the operator to diff two counters.
	// A pointer with omitempty: nil — and the JSON field absent — on any session that never
	// attempted a placement, so a non-Anthropic or cold session's summary is byte-identical to
	// before this field existed.
	AnchorPlacement *metrics.AnchorRefusalReport `json:"anchor_placement,omitempty"`
}

// adjudicationSummary folds the live operation counters into a verdict roll-up.
func (m *gatewayMetrics) adjudicationSummary() AdjudicationSummary {
	sum := AdjudicationSummary{ByReason: map[string]uint64{}}
	if m == nil {
		return sum
	}
	// Provider prompt-cache reuse rides the inference counters (a separate lock from the
	// operation ledger); read it first so the two critical sections never nest.
	if cc := m.warmPrefixLedger().stats(); cc != (DeadlineCacheCreditStats{}) {
		sum.DeadlineCacheCredit = &cc
	}
	m.inferenceMu.Lock()
	sum.CachedPromptTokens = m.inferCachedTokens
	sum.CachedTurns = m.inferCachedHits
	sum.InputTokens = m.inferPromptTokens
	sum.OutputTokens = m.inferComplTokens
	sum.SelfHostedTurns = m.inferSelfHostedTurns
	sum.SelfHostedInputTokens = m.inferSelfHostedPromptTokens
	sum.SelfHostedOutputTokens = m.inferSelfHostedComplTokens
	sum.VendorTurns = m.inferVendorTurns
	sum.VendorInputTokens = m.inferVendorPromptTokens
	sum.VendorOutputTokens = m.inferVendorComplTokens
	sum.CacheCreationTokens = m.inferCacheCreationTokens
	sum.CacheCreationTokensUpgraded = m.inferCacheCreationTokensUpgraded
	sum.CacheCreationTokensTierObserved = m.inferCacheCreationTokensTierObserved
	sum.CacheCreationTokensHeadOnly = m.inferCacheCreationTokensHeadOnly
	sum.CacheCreationTokensMessagePrefix = m.inferCacheCreationTokensMessagePrefix
	// Observed per-turn E2E latency distribution (same lock observeInferenceTimed holds
	// when it writes inferE2EHist), so the exit line can price turns-saved into wall-clock
	// at the session's own measured per-turn cost.
	if m.inferE2EHist != nil {
		sum.E2ELatencySumSeconds = m.inferE2EHist.sum
		sum.E2ELatencyCount = m.inferE2EHist.count
	}
	m.inferenceMu.Unlock()

	comp := m.compactionSnapshotData()
	sum.CompactionFired = comp.attempts["fired"]
	sum.CompactionBailed = comp.attempts["bailed"]
	sum.CompactionOff = comp.attempts["off"]
	sum.CompactionDroppedTurns = comp.dropped
	sum.CompactionRestoredTurns = comp.restored
	sum.CompactionShedTokens = comp.shed
	sum.CompactionCacheReadTokens = comp.cacheReads
	sum.LastCompactionCacheRead = comp.lastCacheRd
	sum.CompactionAnchorStarved = comp.anchorStarved
	sum.CompactionSolvencyForced = comp.solvencyForced
	// Carry the per-reason bail breakdown so the banner can explain the bailed lump (the
	// snapshot already copied it under compactMu); only attach a non-empty map so a clean
	// session keeps the JSON field absent (omitempty).
	sum.ToolPruneTurns, sum.ToolPruneCount = m.inboundToolPruneSnapshot()
	sum.DeferColdTurns, sum.DeferColdCount = m.toolDeferSnapshot()
	sum.DeferColdToolNames = m.toolDeferNamesSnapshot()
	sum.DeferStandDownTurns, sum.DeferStandDownReasons = m.toolDeferStandDownSnapshot()
	sum.AnchorPlacement = m.anchorRefusalReport()
	sum.ToolRefTurns, sum.ToolRefConverted = m.toolRefSanitizeSnapshot()
	sum.DenyAllStops, _ = m.denyAllSnapshot()
	sum.ToolFeedbackTurns, _ = m.toolFeedbackSnapshot()
	// COMPACTION_THRASH (#2424) rides this same map. It is NOT part of the CompactionBailed
	// lump — the lever FIRED on every thrashing turn — but an operator reading "why is
	// compaction not holding this session" needs it beside the bail reasons, and a verdict
	// published nowhere is the gap the issue reports. Kept out of comp.bailReasons (and so
	// out of bail_reason_total's closed set and compactBailPartition's alertable rate) by
	// folding into a fresh map here; its own /metrics counter is
	// fak_gateway_compaction_thrash_sessions_total.
	if len(comp.bailReasons) > 0 || comp.thrashSessions > 0 {
		reasons := make(map[string]uint64, len(comp.bailReasons)+1)
		for k, v := range comp.bailReasons {
			reasons[k] = v
		}
		if comp.thrashSessions > 0 {
			reasons[ReasonCompactionThrash] = comp.thrashSessions
		}
		sum.CompactionBailReasons = reasons
	}
	// Managed-cache TTL-upgrade outcomes (#1844 C6): split the snapshot's one outcome
	// map into the AUTHORED count and the refusal-reason breakdown, attaching the map only
	// when non-empty so a lever-off session keeps the JSON field absent (omitempty). BOTH
	// authoring outcomes count as authored — "upgraded" (an existing stable-head breakpoint
	// extended to 1h) AND "placed_and_upgraded" (the #2175 composed transform that placed a
	// breakpoint AND upgraded it in one turn) — matching the managed-cache self-check's
	// Authored = Upgraded + PlacedAndUpgraded contract (internal/cachewitness). A composed
	// fire is NOT a refusal, so it must never land in the reason map: booking it there
	// undercounted CacheTTLUpgraded AND double-counted it as a bail, sinking the fired-rate
	// the cache-health digest folds (cachevaluereport TTLRefusals → UpgradeFiredHealth).
	sum.CacheTTLUpgraded = comp.ttlUpgrades["upgraded"] + comp.ttlUpgrades[cacheTTLUpgradePlacedAndUpgraded]
	ttlReasons := map[string]uint64{}
	for reason, n := range comp.ttlUpgrades {
		if reason != "upgraded" && reason != cacheTTLUpgradePlacedAndUpgraded {
			ttlReasons[reason] = n
		}
	}
	if len(ttlReasons) > 0 {
		sum.CacheTTLUpgradeReasons = ttlReasons
	}
	kv := cacheobs.Default.Snapshot()
	sum.KVPrefixPromptTokens = kv.PromptTokens
	sum.KVPrefixReusedTokens = kv.ReusedTokens
	// Raise the live frozen-trajectory cache-cliff finding (#3623) only when this
	// process's realized KV-prefix reuse has collapsed; a healthy session leaves the
	// field nil, so its presence on the exit summary / /debug/vars is itself the alarm.
	if cliff := kv.ColdCliff(); cliff.Fired {
		sum.KVPrefixColdCliff = &cliff
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	for key, c := range m.operations {
		n := c.count
		if n == 0 {
			continue
		}
		sum.Total += n
		switch key.verdict {
		case "ALLOW":
			sum.Allowed += n
		case "TRANSFORM":
			sum.Transformed += n
		case "DENY":
			sum.Denied += n
			if key.reason != "" {
				sum.ByReason[key.reason] += n
			}
		case "QUARANTINE":
			sum.Quarantined += n
			if key.reason != "" {
				sum.ByReason[key.reason] += n
			}
		case "DEFER":
			// A non-blocking admit (the inbound result was let through, the call
			// was not refused). Distinct from ALLOW only in that the kernel held a
			// firm opinion in reserve; reporting it as "errored" alarmed the
			// operator over a perfectly healthy decision.
			sum.Deferred += n
		case "REQUIRE_WITNESS":
			// Held pending a witness / approval — an escalation, not an error.
			sum.Escalated += n
		default: // a genuine "ERROR", or an unknown future kind: counted, never silently dropped.
			sum.Errored += n
		}
	}
	return sum
}

// providerCacheEvidence classifies the gateway's recorded provider prompt-cache
// reuse through the cachemeta materialization bridge (issue #432, acceptance #3). A
// provider cache_read is a `provider_prefix` materialization: COST/LATENCY telemetry
// about a prefix the REMOTE engine kept resident, never a re-serveable LOCAL-trust
// artifact. Routing the live telemetry through the same proven gate the kernel uses
// (MaterializeVerdict(MatProviderPrefix, …)) makes the separation mechanical on the
// live path — the verdict is structurally non-serveable (CanServe()==false) and
// marked cost_latency_only — rather than a prose promise in a metric's HELP text.
// cachedTok is the cumulative cache_read tokens observed across served turns.
func providerCacheEvidence(cachedTok uint64) cachemeta.LookupVerdict {
	entry := cachemeta.FromProviderCache(cachemeta.ProviderCache{CachedTokens: int64(cachedTok)})
	return cachemeta.MaterializeVerdict(
		cachemeta.MatProviderPrefix, entry, cachemeta.MaterializationKey{}, cachemeta.QualityEvidence{})
}

// ProviderCacheEvidence classifies the summary's provider prompt-cache reuse
// (CachedPromptTokens) through the #432 bridge: provider cache is PERFORMANCE
// evidence (cost/latency), never local TRUST. The returned verdict is structurally
// non-serveable (CanServe()==false, Meta["provider_cache"]=="cost_latency_only"), so
// a consumer that prints the cached-token saving (the `fak guard` exit summary) can
// prove from the kernel's own gate that it is reporting performance, not trust — the
// cache reuse can never be promoted to authority that a local result may be re-served.
func (s AdjudicationSummary) ProviderCacheEvidence() cachemeta.LookupVerdict {
	return providerCacheEvidence(s.CachedPromptTokens)
}
