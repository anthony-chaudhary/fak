package gateway

import "github.com/anthony-chaudhary/fak/internal/guardvars"

// debugVCacheVars surfaces the NET realized provider-cache economics (read rebate MINUS
// write premium) the same way `fak vcache observe` does — computed via
// vcachegov.ProveTelemetrySavings over the session's cumulative cache counters, so the
// /debug/vars block, the fak_vcache_* metrics, and the offline observe Aggregate all
// agree on the same totals. Every value is OBSERVED (provider-relayed); a hit is a
// realized rebate, never local trust. Nil until a turn carries provider cache activity.
type debugVCacheVars struct {
	CacheReadTokens     uint64  `json:"cache_read_tokens"`     // OBSERVED read axis
	CacheCreationTokens uint64  `json:"cache_creation_tokens"` // OBSERVED write axis
	InputTokens         uint64  `json:"input_tokens"`          // OBSERVED uncached remainder
	BaselineTokenEquiv  float64 `json:"baseline_token_equiv"`
	ActualTokenEquiv    float64 `json:"actual_token_equiv"`
	SavedTokenEquiv     float64 `json:"saved_token_equiv"` // NET; negative until reads repay writes
	SavedPct            float64 `json:"saved_pct"`
	HitRate             float64 `json:"hit_rate"`
	Multiplier          float64 `json:"multiplier"`
	Status              string  `json:"status"` // PROVEN / REFUTED
}

// vcacheVarsFromSnapshot builds the /debug/vars vcache block from the same inference
// snapshot the Prometheus writeVCacheMetrics reads, so the two surfaces never disagree.
// It returns nil (the block is omitted) until a turn carried provider cache activity.
func vcacheVarsFromSnapshot(snap inferenceSnapshot) *debugVCacheVars {
	if snap.cachedTok == 0 && snap.cacheCreateTok == 0 {
		return nil
	}
	proof := vcacheProofFromCounters(snap.promptTok, snap.cachedTok, snap.cacheCreateTok)
	hit, mult := 0.0, 0.0
	if proof.BaselineTokenEquiv > 0 {
		hit = proof.CacheReadTokens / proof.BaselineTokenEquiv
	}
	if proof.ActualTokenEquiv > 0 {
		mult = proof.BaselineTokenEquiv / proof.ActualTokenEquiv
	}
	return &debugVCacheVars{
		CacheReadTokens:     snap.cachedTok,
		CacheCreationTokens: snap.cacheCreateTok,
		InputTokens:         snap.promptTok,
		BaselineTokenEquiv:  proof.BaselineTokenEquiv,
		ActualTokenEquiv:    proof.ActualTokenEquiv,
		SavedTokenEquiv:     proof.SavedTokenEquiv,
		SavedPct:            proof.SavedPct,
		HitRate:             hit,
		Multiplier:          mult,
		Status:              string(proof.Status),
	}
}

// debugCacheAttributionVars surfaces the provider-vs-fak owner split for cache-like savings
// LIVE on /debug/vars, mirroring the /metrics fak_cache_saved_*_by_owner and _by_mechanism
// families (writeCacheAttributionMetrics) field-for-field so an operator watching a session
// sees the SAME split the guard-exit banner prints — not a provider-only "saved X" that
// conflates the provider's default cache with fak-authored savings (#1490/#1844). Every
// token-equiv value is the same input-token currency as the vcache block; VDSO is a separate
// avoided-call counter (its witness is skipped engine calls, not prompt tokens). Nil until
// the split has anything nonzero to say.
// The wire shape lives in internal/guardvars so the `fak info` pane decodes the exact owner
// split this producer emits — one definition, no field-for-field hand-sync to drift.
type debugCacheAttributionVars = guardvars.CacheAttributionVars

// cacheAttributionVars builds the /debug/vars owner-split block from the SAME inputs the
// /metrics renderer folds (m.adjudicationSummary().MechanismSavings() + kernel VDSOHits +
// inline-served turns), so the two surfaces report identical owner totals on one session (the
// acceptance contract of #1849). It returns nil — omitting the block — until the split has a
// nonzero token slice OR an avoided call, so a cold session stays quiet rather than emitting
// an all-zero object. When the fak slice is anchor-starved (#1407) the block still renders
// the provider slice with fak reading ~0, honestly.
func positiveInt64(v int64) uint64 {
	if v <= 0 {
		return 0
	}
	return uint64(v)
}

func cacheAttributionVars(sum AdjudicationSummary, vdsoHits int64, servedInline uint64) *debugCacheAttributionVars {
	ms := sum.MechanismSavings()
	if vdsoHits > 0 {
		ms.FakVDSOAvoidedCalls += uint64(vdsoHits)
	}
	ms.FakVDSOAvoidedCalls += servedInline
	// The cold-tool-DEFER shed (#3647) is a THIRD mechanism that, unlike the two token-equiv
	// sheds, shrinks NO request bytes — its reduction is provider-side (only the hot core loads
	// into context), so it never lands in ms.HasAnyTokenActivity(). Fold it into the same gate
	// so a defer-ON session with no token slice still renders the block rather than reading as
	// a quiet cold session.
	// #3621: an INERT defer session is by definition one with no defer shed to render, so the
	// gate above would suppress the very block that has to carry the alarm. Fold the finding
	// into the gate so an armed-but-never-bit session renders rather than reading as quiet.
	inertDefer := sum.DeferEnabledButInert()
	if !ms.HasAnyTokenActivity() && ms.FakVDSOAvoidedCalls == 0 && sum.DeferColdCount == 0 && !inertDefer {
		return nil
	}
	finding := ""
	if inertDefer {
		finding = guardvars.FindingDeferEnabledButInert
	}
	return &debugCacheAttributionVars{
		ProviderTokenEquiv:                        ms.ProviderTokenEquiv(),
		FakTokenEquiv:                             ms.FakTokenEquiv(),
		TotalTokenEquiv:                           ms.TotalTokenEquiv(),
		ProviderPromptCacheReadTokenEquiv:         ms.ProviderPromptCacheReadTokenEquiv,
		ProviderPromptCacheWritePremiumTokenEquiv: ms.ProviderPromptCacheWritePremiumTokenEquiv,
		CacheCreationTokensHeadOnly:               sum.CacheCreationTokensHeadOnly,
		CacheCreationTokensMessagePrefix:          sum.CacheCreationTokensMessagePrefix,
		FakCompactionShedTokens:                   ms.FakCompactionShedTokens,
		FakCompactionCacheReadTokens:              ms.FakCompactionCacheReadTokens,
		FakKVPrefixReusedTokens:                   ms.FakKVPrefixReusedTokens,
		FakVDSOAvoidedCalls:                       ms.FakVDSOAvoidedCalls,
		FakResponseMemoCalls:                      positiveInt64(vdsoHits),
		FakInlineServedCalls:                      servedInline,
		FakDeferColdTurns:                         sum.DeferColdTurns,
		FakDeferColdCount:                         sum.DeferColdCount,
		FakDeferColdToolNames:                     sum.DeferColdToolNames,
		FakDeferStandDownTurns:                    sum.DeferStandDownTurns,
		FakDeferFinding:                           finding,
	}
}

// debugManagedCacheVars surfaces the managed-cache 1h TTL-upgrade POSTURE and its outcome
// counts LIVE on /debug/vars — the posture sibling of the #1849 cache_attribution owner
// split (#2190). Active is the resolved lever state (Server.cacheTTL1H, from
// --managed-cache / Config.CacheTTL1H) independent of whether any head was eligible;
// Upgraded and Reasons mirror the AdjudicationSummary.CacheTTLUpgrade* fields the durable
// gateway-usage ledger (internal/gatewayusageledger) and the fak_gateway_cache_ttl_upgrade_total
// family carry, so the live pane, the /metrics counter, and the ledger row report the same
// numbers. Inert names the misconfiguration signal the weekly review's ACTIVE-but-inert gap
// keys on: the lever is ON but every head refused, so a long idle session pays the 5m
// re-write the operator opted out of — visible here as a zero-upgrade ACTIVE posture rather
// than an absent panel.
// The wire shape lives in internal/guardvars so the `fak info` pane decodes the exact posture
// block this producer emits — one definition, no field-for-field hand-sync to drift.
type debugManagedCacheVars = guardvars.ManagedCacheVars

// managedCacheVars builds the /debug/vars managed-cache posture block from the session's
// resolved lever state (active), the resolved upstream wire (provider), and the SAME
// AdjudicationSummary ttl-upgrade fields the ledger row folds. It returns nil — omitting the
// block — only when the lever is OFF and nothing was observed, so a passive, cold session stays
// quiet; an ACTIVE session renders even at zero upgrades, keeping "the lever fired and paid"
// distinct from "the lever is off" on the live surface. Reasons is refusal-only (the "upgraded"
// outcome lives in Upgraded), inherited from AdjudicationSummary.
//
// Inert is WIRE-AWARE: it names the #2190 ACTIVE-but-inert misconfiguration only on a wire that
// HAS the Anthropic 1h-TTL lever. On the OpenAI Responses (codex) wire that lever can never move
// (fak's real lever there is the pinned prompt_cache_key), so an ACTIVE zero-upgrade session is
// NOT inert — mirroring bannerLine's `provider == "openai-responses"` branch. Wire carries the
// resolved provider so the `fak info` and sessionaudit consumers stay wire-aware too.
func managedCacheVars(active bool, provider string, sum AdjudicationSummary) *debugManagedCacheVars {
	if !active && sum.CacheTTLUpgraded == 0 && len(sum.CacheTTLUpgradeReasons) == 0 {
		return nil
	}
	block := &debugManagedCacheVars{
		Active:   active,
		Upgraded: sum.CacheTTLUpgraded,
		Reasons:  sum.CacheTTLUpgradeReasons,
		Wire:     provider,
	}
	block.Inert = active && sum.CacheTTLUpgraded == 0 && !block.WireHasNo1hTTLLever()
	// The #3620 live watchdog: Inert says the lever is ON with nothing fired YET; Finding
	// upgrades that to an alarm once enough upgrade-eligible turns accrued all-refused —
	// the session is paying the 5m re-write the ACTIVE posture claimed to remove. Gated on
	// Inert so a passive session or a wire without the 1h-TTL lever can never carry it,
	// and cleared for good by the first fired upgrade (UpgradeNeverFired is then false).
	if block.Inert && sum.UpgradeNeverFired() {
		block.Finding = guardvars.FindingUpgradeNeverFired
	}
	return block
}
