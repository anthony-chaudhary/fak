package cachevalue

// Compaction net-economics classifier (#3028).
//
// A smaller compacted prompt is not automatically cheaper or faster. The
// built-in-compaction audit (docs/notes/BUILT-IN-COMPACTION-AUDIT-2026-07-06.md)
// found compaction routinely reports a token REDUCTION without proving net cache
// economics: if the rewrite destroys a warm prompt-cache prefix, the tokens it
// sheds were cheap warm reads (~0.1x base) while the re-established prefix bills
// as cache-creation writes (~1.25x base). Shrinking the prompt can therefore
// RAISE the full-price bill.
//
// This classifier scores ONE compaction event in the same input-token-equivalent
// currency the cache-value rollup uses (cmd/fak/info_cache_ablation.go, the guard
// exit banner), so "tokens shed" and "net cheaper after cache effects" are two
// distinct, comparable numbers — never conflated. It flags the exact bad case the
// audit named: the prompt shrank but the net token-equivalent cost ROSE because
// the rewrite busted a warm prefix (ReasonCompactionCacheHostile).
//
// Boundary (honesty fence, issue #3028): fak can PRESERVE and REPORT byte
// identity and provider cache counters, but it cannot FORCE a provider cache
// hit. This scores a single before/after sample from WITNESSED counters. The
// provider re-warm cost is the cache-creation counter on the first post-rewrite
// turn; when the provider did not report it (a dollar-blind / codex row), the
// net verdict ABSTAINs (ReasonCompactionEconomicsUnknown) rather than inventing a
// hit — the same fail-open discipline Fold uses for an unknown hit rate.

// Token-equivalent marginals, matching the package header's documented pricing
// (writes bill ~1.25x base, reads ~0.1x base) and cacheprice.ShedTokenEquiv /
// the info-cache ablation pane. Kept as local named constants so this leaf stays
// import-free (std only), consistent with the rest of the package.
const (
	// warmReadMarginal is what a token already served as a provider cache_read
	// is worth when shed: it was only costing the read marginal to KEEP, so it
	// can only save that much when DROPPED.
	warmReadMarginal = 0.1
	// coldInputMarginal is the full-price (uncached) input marginal: a shed cold
	// token saves its whole cost.
	coldInputMarginal = 1.0
	// cacheWriteMarginal is the cache-creation (write) premium the first
	// post-rewrite turn pays to re-establish a prefix the rewrite moved.
	cacheWriteMarginal = 1.25
)

// compactionNetEps is the token-equivalent dead-band below which a net move is
// treated as noise, so float rounding never flips a NEUTRAL compaction into a
// spurious cheaper/hostile verdict.
const compactionNetEps = 1e-9

// Stable reason tokens for a compaction net-economics verdict. Fixed strings so a
// guard summary or report renderer matches on them rather than parsing prose,
// mirroring the Reason* regression tokens above.
const (
	// ReasonCompactionNetCheaper: the shed span's saved value exceeded the
	// re-warm cost — the compaction was net cheaper after cache effects.
	ReasonCompactionNetCheaper = "compaction_net_cheaper"
	// ReasonCompactionCacheHostile: the prompt shrank (ShedTokens > 0) but the
	// re-warm cost EXCEEDED the shed's value — full-price input rose because the
	// rewrite busted a warm prefix. The bad case #3028 exists to surface.
	ReasonCompactionCacheHostile = "compaction_cache_hostile"
	// ReasonCompactionNeutral: nothing was shed, or the net move sat inside the
	// dead-band — no economic claim either way.
	ReasonCompactionNeutral = "compaction_neutral"
	// ReasonCompactionEconomicsUnknown: the provider did not report the
	// post-rewrite cache-creation counter, so net economics cannot be proven —
	// the shed value is still reported, but the net verdict abstains.
	ReasonCompactionEconomicsUnknown = "compaction_economics_unknown"
)

// CompactionSample is the before/after telemetry for ONE compaction pass. The
// prompt-size fields record the shrink; the shed warm/cold split and the
// post-rewrite cache-creation counter drive the net verdict. A caller derives it
// from a warm before-turn and the first settled post-rewrite turn (see the
// steady-state note on ClassifyCompaction).
//
// The provider-side half (the re-warm counters) can either be pasted in by the
// caller's own convention or PROVEN: a caller that stamps JoinKey hands the
// sample to JoinProviderUsage (compaction_join.go, #2788), which fills the
// provider counters from the one usage record sharing the fire's coordinate and
// withdraws them when that binding cannot be proven. An unstamped sample is the
// pre-join path and is left exactly as the caller assembled it.
type CompactionSample struct {
	// PromptTokensBefore / PromptTokensAfter record the resident prompt size
	// around the rewrite, so the report can show the shrink the operator SEES
	// beside the net verdict the economics prove.
	PromptTokensBefore int64
	PromptTokensAfter  int64
	// ShedTokens is the count the compaction dropped (>= 0). Stored explicitly
	// (not just Before-After) so a rewrite that both sheds AND injects a summary
	// still reports its gross shed honestly.
	ShedTokens int64
	// ShedWarmTokens is how many of ShedTokens were being served as warm
	// provider cache_read before the rewrite — the cheap-to-keep tokens whose
	// shed saves only the warm marginal. Clamped to [0, ShedTokens] internally.
	ShedWarmTokens int64
	// RewriteCacheCreationTokens is the provider cache_creation (write) tokens
	// the first post-rewrite turn paid to re-establish the prefix the rewrite
	// moved — the re-warm cost. Meaningful only when RewriteKnown.
	RewriteCacheCreationTokens int64
	// RewriteKnown is false when the provider did not report the post-rewrite
	// cache-creation counter (dollar-blind / codex): the net verdict abstains
	// rather than treating an absent counter as a phantom zero re-warm cost.
	RewriteKnown bool
	// ObservedCacheReadTokens is the OBSERVED provider cache_read the joined
	// turn reported — how much of that turn's prompt the provider still served
	// warm after the rewrite. It is provider-relayed evidence carried BESIDE the
	// verdict, not an input to it: ClassifyCompaction scores the shed value and
	// the re-warm write cost only, so a relayed read count can never move the net
	// sign. JoinProviderUsage stamps it (clamped non-negative) on a proven join
	// and withdraws it on an unproven one; zero when no join was performed.
	ObservedCacheReadTokens int64
	// JoinKey is the event-join coordinate this fire shares with the provider
	// usage record for the turn it rewrote (#2788; see compaction_join.go). The
	// zero value is UNSTAMPED — the honest state of a sample assembled without
	// turn context, which JoinProviderUsage passes through verbatim rather than
	// counting as a failed join. It is a correlation coordinate only; no verdict
	// reads it.
	JoinKey CompactionJoinKey
}

// CompactionVerdict is the classified reading for one compaction event. The shed
// value and the re-warm cost are kept as separate token-equivalent fields so the
// "tokens shed" story never blends into the "net cheaper" story; NetSavedTokEq is
// their difference (positive = net cheaper). UncachedTokenDelta is the estimated
// change in full-price (write) tokens the rewrite caused — the #3028 "estimated
// uncached-token delta" acceptance field.
type CompactionVerdict struct {
	// Sample is the telemetry the verdict was drawn from, kept whole so a flag
	// always carries its own evidence (mirrors Flag.Metrics).
	Sample CompactionSample
	// ShedValueSavedTokEq is the token-equivalent the shed span saved:
	// warm@0.1x + cold@1x. This is the honest value of "tokens shed", already
	// discounted for the warm portion — NOT the raw shed count.
	ShedValueSavedTokEq float64
	// RewriteCostTokEq is the token-equivalent re-warm cost: creation@1.25x.
	// Zero and not meaningful when NetKnown is false.
	RewriteCostTokEq float64
	// NetSavedTokEq is ShedValueSavedTokEq - RewriteCostTokEq: the net after
	// cache effects. Positive = net cheaper; negative = cache-hostile. Zero and
	// not meaningful when NetKnown is false.
	NetSavedTokEq float64
	// UncachedTokenDelta is the estimated change in full-price cache-creation
	// (write) tokens the rewrite forced — the audit's "increased full-price
	// input" signal. Equals RewriteCacheCreationTokens here (the re-warm writes
	// the rewrite newly paid for); positive is the cache-hostile direction.
	UncachedTokenDelta int64
	// NetKnown is false when RewriteKnown was false — net economics could not be
	// proven, so RewriteCostTokEq/NetSavedTokEq carry no claim.
	NetKnown bool
	// Reason is the stable verdict token (one of the ReasonCompaction* constants).
	Reason string
}

// ClassifyCompaction scores one compaction event's net economics. It reports the
// shed span's honest token-equivalent value (warm-discounted) ALWAYS, and — when
// the post-rewrite cache-creation counter is known — the net after subtracting
// the re-warm write cost, flagging the cache-hostile case where a smaller prompt
// cost MORE in full-price writes than the shed saved.
//
// Steady-state boundary: a compaction always pays a one-time re-write premium on
// the immediate post-rewrite turn, so RewriteCacheCreationTokens must be the
// counter attributable to the PREFIX the rewrite MOVED (a warm before-turn vs a
// settled after-turn), not raw first-turn creation — otherwise every compaction
// would read as hostile. The doc note in docs/explainers/long-session-economics.md
// carries the byte-identical-prefix argument this rests on.
func ClassifyCompaction(s CompactionSample) CompactionVerdict {
	warm := s.ShedWarmTokens
	if warm < 0 {
		warm = 0
	}
	if warm > s.ShedTokens {
		warm = s.ShedTokens
	}
	cold := s.ShedTokens - warm
	if cold < 0 {
		cold = 0
	}
	v := CompactionVerdict{
		Sample:              s,
		ShedValueSavedTokEq: warmReadMarginal*float64(warm) + coldInputMarginal*float64(cold),
	}
	if !s.RewriteKnown {
		// Cannot prove net economics without the re-warm counter: report the shed
		// value, abstain on the net (fail-open, like an unknown hit rate).
		v.Reason = ReasonCompactionEconomicsUnknown
		return v
	}
	v.NetKnown = true
	v.RewriteCostTokEq = cacheWriteMarginal * float64(s.RewriteCacheCreationTokens)
	v.NetSavedTokEq = v.ShedValueSavedTokEq - v.RewriteCostTokEq
	v.UncachedTokenDelta = s.RewriteCacheCreationTokens
	switch {
	case s.ShedTokens <= 0:
		// Nothing shed: no compaction economics to judge either way.
		v.Reason = ReasonCompactionNeutral
	case v.NetSavedTokEq < -compactionNetEps:
		v.Reason = ReasonCompactionCacheHostile
	case v.NetSavedTokEq > compactionNetEps:
		v.Reason = ReasonCompactionNetCheaper
	default:
		v.Reason = ReasonCompactionNeutral
	}
	return v
}

// CacheHostile reports whether the verdict flagged the audit's bad case: the
// prompt shrank but the rewrite's re-warm cost exceeded the shed's value. Convenience
// for a guard summary / report gate that keys on the one bad case rather than the
// full reason set.
func (v CompactionVerdict) CacheHostile() bool {
	return v.Reason == ReasonCompactionCacheHostile
}
