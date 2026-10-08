package cacheobs

// Regime names one turn's reuse cliff bucket. The spelling is shared by
// fak_gateway_kv_prefix_turns_by_regime_total, the serving-latency regime cut (#5630),
// and the per-request perf ledger, so the three surfaces join on one label value.
const (
	RegimeFrozen  = "frozen"
	RegimePartial = "partial"
	RegimeCold    = "cold"
	RegimeUnknown = "unknown"
)

// Regimes is the closed regime vocabulary in render order.
var Regimes = [...]string{RegimeFrozen, RegimePartial, RegimeCold, RegimeUnknown}

// RegimeOf buckets a reuse ratio in [0,1] by FrozenFloor / ColdCeil.
func RegimeOf(ratio float64) string {
	switch {
	case ratio >= FrozenFloor:
		return RegimeFrozen
	case ratio < ColdCeil:
		return RegimeCold
	default:
		return RegimePartial
	}
}

// MinPartialReuseTokens is the smallest cache read RegimeForTokens accepts as partial
// reuse. A short cold prompt that only re-reads a chat-template/system header (18
// tokens on the Halo Qwen template) is cold, not partial: that hit cannot move TTFT.
const MinPartialReuseTokens = 64

// RegimeForTokens classifies a turn from its cache-read and uncached prompt tokens
// (disjoint counts). A turn with no prompt tokens on either side has no observable
// reuse and is RegimeUnknown rather than defaulted into a real regime.
func RegimeForTokens(cachedTokens, uncachedTokens int) string {
	if cachedTokens < 0 {
		cachedTokens = 0
	}
	if uncachedTokens < 0 {
		uncachedTokens = 0
	}
	total := cachedTokens + uncachedTokens
	if total == 0 {
		return RegimeUnknown
	}
	regime := RegimeOf(float64(cachedTokens) / float64(total))
	if regime == RegimePartial && cachedTokens < MinPartialReuseTokens {
		return RegimeCold
	}
	return regime
}
