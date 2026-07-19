package cacheprice

// retention.go lifts this package's per-turn disaggregation pricing (DisaggregationDividend)
// to the decision a capacity-bounded disaggregated KV pool actually makes: which prefixes to
// keep. The dividend prices ONE fetch across the fabric; a remote pool holds a prefix across
// MANY fetches under FINITE capacity, so the retention question is amortized and per-byte, not
// per-turn. Three things change the economics that the per-turn leaves deliberately do not model:
//
//   - Amortization: a resident prefix returns its dividend once PER reuse before it churns, so
//     its lifetime value is dividendPerFetch × expectedFetches, not a single dividend.
//   - Residency cost: unlike a one-shot fetch, holding a prefix warm in the pool costs tokens
//     over its life (keep-warm transfers, storage priced in recompute-equivalent tokens).
//   - Capacity: the pool is bounded, so what matters under pressure is value PER UNIT of pool
//     footprint the prefix consumes — a dividend DENSITY, the cost-aware caching key (GDSF),
//     not the raw dividend.
//
// The sign that DisaggregationDividend makes load-bearing stays load-bearing here, and inverts
// the naive policy: a prefix whose cross-fabric fetch LOSES to local recompute has a negative
// per-fetch dividend, so more reuse only deepens its loss — it must be the FIRST evicted no
// matter how hot it is, the exact opposite of what frequency-only LRU/LFU would retain. Like the
// rest of cacheprice this is a tier-1 foundation leaf: plain ints and one value struct, importing
// nothing internal, so the caller supplies the estimates (expected reuse, residency price) and
// this stays the single source of truth for "what a resident prefix is worth to the pool".

// DisaggregationRetentionValue returns the SIGNED net compute, in recompute-token equivalents,
// of keeping a KV prefix resident in the remote / disaggregated tier over its remaining life:
//
//	retention = dividendPerFetch × expectedFetches − residencyCostTokens        (SIGNED)
//
// dividendPerFetch is the prefix's signed per-fetch DisaggregationDividend (external − overhead:
// the recompute a single remote fetch saves after the fabric toll) and is itself already signed —
// it is NOT clamped, because a prefix that loses per fetch must be able to book a lifetime loss.
// expectedFetches is how many more times the prefix is expected to be served from the pool before
// it churns; residencyCostTokens is the token-equivalent price of holding it warm over that same
// horizon. Both of those are non-negative counts and clamp defensively to 0.
//
// The result is SIGNED on purpose, and the sign is the decision: POSITIVE means the prefix returns
// more net compute over its life than it costs to keep — it earns its place in the pool. NEGATIVE
// means holding it costs more than it ever returns; it should be evicted (or never admitted),
// even when a single fetch looked worthwhile, because either a negative per-fetch dividend
// compounds over reuse or a positive one never repays the residency toll. Zero is break-even.
// Deliberately unclamped so that verdict survives, exactly as DisaggregationDividend leaves its
// own sign intact.
func DisaggregationRetentionValue(dividendPerFetch, expectedFetches, residencyCostTokens int) int {
	if expectedFetches < 0 {
		expectedFetches = 0
	}
	if residencyCostTokens < 0 {
		residencyCostTokens = 0
	}
	return dividendPerFetch*expectedFetches - residencyCostTokens
}

// RemoteResident describes one KV prefix resident in the disaggregated pool, carrying just what
// pricing an eviction needs. CapacityTokens is the prefix's footprint in the pool (its KV size in
// token-equivalents) and is what the retention value is weighed against — value per byte held.
type RemoteResident struct {
	Key              string // opaque prefix identifier, for the caller to map back to its entry
	DividendPerFetch int    // signed per-fetch DisaggregationDividend
	ExpectedFetches  int    // expected remaining fetches before the prefix churns
	ResidencyCost    int    // token-equivalent cost of holding it warm over that horizon
	CapacityTokens   int    // pool footprint the prefix occupies; treated as ≥ 1 when weighing density
}

// RetentionValue is DisaggregationRetentionValue for this resident.
func (r RemoteResident) RetentionValue() int {
	return DisaggregationRetentionValue(r.DividendPerFetch, r.ExpectedFetches, r.ResidencyCost)
}

// AdmitToRemote reports whether the prefix earns a place in the disaggregated pool: its lifetime
// retention value is strictly positive. It is to the pool what DisaggregationWorthwhile is to a
// single fetch — admit only what nets a gain — but priced over the whole residency, so a prefix
// whose per-fetch dividend is positive yet cannot repay its keep-warm cost is correctly refused.
// Break-even (zero) admits nothing: with no net gain, keep the pool for prefixes that do pay.
func AdmitToRemote(r RemoteResident) bool {
	return r.RetentionValue() > 0
}

// EvictionVictim returns the index of the resident to evict FIRST under capacity pressure: the one
// returning the LEAST net compute PER UNIT of pool footprint it holds — the lowest retention
// DENSITY (RetentionValue / CapacityTokens). Density is compared EXACTLY by cross-multiplication so
// the leaf stays float-free; a footprint of 0 or less is treated as 1 (a resident occupies at least
// one token of pool), keeping every density well defined. Because the value is signed, a resident
// that costs more than it returns has negative density and sorts below every profitable one — it is
// evicted before any prefix that still pays, which is the whole point: frequency does not save a
// prefix whose fetch loses to recompute. Ties in density break toward evicting the LARGER footprint
// (freeing more pool per eviction), then the lower index for determinism. Returns -1 for an empty
// slice.
func EvictionVictim(residents []RemoteResident) int {
	victim := -1
	var vVal, vCap int // retention value and (floored-to-1) footprint of the current victim
	for i := range residents {
		val := residents[i].RetentionValue()
		cap := residents[i].CapacityTokens
		if cap < 1 {
			cap = 1
		}
		if victim == -1 {
			victim, vVal, vCap = i, val, cap
			continue
		}
		// density_i < density_victim  ⟺  val·vCap < vVal·cap  (caps > 0, direction preserved).
		lhs := val * vCap
		rhs := vVal * cap
		switch {
		case lhs < rhs:
			victim, vVal, vCap = i, val, cap
		case lhs == rhs && cap > vCap:
			// Equal density: prefer to evict the one that frees more pool.
			victim, vVal, vCap = i, val, cap
		}
	}
	return victim
}
