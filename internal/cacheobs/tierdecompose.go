package cacheobs

// tierdecompose.go — COUNTER-DELTA TIER DECOMPOSITION + LINEAGE STATUS VOCABULARY.
//
// Borrow (study-repo, clean-room conceptual adaptation of
// anthony-chaudhary/Benchmark @b60f2f05e23ec38ae4f50ca19717c2f615ed419a; MIT-declared with no
// root LICENSE → ADAPT verdict, no source copied): §B.1 "tier-token decomposition from
// counter deltas" — L3 = Δprefetch; L2 = max(0, Δloadback − Δprefetch); L1 = Δcached − L2 −
// L3; miss = Δprompt × (1 − hit_rate); when L2+L3 overshoots Δcached (transfers include
// speculative/re-fetched blocks), rescale proportionally so l1+l2+l3 == cached holds exactly;
// §B.12 "preserve raw cumulative counters" — a mid-window counter reset is invisible after
// max(0,…) clamping, so detect it (any cumulative counter decreasing) and flag the row
// crash-affected instead of clamping silently; §B.13 "never collapse failure modes into one
// zero" — closed layer-classify status vocabulary plus NaN-not-zero sentinels: "0.0 is a
// measurement; NaN is not measured".
//
// The load-bearing counter fact the through-path subtraction rests on: an L3 (storage) hit
// transits the host pool, so it increments BOTH prefetched_tokens_total and
// load_back_tokens_total. Only the load-back volume BEYOND the prefetch volume is a pure
// host→device transfer; subtracting the transit out keeps the two tiers from double-counting
// the same tokens.
//
// The fak seam this closes: tiers.go books TierAccess records THIS process witnessed; this
// file attributes the cached tokens of a FOREIGN server's window to its L1/L2/L3 tiers from
// the counter deltas a scraper already collected. Leaf 1's alias table (capability.go)
// resolves the wire spellings into the canonical names these snapshot fields carry, so a
// scrape taken before or after an upstream rename feeds the same decomposition. The
// decomposer consumes caller-supplied cumulative snapshots only: no scrape/HTTP code, and it
// never fabricates TierAccess bookings into the observer.
//
// Exactness: every quantity is an integer token count carried in float64. For counter values
// below 2^53 each delta, split, and rescale step is exact integer arithmetic, so the
// invariant L1+L2+L3 == Δcached holds bit-for-bit — property-pinned over randomized inputs
// in the test sibling, including the overshoot regime.
//
// Route: clean-room conceptual adaptation (ADAPT); behavior contracts from the pinned study
// note §B.1/§B.12/§B.13; no bytes vendored.

import "math"

// CounterSnapshot is one cumulative scrape of a foreign server's cache counters — the
// running totals, NOT window deltas: DecomposeTiers differences two of these. Field names
// mirror the canonical metric names of the alias table (capability.go), so a caller that
// resolves wire spellings through ResolveCanonical fills them one-for-one.
type CounterSnapshot struct {
	// CachedTokens is the cumulative cached_tokens_total: tokens served from cache.
	CachedTokens uint64
	// PromptTokens is the cumulative prompt_tokens_total: prompt tokens processed.
	PromptTokens uint64
	// EvictedTokens is the cumulative evicted_tokens_total: host-pool evictions.
	EvictedTokens uint64
	// LoadBackTokens is the cumulative load_back_tokens_total: host→device transfers. An L3
	// hit transits the host pool, so it increments this counter TOO — the through-path
	// subtraction in DecomposeTiers strips that transit out.
	LoadBackTokens uint64
	// BackupedTokens is the cumulative backuped_tokens_total: device→storage backups.
	BackupedTokens uint64
	// PrefetchedTokens is the cumulative prefetched_tokens_total: storage→host fetches.
	PrefetchedTokens uint64
}

// LayerStatus is the CLOSED lineage/status vocabulary of a tier-decomposition row: HOW the
// row's tier attribution was produced. It keeps apart the readings a bare number conflates —
// "the server told us", "we derived it from counter deltas", "we tried to derive it and the
// counters gave no signal", "nothing moved", "the counters themselves are broken".
type LayerStatus int

const (
	// LayerStatusServer: the server itself reported the tier attribution (a direct
	// per-request signal). Part of the closed vocabulary so mixed-row consumers can switch
	// on it; DecomposeTiers never emits it — it only derives.
	LayerStatusServer LayerStatus = iota
	// LayerStatusDerived: tier attribution was derived from counter deltas and at least one
	// tier earned a non-zero share.
	LayerStatusDerived
	// LayerStatusDerivedNoSignal: derivation ran on moving input but no tier earned a
	// non-zero share (pure-miss windows, flat transfer counters by construction,
	// speculative prefetch that cached nothing). The flag fires on OUTPUT, never on input
	// presence.
	LayerStatusDerivedNoSignal
	// LayerStatusNoMovement: no counter moved in the window. Its zeros mean "nothing
	// happened", not "an idle cache measured zero".
	LayerStatusNoMovement
	// LayerStatusCrashAffected: at least one cumulative counter DECREASED across the window
	// (a counter reset, or a crash/restart that zeroed the counters). The deltas span the
	// reset and are untrustworthy; the row is flagged, never clamped.
	LayerStatusCrashAffected
	// numLayerStatuses bounds the closed vocabulary. Not a status.
	numLayerStatuses
)

// String renders the label spelling of a status; "unknown" out of range.
func (s LayerStatus) String() string {
	switch s {
	case LayerStatusServer:
		return "server"
	case LayerStatusDerived:
		return "derived"
	case LayerStatusDerivedNoSignal:
		return "derived_no_signal"
	case LayerStatusNoMovement:
		return "no_movement"
	case LayerStatusCrashAffected:
		return "crash_affected"
	default:
		return "unknown"
	}
}

func (s LayerStatus) valid() bool { return s >= 0 && s < numLayerStatuses }

// AllLayerStatuses returns the closed status vocabulary in declaration order.
//
//enumlint:exempt numLayerStatuses is the exclusive bound sentinel, not a reportable status.
func AllLayerStatuses() []LayerStatus {
	return []LayerStatus{
		LayerStatusServer,
		LayerStatusDerived,
		LayerStatusDerivedNoSignal,
		LayerStatusNoMovement,
		LayerStatusCrashAffected,
	}
}

// TierDecomposition is the attributed row for ONE window between two cumulative snapshots:
// the measured deltas, the L1/L2/L3 split of the cached tokens, the miss figure, and the
// lineage status that says how much of the row to trust.
//
// Sentinel discipline (§B.13): 0 is a MEASUREMENT ("counted, and nothing moved"); NaN is NOT
// MEASURED (an undefined ratio, or a window poisoned by a counter reset). A reader branches
// on Status and math.IsNaN — it never treats a NaN field as zero.
type TierDecomposition struct {
	// The window's measured counter deltas (tokens). Always populated, including on
	// crash-affected rows: a negative delta IS the reset evidence.
	CachedDelta   float64
	PromptDelta   float64
	LoadBackDelta float64
	PrefetchDelta float64

	// L1 is the device-resident (HBM) reuse: the residual of the cached tokens after the
	// transfer tiers are subtracted. NaN when not measured.
	L1 float64
	// L2 is the pure host→device transfer volume: Δloadback with the storage transit
	// (Δprefetch) subtracted out. NaN when not measured.
	L2 float64
	// L3 is the storage→host prefetch volume: Δprefetch. NaN when not measured.
	L3 float64
	// Miss is the recomputed tokens: Δprompt × (1 − HitRate). NaN when HitRate is undefined.
	Miss float64
	// HitRate is the window hit rate, Δcached / Δprompt. NaN when Δprompt is 0.
	HitRate float64

	// Status is the closed lineage vocabulary: how this row was produced, and whether its
	// numbers are trustworthy.
	Status LayerStatus
	// ResetCounters names the canonical counters that decreased across the window, in
	// schema order. Non-empty exactly when Status == LayerStatusCrashAffected.
	ResetCounters []string
}

// DecomposeTiers attributes the cached tokens of the window between two cumulative counter
// snapshots to the L1/L2/L3 tiers:
//
//	L3 = Δprefetch                      (storage→host fetches)
//	L2 = max(0, Δloadback − Δprefetch)  (host→device, storage transit subtracted out)
//	L1 = Δcached − L2 − L3              (device-resident reuse, the residual)
//	hit_rate = Δcached / Δprompt; miss = Δprompt × (1 − hit_rate)
//
// When L2+L3 overshoots Δcached — transfers include speculative or re-fetched blocks that
// did not end up counted as cached — L2 and L3 are rescaled PROPORTIONALLY (their ratio
// preserved) onto the cached total, leaving L1 the zero residual, so L1+L2+L3 equals
// Δcached EXACTLY.
//
// The function is total: it never errors. A window in which any cumulative counter decreased
// is returned as LayerStatusCrashAffected with NaN tier fields and the offending counters
// named — the reset is surfaced, never clamped away. A window in which no counter moved is
// LayerStatusNoMovement. Otherwise the row is LayerStatusDerived, or
// LayerStatusDerivedNoSignal when the derivation ran but no tier earned a non-zero share.
//
// Exactness: for counter values below 2^53 every step is exact integer arithmetic in float64,
// so the rescale invariant L1+L2+L3 == Δcached holds bit-for-bit under either summation
// association.
func DecomposeTiers(before, after CounterSnapshot) TierDecomposition {
	type namedDelta struct {
		name  string
		value float64
	}
	// Schema order: this order is also the ResetCounters order.
	deltas := []namedDelta{
		{"cached_tokens_total", counterDelta(after.CachedTokens, before.CachedTokens)},
		{"prompt_tokens_total", counterDelta(after.PromptTokens, before.PromptTokens)},
		{"evicted_tokens_total", counterDelta(after.EvictedTokens, before.EvictedTokens)},
		{"load_back_tokens_total", counterDelta(after.LoadBackTokens, before.LoadBackTokens)},
		{"backuped_tokens_total", counterDelta(after.BackupedTokens, before.BackupedTokens)},
		{"prefetched_tokens_total", counterDelta(after.PrefetchedTokens, before.PrefetchedTokens)},
	}

	out := TierDecomposition{
		CachedDelta:   deltas[0].value,
		PromptDelta:   deltas[1].value,
		LoadBackDelta: deltas[3].value,
		PrefetchDelta: deltas[5].value,
		L1:            math.NaN(),
		L2:            math.NaN(),
		L3:            math.NaN(),
		Miss:          math.NaN(),
		HitRate:       math.NaN(),
	}

	// Reset detection FIRST: a decreasing cumulative counter means the window spans a
	// counter reset (or a crash/restart that zeroed the counters). Clamping those deltas to
	// zero would fabricate a measurement; flag the row and name the offenders instead.
	var reset []string
	for _, d := range deltas {
		if d.value < 0 {
			reset = append(reset, d.name)
		}
	}
	if len(reset) > 0 {
		out.Status = LayerStatusCrashAffected
		out.ResetCounters = reset
		return out
	}

	c, p := deltas[0].value, deltas[1].value
	evict, lb, bk, pf := deltas[2].value, deltas[3].value, deltas[4].value, deltas[5].value

	if c == 0 && p == 0 && evict == 0 && lb == 0 && bk == 0 && pf == 0 {
		// Nothing moved anywhere: the tier zeros below are MEASURED zeros (the counters
		// were present and flat), and the status says so. HitRate and Miss stay NaN — a
		// 0/0 ratio and a zero-prompt miss are not measurements.
		out.Status = LayerStatusNoMovement
		out.L1, out.L2, out.L3 = 0, 0, 0
		return out
	}

	// Through-path subtraction: an L3 hit transits the host pool, so it increments BOTH the
	// prefetch and the load-back counters. Only the load-back volume BEYOND the prefetch
	// volume is a pure host→device transfer.
	l3 := pf
	l2 := lb - pf
	if l2 < 0 {
		l2 = 0
	}
	l1 := c - l2 - l3
	if l1 < 0 {
		// Overshoot: the transferred volume exceeds the cached volume (speculative or
		// re-fetched blocks). Rescale L2 and L3 proportionally — their ratio preserved —
		// onto the cached total, leaving L1 the zero residual. Every operand is an
		// integer-valued float below 2^53, so l2' = c − l3' is exact and L1+L2+L3 == c
		// holds bit-for-bit.
		scale := c / (l2 + l3)
		l3 = math.Round(l3 * scale)
		l2 = c - l3
		l1 = 0
	}
	out.L1, out.L2, out.L3 = l1, l2, l3

	if p > 0 {
		out.HitRate = c / p
		out.Miss = p * (1 - out.HitRate)
	}

	if out.L1 == 0 && out.L2 == 0 && out.L3 == 0 {
		// Derivation ran on moving input but no tier earned a share: the flag fires on
		// output, never on input presence.
		out.Status = LayerStatusDerivedNoSignal
	} else {
		out.Status = LayerStatusDerived
	}
	return out
}

// counterDelta returns after − before as a float64. Exact for counter values below 2^53 —
// the range no cumulative token counter reaches in practice.
func counterDelta(after, before uint64) float64 {
	return float64(after) - float64(before)
}
