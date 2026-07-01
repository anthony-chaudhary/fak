package frontierswe

// This file is the deterministic fold that turns a FrontierSWE trial's periodic
// /metrics scrapes into the per-turn cache-reuse SERIES the time-to-solution thesis
// is proven on (epic #1706, C8 #1714). It is the long-horizon analogue of
// internal/cachewitness's single-snapshot Record: a 20-hour trial re-prefills a
// growing context every turn, and fak's RadixAttention serves that prefix from the
// cached KV on turns 2..N — this fold records how much, turn over turn, and derives
// the REALIZED reuse rate r that C14 plugs into the C4 TTS projection to turn a
// deterministic floor into a measurement.
//
// Why the arithmetic lives here (not in internal/cachewitness). frontierswe is an
// architest tier-1 leaf that imports nothing internal (mirroring internal/swebench's
// foundation status), so it cannot import the tier-1 cachewitness parser. The split
// is clean anyway: cachewitness.Parse (called by cmd/fak) turns each Prometheus
// /metrics body into three cumulative integers; this fold is the pure integer
// arithmetic over the resulting SEQUENCE. No I/O, no HTTP, no model — deterministic
// and unit-testable, exactly like geometry.go's TTS floor.
//
// The provenance discipline is carried through verbatim from cachewitness (the
// conflation-scorecard line): fak's OWN KV-prefix reuse is WITNESSED; the upstream
// provider's cache_read is OBSERVED. The two are DISTINCT signals over DISTINCT
// caches and are NEVER summed — a series that added them would conflate the trust
// classes. They stay in separate fields and neither is derived from the other.

// CacheWitnessSchema is the versioned schema id stamped on the folded trace, so a
// long trial's reuse curve is inspectable and machine-joinable after the fact.
const CacheWitnessSchema = "fak.frontierswe.cache-witness.v1"

// Provenance vocabulary — mirrors internal/cachewitness so a downstream reader (or
// the conflation scorecard) reads the same trust labels across both benchmarks.
const (
	witnessed = "WITNESSED"
	observed  = "OBSERVED"
)

// CacheSample is one periodic /metrics scrape's CUMULATIVE cache counters during a
// FrontierSWE trial. fak's gateway exposes the KV-prefix family as monotonic
// *_total counters, so a scrape yields cumulative sums across turns 1..k; the
// per-turn/per-interval series is the delta between consecutive samples. The three
// numbers a scrape yields: fak's own KV-prefix reuse (WITNESSED — PromptTokens,
// ReusedTokens) and the provider cache_read (OBSERVED — ProviderCacheReadTokens).
type CacheSample struct {
	// Turn is the trajectory turn (or scrape ordinal) this snapshot was taken at;
	// informational, used only to label the folded point.
	Turn int `json:"turn"`
	// PromptTokens is cumulative prefill tokens across turns 1..Turn (WITNESSED).
	PromptTokens uint64 `json:"prompt_tokens"`
	// ReusedTokens is cumulative KV-prefix reused tokens 1..Turn (WITNESSED) — the
	// prefill the kernel did NOT redo, the headline datum.
	ReusedTokens uint64 `json:"reused_tokens"`
	// ProviderCacheReadTokens is cumulative provider cache_read (OBSERVED, relayed);
	// 0 on the pure in-kernel path.
	ProviderCacheReadTokens uint64 `json:"provider_cache_read_tokens"`
}

// CacheWitnessPoint is one folded point of the per-sample series: the cumulative
// reuse up to this scrape plus the DELTA since the prior scrape (the reuse the
// cache bit during that interval). All WITNESSED except the OBSERVED provider echo.
type CacheWitnessPoint struct {
	Turn int `json:"turn"`

	// Cumulative (WITNESSED): the running sums this scrape reported.
	CumPromptTokens uint64  `json:"cum_prompt_tokens"`
	CumReusedTokens uint64  `json:"cum_reused_tokens"`
	CumReuseRatio   float64 `json:"cum_reuse_ratio"`

	// Delta since the prior scrape (WITNESSED): the per-interval reuse. On a growing
	// trajectory this is the turn-2..N cache bite the thesis is about.
	DeltaPromptTokens uint64  `json:"delta_prompt_tokens"`
	DeltaReusedTokens uint64  `json:"delta_reused_tokens"`
	DeltaReuseRatio   float64 `json:"delta_reuse_ratio"`

	// OBSERVED: the provider cache_read echoed for this scrape — never fak's, never
	// summed into the reused-token series.
	ProviderCacheReadTokens uint64 `json:"provider_cache_read_tokens"`

	// Regressed marks a scrape whose cumulative counters went BACKWARDS versus the
	// prior one (a gateway restart mid-trial). Its deltas are clamped to 0 rather
	// than wrapping negative, and the flag is surfaced so the curve stays honest.
	Regressed bool `json:"regressed,omitempty"`
}

// CacheWitnessSeries is the folded trajectory: the per-sample series, the realized
// reuse rate (the C14 input), whether fak's cache ever bit, and the provenance map.
type CacheWitnessSeries struct {
	Schema string              `json:"schema"`
	Points []CacheWitnessPoint `json:"points"`

	// RealizedReuseRate is the final cumulative reused/prompt ratio — the MEASURED
	// cross-turn reuse rate r C14 plugs into the C4 TTS projection (TTSRatio(r)) to
	// turn the deterministic floor into a measurement. WITNESSED. 0 when no turn was
	// observed (never a divide-by-zero).
	RealizedReuseRate float64 `json:"realized_reuse_rate"`

	// FinalPromptTokens / FinalReusedTokens are the trajectory totals the rate is
	// computed from, kept so a reader can re-derive it. WITNESSED.
	FinalPromptTokens uint64 `json:"final_prompt_tokens"`
	FinalReusedTokens uint64 `json:"final_reused_tokens"`

	// ProviderCacheReadTokens is the final OBSERVED provider cache_read, echoed
	// separately from the WITNESSED reuse it must never be summed with.
	ProviderCacheReadTokens uint64 `json:"provider_cache_read_tokens"`

	// CacheBit reports whether fak's own cache actually engaged on this trial: at
	// least one turn reused a non-zero prefix. A trajectory with 0 reused tokens is
	// reported honestly as "did not bite", not silently as a small win.
	CacheBit bool `json:"cache_bit"`

	// Provenance maps each family to its trust class so the trace is self-describing.
	Provenance map[string]string `json:"provenance"`
}

// FoldCacheWitness folds an ordered sequence of cumulative /metrics scrapes into the
// per-turn reuse series + realized reuse rate. Samples must be in trajectory order
// (ascending Turn); the fold does not sort them, because the caller scrapes in time
// order and a re-sort would hide a real out-of-order capture bug. Deterministic and
// I/O-free.
func FoldCacheWitness(samples []CacheSample) CacheWitnessSeries {
	s := CacheWitnessSeries{
		Schema: CacheWitnessSchema,
		Points: make([]CacheWitnessPoint, 0, len(samples)),
		Provenance: map[string]string{
			"cum_reused_tokens":          witnessed, // fak's own RadixAttention prefix match
			"delta_reused_tokens":        witnessed,
			"realized_reuse_rate":        witnessed,
			"provider_cache_read_tokens": observed, // the upstream provider's cache_read, relayed
		},
	}

	var prev *CacheSample
	for i := range samples {
		cur := samples[i]
		pt := CacheWitnessPoint{
			Turn:                    cur.Turn,
			CumPromptTokens:         cur.PromptTokens,
			CumReusedTokens:         cur.ReusedTokens,
			CumReuseRatio:           ratio(cur.ReusedTokens, cur.PromptTokens),
			ProviderCacheReadTokens: cur.ProviderCacheReadTokens,
		}
		if prev != nil {
			// Cumulative counters are monotonic; a backwards step is a gateway restart.
			// Clamp the deltas to 0 and flag it rather than wrapping a uint64 negative.
			if cur.PromptTokens < prev.PromptTokens || cur.ReusedTokens < prev.ReusedTokens {
				pt.Regressed = true
			} else {
				pt.DeltaPromptTokens = cur.PromptTokens - prev.PromptTokens
				pt.DeltaReusedTokens = cur.ReusedTokens - prev.ReusedTokens
				pt.DeltaReuseRatio = ratio(pt.DeltaReusedTokens, pt.DeltaPromptTokens)
			}
		} else {
			// First scrape: the whole cumulative sum IS its own delta (turn-1 prefill).
			pt.DeltaPromptTokens = cur.PromptTokens
			pt.DeltaReusedTokens = cur.ReusedTokens
			pt.DeltaReuseRatio = pt.CumReuseRatio
		}
		s.Points = append(s.Points, pt)
		prev = &samples[i]
	}

	if prev != nil {
		s.FinalPromptTokens = prev.PromptTokens
		s.FinalReusedTokens = prev.ReusedTokens
		s.ProviderCacheReadTokens = prev.ProviderCacheReadTokens
		s.RealizedReuseRate = ratio(prev.ReusedTokens, prev.PromptTokens)
		s.CacheBit = prev.ReusedTokens > 0
	}
	return s
}

// ratio is reused/prompt as a float, 0 when the denominator is 0 (never a
// divide-by-zero), mirroring cachewitness.KVPrefixWitness.ReuseRatio.
func ratio(reused, prompt uint64) float64 {
	if prompt == 0 {
		return 0
	}
	return float64(reused) / float64(prompt)
}
