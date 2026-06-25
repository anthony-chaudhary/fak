package vcachecal

// probe.go is the vCache offline-calibration probe harness — issue #716 scope 2 and
// §11.4 Law D2 (calibrate-don't-assume). Provider TTL / minimum-prefix / read-discount
// are version-dependent and undocumented for some providers, so vCache PROBES: send a
// known anchor, wait, re-send, read cache_read, and fit T, M_min, and r per (provider,
// model, endpoint). The result feeds the estimator's belief policy and the §5 economics
// INSTEAD of a hard-coded number.
//
// This file is the PURE fitting surface: it takes replay samples (which a live driver
// records by re-sending a known anchor at 30s/2m/5m/10m/30m/1h) and returns the fitted
// constants plus an LRU probe budget that stops calibration from evicting the warm set
// it is measuring. It issues no network call — observe-only (issue acceptance).

// Hypothesis is the §5 starting hypothesis for a provider's constants — the numbers the
// doc's economics stands on BEFORE the probe harness measures them. Every constant the
// harness cannot measure falls back to this and is flagged Assumed (Calibration.*
// Measured = false): calibrate-don't-assume, made mechanical.
type Hypothesis struct {
	TTLMillis       int64
	MinPrefixTokens int64
	ReadMult        float64
}

// DefaultHypothesis is the doc's starting point: a 5-minute TTL, the corrected 1024-
// token Anthropic minimum (Opus 4.8 / Sonnet 4.6 / Sonnet 4.5 — 4096 belongs to Haiku
// 4.5), and the 0.1 cached-read multiplier. These are a HYPOTHESIS, not ground truth;
// FitCalibration replaces each with a measured value when a probe observes it.
func DefaultHypothesis() Hypothesis {
	return Hypothesis{TTLMillis: 5 * 60 * 1000, MinPrefixTokens: 1024, ReadMult: 0.1}
}

// ProbeSample is one offline-calibration replay observation (scope 2): send a known
// anchor of PrefixTokens, wait DelayMillis, re-send, and read CachedTokens from the
// provider. ReadCostEquiv — the billed token-equivalent cost of the cached-read portion,
// when the provider itemizes it — lets the harness fit the read discount r; zero means
// "not observed" and r falls back to the hypothesis.
//
// The canonical replay ladder (scope 2) is DelayMillis ∈ {30s, 2m, 5m, 10m, 30m, 1h};
// a sample per delay bounds the TTL. A prefix-size sweep (varying PrefixTokens at a
// fixed short delay) bounds M_min.
type ProbeSample struct {
	Provider      string
	ModelID       string
	Endpoint      string
	DelayMillis   int64
	PrefixTokens  int64
	CachedTokens  int64
	ReadCostEquiv float64
}

// Calibration is the fitted per-(provider, model, endpoint) constant set every later
// milestone reads instead of a hard-coded number. Each field carries whether it was
// MEASURED from a probe or is the ASSUMED §5 hypothesis — the calibrate-don't-assume
// guard made mechanical and auditable.
type Calibration struct {
	Provider          string
	ModelID           string
	Endpoint          string
	TTLMillis         int64
	TTLMeasured       bool
	MinPrefixTokens   int64
	MinPrefixMeasured bool
	ReadMult          float64
	ReadMultMeasured  bool
}

// FitCalibration fits T (TTL), M_min (minimum cacheable prefix), and r (read discount)
// per (provider, model, endpoint) from real replay samples, falling back to the
// hypothesis for any constant no sample observed. Issue #716 acceptance: "the §5
// constants are read from the probe, not hard-coded."
//
//   - T  is the longest DelayMillis at which a re-send still read cache (CachedTokens>0).
//     A prefix that survives to delay D and misses at D' > D bounds the TTL to
//     [D, D'); the right-censored estimate T = D is the largest confirmed-warm
//     delay. When no sample hit, T falls back to the hypothesis (unmeasured).
//   - M_min is the smallest PrefixTokens at which a re-send read cache. Below it the
//     prefix is too short for the provider to cache. Unmeasured → hypothesis.
//   - r  is ReadCostEquiv / CachedTokens, averaged over the samples that itemize the
//     cached-read bill. Unmeasured → hypothesis.
//
// Samples whose CachedTokens is zero (a miss) are NOT fit inputs for T/M_min (they are
// the negative observations that bound the constants from above) but they are recorded
// by the caller; the fit uses only confirmed-warm samples, matching "verify-then-trust"
// (Rule A1: warmth is confirmed only by a real read).
func FitCalibration(samples []ProbeSample, hypo Hypothesis) Calibration {
	c := Calibration{
		TTLMillis:       hypo.TTLMillis,
		MinPrefixTokens: hypo.MinPrefixTokens,
		ReadMult:        hypo.ReadMult,
	}
	if len(samples) > 0 {
		c.Provider, c.ModelID, c.Endpoint = samples[0].Provider, samples[0].ModelID, samples[0].Endpoint
	}
	var maxHitDelay int64 = -1
	var minHitPrefix int64 = -1
	var rSum float64
	var rN int
	for _, s := range samples {
		if s.CachedTokens <= 0 {
			continue // a miss: bounds the constant from above, not a fit input.
		}
		if maxHitDelay < 0 || s.DelayMillis > maxHitDelay {
			maxHitDelay = s.DelayMillis
		}
		if minHitPrefix < 0 || s.PrefixTokens < minHitPrefix {
			minHitPrefix = s.PrefixTokens
		}
		if s.ReadCostEquiv > 0 {
			rSum += s.ReadCostEquiv / float64(s.CachedTokens)
			rN++
		}
	}
	if maxHitDelay >= 0 {
		c.TTLMillis = maxHitDelay
		c.TTLMeasured = true
	}
	if minHitPrefix >= 0 {
		c.MinPrefixTokens = minHitPrefix
		c.MinPrefixMeasured = true
	}
	if rN > 0 {
		c.ReadMult = rSum / float64(rN)
		c.ReadMultMeasured = true
	}
	return c
}

// ProbeBudget charges probe traffic against an LRU budget so MEASURING warmth does not
// evict what you believe warm — the §11.4 Rule-D2 "observer-perturbs-state" guard. The
// provider cache is LRU; a probe request could displace a warm entry. vCache marks the
// slots it believes warm as PROTECTED and refuses a probe that would evict one, forcing
// the harness to wait or pick another slot. Calibration must never destroy the warm set
// it is measuring.
//
// It is a pure accounting type: the caller drives it (MarkWarm mirrors the estimator's
// belief, AdmitProbe mirrors a real probe request). Capacity models the provider's LRU
// slot budget the harness may perturb.
type ProbeBudget struct {
	Capacity int
	order    []string
	warm     map[string]bool
}

// NewProbeBudget creates a budget of the given LRU slot capacity.
func NewProbeBudget(capacity int) *ProbeBudget {
	if capacity < 0 {
		capacity = 0
	}
	return &ProbeBudget{Capacity: capacity, warm: map[string]bool{}}
}

// MarkWarm protects a slot the estimator believes warm: a probe that would evict it is
// refused. MarkCold releases the protection (the estimator no longer believes it warm).
func (b *ProbeBudget) MarkWarm(slot string) { b.warm[slot] = true }
func (b *ProbeBudget) MarkCold(slot string) { delete(b.warm, slot) }

// IsWarm reports whether a slot is currently protected as believed-warm.
func (b *ProbeBudget) IsWarm(slot string) bool { return b.warm[slot] }

// AdmitProbe records a probe of slot and reports whether it fit the budget WITHOUT
// evicting a believed-warm entry. When the budget is full, the LRU victim is the
// least-recently-used slot; if that victim is warm the probe is REFUSED (the caller
// retries later), otherwise the cold victim is evicted and the probe is admitted. A
// probe that re-touches an already-present slot is always admitted (it refreshes the
// slot's LRU position, like a real cache read).
func (b *ProbeBudget) AdmitProbe(slot string) (admitted bool, evictedCold string) {
	if idx := b.indexOf(slot); idx >= 0 {
		b.touch(idx)
		return true, ""
	}
	if len(b.order) < b.Capacity {
		b.order = append(b.order, slot)
		return true, ""
	}
	if b.Capacity == 0 {
		return false, ""
	}
	victim := b.order[0]
	if b.warm[victim] {
		return false, "" // refuse — measuring must not evict a believed-warm slot
	}
	b.order = append(b.order[1:], slot)
	delete(b.warm, victim)
	return true, victim
}

func (b *ProbeBudget) indexOf(slot string) int {
	for i, s := range b.order {
		if s == slot {
			return i
		}
	}
	return -1
}

// touch moves the slot at idx to the most-recently-used position (the tail).
func (b *ProbeBudget) touch(idx int) {
	slot := b.order[idx]
	next := make([]string, 0, len(b.order))
	next = append(next, b.order[:idx]...)
	next = append(next, b.order[idx+1:]...)
	b.order = append(next, slot)
}
