// Bloom-filter reuse-POTENTIAL estimator (#3396) — the probabilistic sibling of the
// realized-reuse tap in cacheobs.go. Technique borrowed clean-room from LMCache's
// memory bloom-filter record strategy (borrow id C4-bloom-reuse-estimator): remember
// every content key ever observed in a bounded, tunable-FPR bit set, and report the
// fraction of observations that were PROBABLY seen before. Where Observer measures
// what the kernel actually served from the cached prefix (realized, exact), this
// estimates what a cache COULD have served if every previously-seen chunk were
// resident (potential, probabilistic) — the live counterpart of the static
// kv_nonprefix_reuse_gap witness.
//
// The estimate is one-sided by construction: a bloom filter has NO false negatives,
// so a genuine repeat is never missed, and false positives are bounded by the
// configured target rate. ReusePotential therefore over-reports by at most ~the FPR
// and never under-reports — an honest upper-bound estimator.
//
// Off-hot-path friendly: the estimator is a pure data structure — no goroutines, no
// queues, fixed memory after construction, O(k) integer work per Observe. Callers
// keep it off the serve path by feeding it asynchronously (e.g. from their own
// background goroutine over a bounded queue); the mutex makes concurrent feeding
// safe, and a dropped observation only shrinks the sample, never corrupts the bits.
package cacheobs

import (
	"math"
	"sync"
)

// Target-FPR clamp bounds for NewBloomEstimator. A non-positive (or NaN) target is
// clamped up to MinTargetFPR — the strictest rate the sizing supports without an
// absurd allocation (~43 bits/item, k≈30) — and a target at or above 1 is clamped
// down to MaxTargetFPR, the loosest rate at which the filter still discriminates.
// Either way the constructor returns a usable filter instead of a degenerate one.
const (
	MinTargetFPR = 1e-9
	MaxTargetFPR = 0.5
)

// BloomEstimator is a fixed-size bloom filter over 64-bit content keys with
// repeat/total counters on top: Observe both asks "probably seen before?" and
// records the key, so the running ratio of probable repeats IS the estimated
// reuse-potential fraction of the observed stream.
//
// It is a pure data structure (no goroutines) safe for concurrent use; see the
// file comment for the off-hot-path feeding contract. Memory is fixed at
// construction (m bits) and bits are only ever set, never cleared, so the
// no-false-negative property holds for the life of the estimator. The zero value
// is inert (Observe reports false and records nothing); use NewBloomEstimator.
type BloomEstimator struct {
	mu   sync.Mutex
	m    uint64   // filter width in bits (>= 1)
	k    int      // probes per key (>= 1)
	bits []uint64 // m bits packed 64 per word
	// observations / probableRepeats are the two counters ReusePotential is derived
	// from: every recorded key, and the subset whose k bits were already all set.
	observations    uint64
	probableRepeats uint64
}

// NewBloomEstimator sizes a bloom filter for expectedItems distinct keys at
// targetFPR false-positive rate using the standard formulas
//
//	m = ceil(-n*ln(p) / (ln 2)^2)   bits
//	k = round((m/n) * ln 2)         hash probes, clamped to >= 1
//
// expectedItems below 1 is treated as 1 and targetFPR is clamped into
// [MinTargetFPR, MaxTargetFPR], so every call returns a usable filter. The rate
// only holds while at most expectedItems distinct keys have been observed;
// past that the filter degrades toward reporting everything as a repeat
// (over-estimating potential, never under-estimating).
func NewBloomEstimator(expectedItems int, targetFPR float64) *BloomEstimator {
	n := expectedItems
	if n < 1 {
		n = 1
	}
	p := targetFPR
	if !(p > MinTargetFPR) { // NaN and non-positive land here too
		p = MinTargetFPR
	}
	if p > MaxTargetFPR {
		p = MaxTargetFPR
	}
	ln2 := math.Ln2
	mf := math.Ceil(-float64(n) * math.Log(p) / (ln2 * ln2))
	if mf < 1 {
		mf = 1
	}
	m := uint64(mf)
	k := int(math.Round(mf / float64(n) * ln2))
	if k < 1 {
		k = 1
	}
	return &BloomEstimator{
		m:    m,
		k:    k,
		bits: make([]uint64, (m+63)/64),
	}
}

// Observe records one content key and reports whether it was probably seen before:
// the k probe bits are read first — all already set means probablySeen — and then
// set, so a repeat of any recorded key ALWAYS reports true (no false negatives)
// while a fresh key reports true only at the bounded false-positive rate. The k
// probe positions come from double hashing a single 64-bit key (two independent
// integer mixes; Kirsch–Mitzenmacher h1 + i*h2), no external hash dependency.
// Counters saturate like the Observer accumulators instead of wrapping.
func (b *BloomEstimator) Observe(key uint64) (probablySeen bool) {
	if b == nil || b.m == 0 {
		return false // nil or zero-value estimator: inert, never a phantom repeat
	}
	h1 := mix64(key)
	h2 := mix64(key^0x9e3779b97f4a7c15) | 1 // odd stride: never a zero-step probe
	b.mu.Lock()
	defer b.mu.Unlock()
	probablySeen = true
	pos := h1
	for i := 0; i < b.k; i++ {
		idx := pos % b.m
		word, bit := idx/64, uint64(1)<<(idx%64)
		if b.bits[word]&bit == 0 {
			probablySeen = false
			b.bits[word] |= bit
		}
		pos += h2
	}
	b.observations = saturatingAddU64(b.observations, 1)
	if probablySeen {
		b.probableRepeats = saturatingAddU64(b.probableRepeats, 1)
	}
	return probablySeen
}

// ReusePotential returns the estimated reuse-potential fraction of the observed
// stream: probable-repeat observations / total observations. 0 with no observations
// yet (an idle estimator never reports a phantom potential, matching ReuseRatio's
// contract). The value over-reports true reuse by at most ~the configured FPR and
// never under-reports, per the bloom no-false-negative guarantee.
func (b *BloomEstimator) ReusePotential() float64 {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.observations == 0 {
		return 0
	}
	return float64(b.probableRepeats) / float64(b.observations)
}

// mix64 is the 64-bit finalizer mix (MurmurHash3 fmix64): a full-avalanche integer
// mix so both double-hash halves are well distributed from a single key without
// pulling in any hash dependency.
func mix64(x uint64) uint64 {
	x ^= x >> 33
	x *= 0xff51afd7ed558ccd
	x ^= x >> 33
	x *= 0xc4ceb9fe1a85ec53
	x ^= x >> 33
	return x
}
