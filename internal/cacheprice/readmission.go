package cacheprice

// readmission.go is the ADMISSION dual of retention.go's GDSF eviction pricing (#5290,
// clean-room borrow of Mooncake's CountMinSketch-gated re-promotion, Apache-2.0, no bytes
// vendored). retention.go prices which resident prefix to EVICT under capacity pressure;
// this file prices whether an evicted-then-re-touched prefix earns RE-admission to the fast
// tier at all. Without the gate, a burst of one-hit-wonder prefixes — each touched exactly
// once after a miss — is copied back on first touch and thrashes out genuinely hot entries
// (classic cache pollution; the TinyLFU admission insight). With it, a candidate must PROVE
// recurring heat via a cheap frequency estimate before the pool pays the fabric toll to
// promote it, and only when the fast tier has headroom and a global in-flight promotion
// budget is not saturated (Mooncake's TryPushPromotionQueue gate order: frequency →
// watermark → budget).
//
// The estimator is a count-min sketch: depth hash rows of width uint8 counters. A touch
// increments one counter per row (saturating at 255 so a hot key can never wrap to cold);
// the estimate is the MINIMUM across rows, so hash collisions can only OVER-count, never
// under-count — a key's estimate is an upper bound on its true touch count between decays.
// Popularity ages out with no timer or clock: once total increments cross width×depth, every
// counter halves (Mooncake's count_min_sketch.h self-decay), so a key that WAS hot last
// epoch must keep earning its estimate. Halving preserves relative order, which is all the
// threshold gate consumes.
//
// Like the rest of cacheprice this is a tier-1 foundation leaf: plain ints, deterministic
// (inline FNV-1a double hashing, no seeds, no clocks), importing nothing. internal/radixkv
// now feeds Touch from the live demand lookup and uses the threshold half of ShouldReadmit
// before its value-aware victim comparison; disaggregated pool endpoints can use the full
// frequency/headroom/in-flight verdict. This stays the one source of truth for "has this
// prefix demonstrated enough recurring demand to challenge fast-tier residency".

// FrequencySketch is a fixed-size count-min sketch over opaque string keys: depth rows ×
// width uint8 counters, self-decaying on saturation. The zero value is unusable; construct
// with NewFrequencySketch. It is NOT goroutine-safe — the pool serializes accesses the same
// way it serializes its resident map.
type FrequencySketch struct {
	width    int
	depth    int
	counters []uint8 // row-major: counters[row*width : (row+1)*width]
	adds     int     // total Touch calls since the last decay; decay fires at width*depth
}

// NewFrequencySketch returns a sketch with the given geometry. Width is the counters per
// row (drives collision rate: overcount probability shrinks as width grows); depth is the
// number of independent rows (drives confidence: the min over more rows is a tighter upper
// bound). Both clamp to at least 1 so every sketch is well defined; width×depth is also the
// self-decay period, so a tiny sketch ages popularity faster — exactly right, since a tiny
// sketch saturates faster too.
func NewFrequencySketch(width, depth int) *FrequencySketch {
	if width < 1 {
		width = 1
	}
	if depth < 1 {
		depth = 1
	}
	return &FrequencySketch{
		width:    width,
		depth:    depth,
		counters: make([]uint8, width*depth),
	}
}

// Cells reports the fixed number of uint8 counters owned by the sketch. It is the
// estimator's complete durable state footprint: Touch never allocates per key, so this
// value stays constant for the sketch's lifetime regardless of workload cardinality.
func (s *FrequencySketch) Cells() int {
	if s == nil {
		return 0
	}
	return len(s.counters)
}

// fnv1a64 is the 64-bit FNV-1a hash of key, inlined so the leaf keeps importing nothing.
// Deterministic across runs and platforms: no random seed, so estimates (and therefore
// admission verdicts) are replayable — a property every other cacheprice decision keeps.
func fnv1a64(key string) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	h := uint64(offset64)
	for i := 0; i < len(key); i++ {
		h ^= uint64(key[i])
		h *= prime64
	}
	return h
}

// rowIndex returns the counter index for key in the given row via Kirsch-Mitzenmacher
// double hashing: index_i = (h1 + i·h2) mod width, with h2 forced odd so successive rows
// never degenerate to the same column stride.
func (s *FrequencySketch) rowIndex(h1, h2 uint64, row int) int {
	return int((h1 + uint64(row)*h2) % uint64(s.width))
}

// splitHash derives the two independent hashes double hashing needs from one FNV pass:
// h1 is the raw hash, h2 a re-mixed variant (xor-fold + odd multiplier), |1 to stay odd.
func splitHash(key string) (h1, h2 uint64) {
	h1 = fnv1a64(key)
	h2 = ((h1 >> 32) ^ h1) * 0x9E3779B97F4A7C15 // golden-ratio remix of the folded halves
	h2 |= 1
	return h1, h2
}

// Touch records one access to key: every row's counter for the key increments, saturating
// at 255 rather than wrapping (a wrap would flip the hottest key to coldest — the one error
// direction an admission gate must never take; saturation only flattens VERY hot keys
// together, and they all clear any sane threshold anyway). When total touches since the
// last decay reach width×depth, every counter halves and the epoch resets: stale popularity
// decays with no timer, so an evicted key cannot coast forever on last week's heat.
func (s *FrequencySketch) Touch(key string) {
	h1, h2 := splitHash(key)
	for row := 0; row < s.depth; row++ {
		idx := row*s.width + s.rowIndex(h1, h2, row)
		if s.counters[idx] < 255 {
			s.counters[idx]++
		}
	}
	s.adds++
	if s.adds >= s.width*s.depth {
		s.decay()
	}
}

// decay halves every counter and resets the epoch — Mooncake's self-decay-on-saturation.
// Integer halving floors, so a count of 1 ages fully out (1→0): a one-hit-wonder's trace
// vanishes after a single epoch, which is precisely the anti-thrash bias.
func (s *FrequencySketch) decay() {
	for i := range s.counters {
		s.counters[i] >>= 1
	}
	s.adds = 0
}

// Estimate returns the sketched access frequency of key: the minimum counter across rows.
// Between decays this never UNDER-counts a key's true touches (each touch incremented every
// row, so every row ≥ the true count; collisions only add) — the estimate is a bounded
// overcount, an upper bound whose error one wider row shrinks. A never-touched key with no
// colliding neighbors estimates 0. Monotone: more touches never lower the estimate within
// an epoch.
func (s *FrequencySketch) Estimate(key string) int {
	h1, h2 := splitHash(key)
	min := 256
	for row := 0; row < s.depth; row++ {
		c := int(s.counters[row*s.width+s.rowIndex(h1, h2, row)])
		if c < min {
			min = c
		}
	}
	return min
}

// ShouldReadmit is the re-admission verdict for an evicted prefix that just got re-touched:
// may the pool copy it back into the fast tier? Three gates, all mandatory, mirroring
// Mooncake's TryPushPromotionQueue order (frequency → headroom → in-flight budget):
//
//   - freq < threshold refuses: the candidate has not proven recurring heat. freq is the
//     caller's FrequencySketch.Estimate for the key; threshold is the pool's re-admission
//     bar (≥ 2 makes one-hit-wonders unadmittable by construction, since a single re-touch
//     estimates 1). A threshold ≤ 0 disables the frequency gate — every freq clears it.
//   - !headroom refuses: the fast tier is above its watermark. Re-admission with no
//     headroom is exactly the displacement that thrashes hot residents, so the gate refuses
//     rather than letting a challenger force an eviction; retention.go's EvictionVictim
//     decides evictions on VALUE DENSITY, never on an admission's say-so.
//   - inflight ≥ inflightCap refuses (for inflightCap ≥ 0): the global promotion budget is
//     saturated — each re-admission is a fabric copy in flight, and an unbounded promotion
//     storm is itself a thrash mode. inflightCap 0 means no budget (refuse all);
//     a NEGATIVE inflightCap means unbounded (the caller opted out of budget gating).
//
// Pure and deterministic: same inputs, same verdict, so a refused promotion is replayable
// from its four numbers.
func ShouldReadmit(freq, threshold int, headroom bool, inflight, inflightCap int) bool {
	if freq < threshold {
		return false
	}
	if !headroom {
		return false
	}
	if inflightCap >= 0 && inflight >= inflightCap {
		return false
	}
	return true
}
