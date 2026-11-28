// Ported from github.com/dgraph-io/ristretto (sketch.go, policy.go, z/bbloom.go

package radixkv

import (
	"math"
	"math/rand"
)

// TinyLFUAdmission is an opt-in W-TinyLFU admission filter: a doorkeeper bloom filter in
// front of a 4-bit count-min sketch, with every counter halved (and the doorkeeper
// cleared) after numCounters increments. It is a standalone primitive keyed by caller-
// hashed uint64 keys and is not wired into Tree; the existing admission gate and SLRU
// eviction are unchanged. Not safe for concurrent use.
type TinyLFUAdmission struct {
	freq    *tinyLFUSketch
	door    *tinyLFUBloom
	incrs   int64
	resetAt int64
}

// TinyLFUCandidate is one resident entry the admission loop may displace.
type TinyLFUCandidate struct {
	Key  uint64
	Cost int64
}

// TinyLFUSample is the eviction sample size ristretto's policy compares against.
const TinyLFUSample = 5

// NewTinyLFUAdmission sizes the sketch to numCounters (rounded up to a power of two, at
// least 2 so a counter row is never empty) and the doorkeeper for numCounters entries at
// a 1% false-positive rate. seed replaces upstream's wall-clock seeding so runs replay.
func NewTinyLFUAdmission(numCounters int64, seed int64) *TinyLFUAdmission {
	if numCounters <= 0 {
		panic("radixkv: TinyLFUAdmission needs numCounters > 0")
	}
	return &TinyLFUAdmission{
		freq:    newTinyLFUSketch(numCounters, seed),
		door:    newTinyLFUBloom(float64(numCounters), 0.01),
		resetAt: numCounters,
	}
}

// Increment records one access. The first sighting only sets the doorkeeper, so one-hit
// wonders never occupy sketch counters.
func (a *TinyLFUAdmission) Increment(key uint64) {
	if added := a.door.addIfNotHas(key); !added {
		a.freq.increment(key)
	}
	a.incrs++
	if a.incrs >= a.resetAt {
		a.reset()
	}
}

// Push records a batch of accesses in order.
func (a *TinyLFUAdmission) Push(keys []uint64) {
	for _, key := range keys {
		a.Increment(key)
	}
}

// Estimate returns the sketch minimum plus one when the doorkeeper has seen the key.
func (a *TinyLFUAdmission) Estimate(key uint64) int64 {
	hits := a.freq.estimate(key)
	if a.door.has(key) {
		hits++
	}
	return hits
}

// Reset clears the doorkeeper and halves every counter, as the periodic aging does.
func (a *TinyLFUAdmission) Reset() { a.reset() }

// Clear zeroes all frequency state.
func (a *TinyLFUAdmission) Clear() {
	a.incrs = 0
	a.door.clear()
	a.freq.clear()
}

func (a *TinyLFUAdmission) reset() {
	a.incrs = 0
	a.door.clear()
	a.freq.reset()
}

// Admit decides whether an incoming key of the given cost enters a cache of maxCost
// currently holding used cost. Under pressure it repeatedly samples up to TinyLFUSample
// residents, and evicts the lowest-estimate one while the incoming estimate is at least
// that victim's; it rejects as soon as a sampled minimum out-scores the incoming key.
// Victims evicted before a rejection stay evicted, matching upstream: they are returned
// either way. Residents are sampled in slice order where upstream draws from map order.
func (a *TinyLFUAdmission) Admit(key uint64, cost, used, maxCost int64, residents []TinyLFUCandidate) ([]TinyLFUCandidate, bool) {
	if cost > maxCost {
		return nil, false
	}
	room := maxCost - (used + cost)
	if room >= 0 {
		return nil, true
	}
	incHits := a.Estimate(key)
	sample := make([]TinyLFUCandidate, 0, TinyLFUSample)
	var victims []TinyLFUCandidate
	next := 0
	for room < 0 {
		for len(sample) < TinyLFUSample && next < len(residents) {
			sample = append(sample, residents[next])
			next++
		}
		minHits, minID := int64(math.MaxInt64), 0
		for i, c := range sample {
			if hits := a.Estimate(c.Key); hits < minHits {
				minHits, minID = hits, i
			}
		}
		if incHits < minHits {
			return victims, false
		}
		victim := sample[minID]
		sample[minID] = sample[len(sample)-1]
		sample = sample[:len(sample)-1]
		victims = append(victims, victim)
		used -= victim.Cost
		room = maxCost - (used + cost)
	}
	return victims, true
}

const tinyLFUDepth = 4

// tinyLFUSketch is a count-min sketch of 4-bit saturating counters, two per byte.
type tinyLFUSketch struct {
	rows [tinyLFUDepth][]byte
	seed [tinyLFUDepth]uint64
	mask uint64
}

func newTinyLFUSketch(numCounters, seed int64) *tinyLFUSketch {
	numCounters = tinyLFUNext2Power(max(numCounters, 2))
	s := &tinyLFUSketch{mask: uint64(numCounters - 1)}
	source := rand.New(rand.NewSource(seed))
	for i := range s.rows {
		s.seed[i] = source.Uint64()
		s.rows[i] = make([]byte, numCounters/2)
	}
	return s
}

func (s *tinyLFUSketch) increment(hashed uint64) {
	for i, row := range s.rows {
		n := (hashed ^ s.seed[i]) & s.mask
		shift := (n & 1) * 4
		// Saturate at 15: wrapping would make the hottest key look coldest.
		if (row[n/2]>>shift)&0x0f < 15 {
			row[n/2] += 1 << shift
		}
	}
}

func (s *tinyLFUSketch) estimate(hashed uint64) int64 {
	minVal := byte(255)
	for i, row := range s.rows {
		n := (hashed ^ s.seed[i]) & s.mask
		if v := (row[n/2] >> ((n & 1) * 4)) & 0x0f; v < minVal {
			minVal = v
		}
	}
	return int64(minVal)
}

// reset halves both nibbles at once; the 0x77 mask drops the bit shifted across the
// nibble boundary.
func (s *tinyLFUSketch) reset() {
	for _, row := range s.rows {
		for i := range row {
			row[i] = (row[i] >> 1) & 0x77
		}
	}
}

func (s *tinyLFUSketch) clear() {
	for _, row := range s.rows {
		clear(row)
	}
}

func tinyLFUNext2Power(x int64) int64 {
	x--
	x |= x >> 1
	x |= x >> 2
	x |= x >> 4
	x |= x >> 8
	x |= x >> 16
	x |= x >> 32
	x++
	return x
}

// tinyLFUBloom is the doorkeeper: a power-of-two bitset probed by double hashing over the
// high and low bits of the caller's already-hashed key.
type tinyLFUBloom struct {
	bitset  []uint64
	size    uint64
	setLocs uint64
	shift   uint64
}

func newTinyLFUBloom(numEntries, wrongs float64) *tinyLFUBloom {
	bits := -1 * numEntries * math.Log(wrongs) / math.Pow(0.69314718056, 2)
	locs := uint64(math.Ceil(0.69314718056 * bits / numEntries))
	size, exponent := uint64(1), uint64(0)
	for target := max(uint64(bits), 512); size < target; {
		size <<= 1
		exponent++
	}
	return &tinyLFUBloom{
		bitset:  make([]uint64, size>>6),
		size:    size - 1,
		setLocs: locs,
		shift:   64 - exponent,
	}
}

func (b *tinyLFUBloom) locations(hash uint64) (h, l uint64) {
	return hash >> b.shift, hash << b.shift >> b.shift
}

func (b *tinyLFUBloom) add(hash uint64) {
	h, l := b.locations(hash)
	for i := uint64(0); i < b.setLocs; i++ {
		idx := (h + i*l) & b.size
		b.bitset[idx>>6] |= 1 << (idx & 63)
	}
}

func (b *tinyLFUBloom) has(hash uint64) bool {
	h, l := b.locations(hash)
	for i := uint64(0); i < b.setLocs; i++ {
		idx := (h + i*l) & b.size
		if b.bitset[idx>>6]&(1<<(idx&63)) == 0 {
			return false
		}
	}
	return true
}

func (b *tinyLFUBloom) addIfNotHas(hash uint64) bool {
	if b.has(hash) {
		return false
	}
	b.add(hash)
	return true
}

func (b *tinyLFUBloom) clear() { clear(b.bitset) }
