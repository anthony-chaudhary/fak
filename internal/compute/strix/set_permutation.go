// Package strix implements MALL (Memory Access at Local Level) cache geometry modeling,
// prime-modulo set-index permutation, and 64-token block layout packing for AMD Strix Halo
// (Ryzen AI Max+ 395 / GFX1151).
package strix

import (
	"math"
	"sync"
)

// MALL physical architecture constants for AMD Strix Halo (GFX1151).
const (
	// MALLCacheLineBytes is the 64-byte line size matching Zen 5 / RDNA 3.5 cache lines.
	MALLCacheLineBytes = 64

	// MALLTotalCacheLines is the total number of lines in MALL (32,768 * 16 = 524,288 lines).
	MALLTotalCacheLines = MALLSets * MALLWays

	// Bit slicing constants for physical address mapping:
	// - Line byte offset: bits [5:0] (6 bits, 64 bytes)
	// - Set index: bits [20:6] (15 bits, 32,768 sets)
	// - Tag: bits [47:21] (27 bits)
	MALLOffsetBits    = 6
	MALLOffsetMask    = 0x3F
	MALLSetIndexShift = 6
	MALLSetIndexBits  = 15
	MALLSetIndexMask  = 0x7FFF
	MALLTagShift      = 21
	MALLTagMask       = 0x7FFFFFF

	// Prime & coprime constants for non-aliasing set distribution:
	// PrimeStrideP is the largest prime less than 65536.
	// gcd(65521, 32768) = 1, ensuring a bijection over 2^15 sets.
	PrimeStrideP uint32 = 65521

	// CoprimeTagMultiplier is coprime to 32768 (gcd(4097, 32768) = 1).
	// Swizzles different address tags across distinct set index sequences.
	CoprimeTagMultiplier uint32 = 4097
)

// PermutationArm identifies the address-to-set index mapping strategy.
type PermutationArm string

const (
	// ArmPrimeModulo applies prime-modulo permutation (stride P = 65,521)
	// combined with coprime tag swizzling and 64-token block packing.
	ArmPrimeModulo PermutationArm = "PRIME_MODULO"

	// ArmXORFolding ablates prime permutation, applying XOR bit-folding
	// between lower set bits [20:6] and upper tag bits [35:21].
	ArmXORFolding PermutationArm = "XOR_FOLDING"

	// ArmPowerOfTwo represents the unmitigated baseline: standard power-of-two
	// address extraction using raw bits [20:6].
	ArmPowerOfTwo PermutationArm = "POWER_OF_TWO"
)

// GCD computes the greatest common divisor of two integers using Euclid's algorithm.
func GCD(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	if a < 0 {
		return -a
	}
	return a
}

// DecomposeAddress unpacks a virtual or physical address into its MALL components:
// set index (bits [20:6]), tag (bits [47:21]), and byte offset (bits [5:0]).
func DecomposeAddress(addr uintptr) (setIndex uint32, tag uint32, offset uint32) {
	offset = uint32(addr & MALLOffsetMask)
	setIndex = uint32((addr >> MALLSetIndexShift) & MALLSetIndexMask)
	tag = uint32((addr >> MALLTagShift) & MALLTagMask)
	return setIndex, tag, offset
}

// SetPermutationEngine manages the precomputed prime-modulo permutation tables
// and executes O(1) set-index address translation.
type SetPermutationEngine struct {
	permTable [MALLSets]uint32
}

// NewSetPermutationEngine constructs and initializes a SetPermutationEngine.
// Precomputes the prime-stride permutation table: table[s] = (s * PrimeStrideP) % MALLSets.
func NewSetPermutationEngine() *SetPermutationEngine {
	e := &SetPermutationEngine{}
	for s := uint32(0); s < MALLSets; s++ {
		e.permTable[s] = uint32((uint64(s) * uint64(PrimeStrideP)) % MALLSets)
	}
	return e
}

// PermuteSet computes the permuted set index for a given raw set and tag:
// (table[rawSet & 0x7FFF] + (tag * CoprimeTagMultiplier)) & 0x7FFF.
// For any fixed tag, this operation is a proven mathematical bijection on [0, 32767].
func (e *SetPermutationEngine) PermuteSet(rawSet uint32, tag uint32) uint32 {
	basePerm := e.permTable[rawSet&MALLSetIndexMask]
	tagShift := (tag * CoprimeTagMultiplier) & MALLSetIndexMask
	return (basePerm + tagShift) & MALLSetIndexMask
}

// AddressToSetIndex translates an address to its target MALL set index according to the specified arm.
func (e *SetPermutationEngine) AddressToSetIndex(addr uintptr, arm PermutationArm) uint32 {
	switch arm {
	case ArmPrimeModulo:
		rawSet, tag, _ := DecomposeAddress(addr)
		return e.PermuteSet(rawSet, tag)
	case ArmXORFolding:
		rawSet := uint32((addr >> MALLSetIndexShift) & MALLSetIndexMask)
		folding := uint32((addr >> MALLTagShift) & MALLSetIndexMask)
		return (rawSet ^ folding) & MALLSetIndexMask
	case ArmPowerOfTwo:
		return uint32((addr >> MALLSetIndexShift) & MALLSetIndexMask)
	default:
		return uint32((addr >> MALLSetIndexShift) & MALLSetIndexMask)
	}
}

// DecomposeAddress extracts set index, tag, and line byte offset from an address.
func (e *SetPermutationEngine) DecomposeAddress(addr uintptr) (setIndex uint32, tag uint32, offset uint32) {
	return DecomposeAddress(addr)
}

// SetOccupancyHistogram captures the distribution of line occupancy across the 32,768 MALL sets.
type SetOccupancyHistogram struct {
	Bin0Ways      int // Sets with 0 ways occupied
	Bin1To4Ways   int // Sets with 1 to 4 ways occupied
	Bin5To8Ways   int // Sets with 5 to 8 ways occupied
	Bin9To12Ways  int // Sets with 9 to 12 ways occupied
	Bin13To15Ways int // Sets with 13 to 15 ways occupied
	Bin16Ways     int // Sets with exactly 16 ways occupied
	BinOverflow   int // Sets with >= 17 lines contending (conflict thrashing)

	Mean         float64 // Average occupancy per set
	Variance     float64 // Variance across set occupancy
	StdDev       float64 // Standard deviation of set occupancy
	CV           float64 // Coefficient of Variation: StdDev / Mean
	MaxOccupancy int     // Highest occupancy observed in any set
	MinOccupancy int     // Lowest occupancy observed in any set
}

// ComputeSetOccupancyHistogram calculates occupancy bins and statistical moments from set counts.
func ComputeSetOccupancyHistogram(counts []int) SetOccupancyHistogram {
	var h SetOccupancyHistogram
	if len(counts) == 0 {
		return h
	}

	minOcc := math.MaxInt32
	maxOcc := 0
	sum := 0

	for _, c := range counts {
		sum += c
		if c < minOcc {
			minOcc = c
		}
		if c > maxOcc {
			maxOcc = c
		}

		switch {
		case c == 0:
			h.Bin0Ways++
		case c >= 1 && c <= 4:
			h.Bin1To4Ways++
		case c >= 5 && c <= 8:
			h.Bin5To8Ways++
		case c >= 9 && c <= 12:
			h.Bin9To12Ways++
		case c >= 13 && c <= 15:
			h.Bin13To15Ways++
		case c == 16:
			h.Bin16Ways++
		case c >= 17:
			h.BinOverflow++
		}
	}

	n := float64(len(counts))
	h.Mean = float64(sum) / n

	var sumSqDiff float64
	for _, c := range counts {
		diff := float64(c) - h.Mean
		sumSqDiff += diff * diff
	}
	h.Variance = sumSqDiff / n
	h.StdDev = math.Sqrt(h.Variance)
	if h.Mean > 0 {
		h.CV = h.StdDev / h.Mean
	}
	h.MinOccupancy = minOcc
	h.MaxOccupancy = maxOcc
	return h
}

// CacheWay represents a single way inside a 16-way associative cache set.
type CacheWay struct {
	valid      bool
	tag        uint32
	lastAccess uint64
}

// CacheSet represents one 16-way set in MALL.
type CacheSet struct {
	ways [MALLWays]CacheWay
}

// AssociativitySimulator models the 32,768-set x 16-way MALL cache with LRU replacement.
type AssociativitySimulator struct {
	mu             sync.RWMutex
	sets           [MALLSets]CacheSet
	setOccupancy   [MALLSets]int
	lineSeen       [MALLSets]map[uint32]struct{}
	accessCount    uint64
	totalAccesses  uint64
	hits           uint64
	misses         uint64
	conflictMisses uint64
	evictions      uint64
	validLines     int
}

// NewAssociativitySimulator instantiates an empty 32 MiB MALL cache simulator.
func NewAssociativitySimulator() *AssociativitySimulator {
	s := &AssociativitySimulator{}
	s.Reset()
	return s
}

// Reset clears all cache sets, occupancy records, and access statistics.
func (s *AssociativitySimulator) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i := 0; i < MALLSets; i++ {
		for w := 0; w < MALLWays; w++ {
			s.sets[i].ways[w] = CacheWay{}
		}
		s.setOccupancy[i] = 0
		s.lineSeen[i] = nil
	}
	s.accessCount = 0
	s.totalAccesses = 0
	s.hits = 0
	s.misses = 0
	s.conflictMisses = 0
	s.evictions = 0
	s.validLines = 0
}

// Access simulates a memory reference to addr under the selected permutation arm.
// Returns hit=true if the line was present; conflict=true if all 16 ways were full and caused eviction.
func (s *AssociativitySimulator) Access(addr uintptr, arm PermutationArm, engine *SetPermutationEngine) (hit, conflict bool) {
	setIndex := engine.AddressToSetIndex(addr, arm)
	_, tag, _ := DecomposeAddress(addr)
	return s.RecordLine(setIndex, tag)
}

// RecordLine tracks an access to setIndex with tag, updating LRU state and conflict counters.
func (s *AssociativitySimulator) RecordLine(setIndex uint32, tag uint32) (hit, conflict bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.totalAccesses++
	s.accessCount++

	setIdx := setIndex & MALLSetIndexMask
	set := &s.sets[setIdx]

	// Check for hit
	for w := 0; w < MALLWays; w++ {
		if set.ways[w].valid && set.ways[w].tag == tag {
			s.hits++
			set.ways[w].lastAccess = s.accessCount
			return true, false
		}
	}

	// Line is a miss. Track unique line allocation to this set.
	s.misses++
	if s.lineSeen[setIdx] == nil {
		s.lineSeen[setIdx] = make(map[uint32]struct{})
	}
	if _, seen := s.lineSeen[setIdx][tag]; !seen {
		s.lineSeen[setIdx][tag] = struct{}{}
		s.setOccupancy[setIdx]++
	}

	// Look for an invalid (empty) way
	for w := 0; w < MALLWays; w++ {
		if !set.ways[w].valid {
			set.ways[w].valid = true
			set.ways[w].tag = tag
			set.ways[w].lastAccess = s.accessCount
			s.validLines++
			return false, false
		}
	}

	// All 16 ways occupied: conflict miss and eviction
	s.conflictMisses++
	s.evictions++

	// Select LRU victim
	lruWay := 0
	minAccess := set.ways[0].lastAccess
	for w := 1; w < MALLWays; w++ {
		if set.ways[w].lastAccess < minAccess {
			minAccess = set.ways[w].lastAccess
			lruWay = w
		}
	}

	set.ways[lruWay].tag = tag
	set.ways[lruWay].lastAccess = s.accessCount
	return false, true
}

// EffectiveUtilization returns the percentage (0.0 - 100.0%) of total MALL capacity
// currently populated with valid cache lines.
func (s *AssociativitySimulator) EffectiveUtilization() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return (float64(s.validLines) / float64(MALLTotalCacheLines)) * 100.0
}

// ConflictMissRatio returns the ratio of conflict misses to total accesses (0.0 - 1.0).
func (s *AssociativitySimulator) ConflictMissRatio() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.totalAccesses == 0 {
		return 0.0
	}
	return float64(s.conflictMisses) / float64(s.totalAccesses)
}

// HitRate returns the ratio of cache hits to total accesses (0.0 - 1.0).
func (s *AssociativitySimulator) HitRate() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.totalAccesses == 0 {
		return 0.0
	}
	return float64(s.hits) / float64(s.totalAccesses)
}

// DispersionHistogram returns the occupancy distribution across all 32,768 sets.
func (s *AssociativitySimulator) DispersionHistogram() SetOccupancyHistogram {
	s.mu.RLock()
	defer s.mu.RUnlock()

	counts := make([]int, MALLSets)
	copy(counts, s.setOccupancy[:])
	return ComputeSetOccupancyHistogram(counts)
}

// TotalAccesses returns the total number of simulated memory references.
func (s *AssociativitySimulator) TotalAccesses() uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.totalAccesses
}

// Hits returns the total number of cache hits.
func (s *AssociativitySimulator) Hits() uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.hits
}

// Misses returns the total number of cache misses.
func (s *AssociativitySimulator) Misses() uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.misses
}

// ConflictMisses returns the total number of conflict misses (evictions from full sets).
func (s *AssociativitySimulator) ConflictMisses() uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.conflictMisses
}

// Evictions returns the total number of line evictions.
func (s *AssociativitySimulator) Evictions() uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.evictions
}

// ValidLines returns the count of currently valid lines in the cache.
func (s *AssociativitySimulator) ValidLines() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.validLines
}
