package eviction

// CountMinSketch is a probabilistic frequency counter.
// Uses 4 rows of uint8 counters with independent hash seeds.
type CountMinSketch struct {
	rows    [4][]uint8
	width   uint64
	seeds   [4]uint64
	counter uint64 // total increments for periodic reset
	resetAt uint64 // reset threshold (10x width)
}

// NewCountMinSketch creates a new sketch with the given width.
// Width should be ~10x expected unique items for good accuracy.
func NewCountMinSketch(width uint64) *CountMinSketch {
	if width < 16 {
		width = 16
	}
	// Round to power of 2
	width = nextPow2CM(width)

	s := &CountMinSketch{
		width:   width,
		seeds:   [4]uint64{0x16f11fe89b0d677c, 0x64b525f8e5c5e90a, 0x9e3779b97f4a7c15, 0xc6bc279692b5c323},
		resetAt: width * 10,
	}
	for i := range s.rows {
		s.rows[i] = make([]uint8, width)
	}
	return s
}

// Increment adds 1 to the estimated count for the given hash.
func (s *CountMinSketch) Increment(hash uint64) {
	for i := 0; i < 4; i++ {
		idx := (hash ^ s.seeds[i]) & (s.width - 1)
		if s.rows[i][idx] < 255 {
			s.rows[i][idx]++
		}
	}
	s.counter++
	if s.counter >= s.resetAt {
		s.reset()
	}
}

// Estimate returns the minimum count across all rows (conservative estimate).
func (s *CountMinSketch) Estimate(hash uint64) uint8 {
	min := uint8(255)
	for i := 0; i < 4; i++ {
		idx := (hash ^ s.seeds[i]) & (s.width - 1)
		if s.rows[i][idx] < min {
			min = s.rows[i][idx]
		}
	}
	return min
}

// reset halves all counters (aging/decay) to prevent counter saturation.
func (s *CountMinSketch) reset() {
	for i := 0; i < 4; i++ {
		for j := range s.rows[i] {
			s.rows[i][j] >>= 1
		}
	}
	s.counter = 0
}

func nextPow2CM(v uint64) uint64 {
	if v == 0 {
		return 1
	}
	v--
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v |= v >> 32
	v++
	return v
}
