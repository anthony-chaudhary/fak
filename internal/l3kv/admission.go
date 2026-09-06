package l3kv

import (
	"sync"
)

// AdmissionFilter gates whether a demoted span should be admitted to durable L3
// SSD offload storage based on observed access frequency and metadata (#11039,
// parent #10964).
type AdmissionFilter interface {
	Admit(digest string, byteSize int) bool
	RecordTouch(digest string)
}

// defaultSeeds provides 4 independent 64-bit prime seeds for the 4-way sketch.
var defaultSeeds = [4]uint64{
	0x16f11fe89b0d677c,
	0x64b525f8e5c5e90a,
	0x9e3779b97f4a7c15,
	0xc6bc279692b5c323,
}

// TinyLFUConfig specifies the geometry and policy parameters for TinyLFUFilter.
type TinyLFUConfig struct {
	// Width is the number of counters per row. Clamped to minimum 16 and rounded
	// up to power of 2. Default: 4096.
	Width int

	// WindowCap is the number of touches before the sketch performs aging/decay
	// (halving all counters). Default: Width * 10.
	WindowCap int

	// Threshold is the minimum touch count required for Admit to return true.
	// Default: 2 (the double-touch rule: 1st touch throttled, 2nd touch admitted).
	Threshold int
}

// DefaultTinyLFUConfig returns standard defaults for SSD offload throttling.
func DefaultTinyLFUConfig() TinyLFUConfig {
	return TinyLFUConfig{
		Width:     4096,
		WindowCap: 40960,
		Threshold: 2,
	}
}

// TinyLFUFilter implements a frequency-gated TinyLFU admission filter backed by
// a 4-way Count-Min sketch with 4 independent hash seeds, configurable threshold
// gate, and periodic aging decay (#11039).
type TinyLFUFilter struct {
	mu          sync.Mutex
	rows        [4][]uint8
	width       uint64
	seeds       [4]uint64
	threshold   int
	sampleCount uint64
	windowCap   uint64
}

// NewTinyLFUFilter constructs a validated, thread-safe TinyLFUFilter.
// If cfgs is omitted, DefaultTinyLFUConfig() is used.
func NewTinyLFUFilter(cfgs ...TinyLFUConfig) *TinyLFUFilter {
	cfg := DefaultTinyLFUConfig()
	if len(cfgs) > 0 {
		c := cfgs[0]
		if c.Width > 0 {
			cfg.Width = c.Width
		}
		if c.WindowCap > 0 {
			cfg.WindowCap = c.WindowCap
		}
		if c.Threshold > 0 {
			cfg.Threshold = c.Threshold
		}
	}
	if cfg.Width < 16 {
		cfg.Width = 16
	}
	w := nextPow2(uint64(cfg.Width))
	windowCap := uint64(cfg.WindowCap)
	if windowCap == 0 {
		windowCap = w * 10
	}
	threshold := cfg.Threshold
	if threshold <= 0 {
		threshold = 2
	}

	f := &TinyLFUFilter{
		width:     w,
		seeds:     defaultSeeds,
		threshold: threshold,
		windowCap: windowCap,
	}
	for i := range f.rows {
		f.rows[i] = make([]uint8, w)
	}
	return f
}

// Admit returns true if the digest has reached or exceeded the recurrence threshold
// within the current temporal window. Untouched or single-touch spans return false.
func (f *TinyLFUFilter) Admit(digest string, byteSize int) bool {
	if f == nil || digest == "" || byteSize < 0 {
		return false
	}
	f.mu.Lock()
	defer f.mu.Unlock()

	minCount := uint8(255)
	for i := 0; i < 4; i++ {
		idx := hashDigest(digest, f.seeds[i]) & (f.width - 1)
		c := f.rows[i][idx]
		if c < minCount {
			minCount = c
		}
	}
	return int(minCount) >= f.threshold
}

// RecordTouch increments the frequency counters for the digest across all 4 sketch
// rows. When total samples reach window capacity, periodic aging halves all counters.
func (f *TinyLFUFilter) RecordTouch(digest string) {
	if f == nil || digest == "" {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()

	for i := 0; i < 4; i++ {
		idx := hashDigest(digest, f.seeds[i]) & (f.width - 1)
		if f.rows[i][idx] < 255 {
			f.rows[i][idx]++
		}
	}
	f.sampleCount++
	if f.sampleCount >= f.windowCap {
		f.decayLocked()
	}
}

// Estimate returns the conservative minimum frequency count across all 4 sketch rows.
func (f *TinyLFUFilter) Estimate(digest string) int {
	if f == nil || digest == "" {
		return 0
	}
	f.mu.Lock()
	defer f.mu.Unlock()

	minCount := uint8(255)
	for i := 0; i < 4; i++ {
		idx := hashDigest(digest, f.seeds[i]) & (f.width - 1)
		c := f.rows[i][idx]
		if c < minCount {
			minCount = c
		}
	}
	return int(minCount)
}

// Reset clears all counters and resets sample count.
func (f *TinyLFUFilter) Reset() {
	if f == nil {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	for i := 0; i < 4; i++ {
		for j := range f.rows[i] {
			f.rows[i][j] = 0
		}
	}
	f.sampleCount = 0
}

func (f *TinyLFUFilter) decayLocked() {
	for i := 0; i < 4; i++ {
		for j := range f.rows[i] {
			f.rows[i][j] >>= 1
		}
	}
	f.sampleCount = 0
}

func hashDigest(key string, seed uint64) uint64 {
	const prime64 = 1099511628211
	h := seed ^ 14695981039346656037
	for i := 0; i < len(key); i++ {
		h ^= uint64(key[i])
		h *= prime64
	}
	h ^= h >> 33
	h *= 0xff51afd7ed558ccd
	h ^= h >> 33
	h *= 0xc4ceb9fe1a85ec53
	h ^= h >> 33
	return h
}

func nextPow2(v uint64) uint64 {
	if v <= 1 {
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

// AlwaysAdmitFilter admits all spans unconditionally (no-op filter).
type AlwaysAdmitFilter struct{}

// Admit returns true for any non-empty digest with non-negative size.
func (AlwaysAdmitFilter) Admit(digest string, byteSize int) bool {
	return digest != "" && byteSize >= 0
}

// RecordTouch is a no-op on AlwaysAdmitFilter.
func (AlwaysAdmitFilter) RecordTouch(string) {}

var (
	_ AdmissionFilter = (*TinyLFUFilter)(nil)
	_ AdmissionFilter = AlwaysAdmitFilter{}
)
