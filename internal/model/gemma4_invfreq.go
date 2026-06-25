package model

import (
	"math"
	"sync"
)

// gemma4InvKey identifies a Gemma-4 per-layer RoPE inverse-frequency table. Every
// input to the table is pinned for the model's lifetime (immutable per-layer config
// + the loaded rope_freqs tensor), so a key collision means an identical table — the
// same invariance cachedInvFreq relies on (rope.go). The sliding flag and the
// rope_freqs hash matter because non-sliding (global/full-attention) layers divide
// each frequency by the shared rope_freqs factor and sliding layers do not. The skip
// flag folds in the FAK_GEMMA4_NO_ROPEFREQS diagnostic knob so toggling it picks a
// fresh entry rather than serving a stale table.
type gemma4InvKey struct {
	layer        int
	thetaBits    uint64
	ropeDim      int
	half         int
	sliding      bool
	skipFreqs    bool
	ropeFreqsLen int
	ropeFreqHash uint64
}

var gemma4InvCache sync.Map // map[gemma4InvKey][]float64

// gemma4InvFreq returns the per-layer RoPE inverse-frequency table for Gemma 4,
// memoized across forwards. The build is byte-for-byte the prior inline computation
// in gemma4AttnSeq (same float64 ops, same order, same rope_freqs division), so a hit
// yields exactly what a recompute would — only the ~half math.Pow calls per layer per
// forward are avoided. There is no runtime invalidation: the config that produced the
// table cannot change within a session; a new model load is a new key.
func (m *Model) gemma4InvFreq(l, ropeDim int, ropeFreqs []float64) []float64 {
	cfg := m.Cfg
	half := ropeDim / 2
	theta := cfg.ropeThetaForLayer(l)
	sliding := cfg.gemma4LayerIsSliding(l)
	skip := gemma4SkipRopeFreqs()
	useFreqs := !sliding && ropeFreqs != nil && !skip

	key := gemma4InvKey{
		layer:        l,
		thetaBits:    math.Float64bits(theta),
		ropeDim:      ropeDim,
		half:         half,
		sliding:      sliding,
		skipFreqs:    skip,
		ropeFreqsLen: len(ropeFreqs),
	}
	if useFreqs {
		key.ropeFreqHash = gemma4RopeFreqsHash(ropeFreqs)
	}
	if inv, ok := gemma4InvCache.Load(key); ok {
		return inv.([]float64)
	}

	inv := make([]float64, half)
	for j := 0; j < half; j++ {
		inv[j] = 1.0 / math.Pow(theta, float64(2*j)/float64(ropeDim))
	}
	if useFreqs {
		for j := 0; j < half && j < len(ropeFreqs); j++ {
			if ropeFreqs[j] != 0 {
				inv[j] /= ropeFreqs[j]
			}
		}
	}
	actual, _ := gemma4InvCache.LoadOrStore(key, inv)
	return actual.([]float64)
}

// gemma4RopeFreqsHash folds the pinned rope_freqs tensor into the inv-freq cache key
// (FNV-1a over the float64 bits) so two non-sliding layers that share theta+ropeDim
// but differ in their rope_freqs vector never collide on a cached table. The tensor
// is pinned for the session, so the hash is constant for the model's lifetime.
func gemma4RopeFreqsHash(f []float64) uint64 {
	h := uint64(1469598103934665603)
	for _, v := range f {
		b := math.Float64bits(v)
		for i := 0; i < 8; i++ {
			h ^= (b >> (uint(i) * 8)) & 0xff
			h *= 1099511628211
		}
	}
	return h
}
