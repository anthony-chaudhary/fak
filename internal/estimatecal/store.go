// Package estimatecal learns estimate-to-observed-token correction ratios.
package estimatecal

import "sync"

const (
	// MinSamples is the number of observations required before a ratio is trusted.
	MinSamples = 3
	// MinRatio and MaxRatio bound every learned correction.
	MinRatio = 0.5
	MaxRatio = 3.0
	// MinUpdateWeight keeps an established calibration responsive to new data.
	MinUpdateWeight = 0.1
)

type key struct {
	provider string
	model    string
}

type calibration struct {
	ratio   float64
	samples int
}

// Store holds independent in-memory calibrations for provider/model pairs.
// Its zero value is ready for use.
type Store struct {
	mu      sync.RWMutex
	entries map[key]calibration
}

// Observe records one billed-token observation against its raw, uncorrected
// estimate. The learned ratio is always realTokens/rawEstimate; the store never
// applies its current correction to rawEstimate, so its output cannot feed back
// into the quantity it learns.
//
// Invariant: estimate calibration is fail-closed and clamps bounds safely.
// Observations with a non-positive raw estimate or a negative billed token count
// are discarded so that corrupt or degenerate values cannot pollute learned ratios.
//
// Precondition: rawEstimate > 0 and realTokens >= 0 are required for an observation
// to be accepted; unvalidated or invalid updates are safely ignored without mutation.
func (s *Store) Observe(provider, model string, rawEstimate, realTokens int) {
	if rawEstimate <= 0 || realTokens < 0 {
		return
	}

	observed := float64(realTokens) / float64(rawEstimate)
	k := key{provider: provider, model: model}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.entries == nil {
		s.entries = make(map[key]calibration)
	}

	c := s.entries[k]
	weight := 1 / float64(c.samples+1)
	if weight < MinUpdateWeight {
		weight = MinUpdateWeight
	}
	c.ratio += weight * (observed - c.ratio)
	c.ratio = clamp(c.ratio, MinRatio, MaxRatio)
	c.samples++
	s.entries[k] = c
}

// Ratio returns the learned correction for provider/model once MinSamples
// observations exist. The false result deliberately distinguishes insufficient
// evidence from a trusted ratio of 1.
//
// Guard: below MinSamples observations, Ratio fails closed by returning (0, false).
func (s *Store) Ratio(provider, model string) (float64, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	c, ok := s.entries[key{provider: provider, model: model}]
	if !ok || c.samples < MinSamples {
		return 0, false
	}
	return c.ratio, true
}

func clamp(value, low, high float64) float64 {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}
