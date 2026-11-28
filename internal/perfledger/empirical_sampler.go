// empirical_sampler.go ports aiperf's empirical distribution sampler to Go.
//
// NOTICE: ported from ai-dynamo/aiperf, src/aiperf/dataset/synthesis/empirical_sampler.py
// distinct values of an observed sample and their frequencies, then draws new
// values with probability proportional to those frequencies. This port does the
// same with exact integer counts and a caller-supplied math/rand/v2 generator,
// so a fixed seed always reproduces the same draws.
package perfledger

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"sort"
)

// ErrEmpiricalSamplerNonFinite reports a NaN or infinite observation, which
// has no place in a sortable, averageable empirical distribution.
var ErrEmpiricalSamplerNonFinite = errors.New("perfledger: empirical sampler observation is not finite")

// EmpiricalSampler draws from the empirical distribution of observed values.
// It is immutable after construction and safe for concurrent use as long as
// each goroutine supplies its own *rand.Rand.
type EmpiricalSampler struct {
	values []float64 // distinct observations, ascending
	cumul  []uint64  // cumul[i] = total count of values[0..i]
}

// EmpiricalSamplerStats summarizes the learned distribution.
type EmpiricalSamplerStats struct {
	Min  float64
	Max  float64
	Mean float64 // frequency-weighted mean of the observations
	// Median is taken over the distinct values, not the raw observations,
	// matching upstream so its stats stay comparable.
	Median    float64
	NumUnique int
}

// NewEmpiricalSampler learns the distribution of data. Empty data yields a
// degenerate distribution that always returns 0, as upstream does.
func NewEmpiricalSampler(data []float64) (*EmpiricalSampler, error) {
	if len(data) == 0 {
		return &EmpiricalSampler{values: []float64{0}, cumul: []uint64{1}}, nil
	}
	sorted := slices.Clone(data)
	for _, v := range sorted {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, ErrEmpiricalSamplerNonFinite
		}
	}
	slices.Sort(sorted)
	s := &EmpiricalSampler{}
	for _, v := range sorted {
		n := len(s.values)
		if n > 0 && s.values[n-1] == v {
			s.cumul[n-1]++
			continue
		}
		var prev uint64
		if n > 0 {
			prev = s.cumul[n-1]
		}
		s.values = append(s.values, v)
		s.cumul = append(s.cumul, prev+1)
	}
	return s, nil
}

// NewEmpiricalSamplerRand returns a PCG-backed generator for seed, the
// explicit seed source callers should pass to Sample and SampleBatch.
func NewEmpiricalSamplerRand(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, esampPCGStream))
}

// esampPCGStream fixes the second PCG word so one seed fully names a stream.
const esampPCGStream = 0x9e3779b97f4a7c15

// Sample draws one value using r.
func (s *EmpiricalSampler) Sample(r *rand.Rand) float64 {
	total := s.cumul[len(s.cumul)-1]
	k := r.Uint64N(total)
	// First bucket whose cumulative count exceeds k: an exact integer
	// inverse-CDF lookup, free of floating-point probability drift.
	i := sort.Search(len(s.cumul), func(i int) bool { return s.cumul[i] > k })
	return s.values[i]
}

// SampleBatch draws size values using r; a non-positive size yields an empty slice.
func (s *EmpiricalSampler) SampleBatch(r *rand.Rand, size int) []float64 {
	if size <= 0 {
		return []float64{}
	}
	out := make([]float64, size)
	for i := range out {
		out[i] = s.Sample(r)
	}
	return out
}

// Stats reports summary statistics of the learned distribution.
func (s *EmpiricalSampler) Stats() EmpiricalSamplerStats {
	n := len(s.values)
	total := float64(s.cumul[n-1])
	var mean float64
	var prev uint64
	for i, v := range s.values {
		mean += v * (float64(s.cumul[i]-prev) / total)
		prev = s.cumul[i]
	}
	median := s.values[n/2]
	if n%2 == 0 {
		median = (s.values[n/2-1] + s.values[n/2]) / 2
	}
	return EmpiricalSamplerStats{
		Min:       s.values[0],
		Max:       s.values[n-1],
		Mean:      mean,
		Median:    median,
		NumUnique: n,
	}
}
