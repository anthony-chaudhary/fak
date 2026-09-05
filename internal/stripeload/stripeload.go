// Package stripeload fans a single logical read across byte-identical
// mirrors of the same file, sized proportionally by relative bandwidth.
package stripeload

import (
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
)

// DefaultMinChunk defines the minimum byte threshold before striping across mirrors.
const DefaultMinChunk = 1 << 20

// Source represents a byte-identical mirror with relative bandwidth weighting.
type Source struct {
	R        io.ReaderAt
	BWWeight float64
}

// StripedReaderAt distributes concurrent ReadAt operations across mirror sources.
type StripedReaderAt struct {
	sources  []Source
	weights  []float64
	fastest  int
	minChunk int64
	maxConc  int
}

var _ io.ReaderAt = (*StripedReaderAt)(nil)

// Option modifies configuration parameters during StripedReaderAt creation.
type Option func(*StripedReaderAt)

// WithMinChunk configures the byte floor below which reads bypass multi-mirror striping.
func WithMinChunk(n int64) Option {
	return func(s *StripedReaderAt) { s.minChunk = n }
}

// WithMaxConcurrency bounds simultaneous in-flight reads across all mirrors.
func WithMaxConcurrency(n int) Option {
	return func(s *StripedReaderAt) { s.maxConc = n }
}

// New validates mirror sources and builds an initialized StripedReaderAt instance.
func New(sources []Source, opts ...Option) (*StripedReaderAt, error) {
	if len(sources) == 0 {
		return nil, errors.New("stripeload: no sources")
	}
	s := &StripedReaderAt{
		sources:  append([]Source(nil), sources...),
		weights:  make([]float64, len(sources)),
		minChunk: DefaultMinChunk,
		maxConc:  len(sources),
	}
	for i, src := range s.sources {
		if src.R == nil {
			return nil, fmt.Errorf("stripeload: source %d has nil reader", i)
		}
		if !(src.BWWeight > 0) || math.IsInf(src.BWWeight, 0) {
			return nil, fmt.Errorf("stripeload: source %d has invalid BWWeight %v (must be finite and > 0)", i, src.BWWeight)
		}
		s.weights[i] = src.BWWeight
		if src.BWWeight > s.weights[s.fastest] {
			s.fastest = i
		}
	}
	for _, opt := range opts {
		opt(s)
	}
	if s.minChunk <= 0 {
		return nil, fmt.Errorf("stripeload: MinChunk must be > 0, got %d", s.minChunk)
	}
	if s.maxConc <= 0 {
		return nil, fmt.Errorf("stripeload: MaxConcurrency must be > 0, got %d", s.maxConc)
	}
	return s, nil
}

type subRange struct {
	off int64
	n   int64
	src int
}

// carveRanges partitions the requested span into proportional mirror allocations.
func carveRanges(off, n int64, weights []float64, minChunk int64) []subRange {
	if n <= 0 || len(weights) == 0 {
		return nil
	}
	if len(weights) == 1 || n <= minChunk {
		fastest := 0
		for i, w := range weights {
			if w > weights[fastest] {
				fastest = i
			}
		}
		return []subRange{{off: off, n: n, src: fastest}}
	}
	var total float64
	for _, w := range weights {
		total += w
	}
	ranges := make([]subRange, 0, len(weights))
	cur := off
	left := n
	for i, w := range weights {
		if left == 0 {
			break
		}
		var length int64
		if i == len(weights)-1 {
			length = left
		} else {
			length = int64(float64(n) * (w / total))
			if length > left {
				length = left
			}
		}
		if length == 0 {
			continue
		}
		ranges = append(ranges, subRange{off: cur, n: length, src: i})
		cur += length
		left -= length
	}
	if left > 0 {
		ranges[len(ranges)-1].n += left
	}
	return ranges
}

// ReadAt dispatches sub-ranges across mirrors or delegates small reads directly.
func (s *StripedReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("stripeload: negative offset")
	}
	if len(s.sources) == 1 || int64(len(p)) <= s.minChunk {
		return s.sources[s.fastest].R.ReadAt(p, off)
	}

	plan := carveRanges(off, int64(len(p)), s.weights, s.minChunk)
	type result struct {
		n   int
		err error
	}
	results := make([]result, len(plan))
	sem := make(chan struct{}, s.maxConc)
	var wg sync.WaitGroup
	for i, sr := range plan {
		wg.Add(1)
		go func(i int, sr subRange) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			buf := p[sr.off-off : sr.off-off+sr.n]
			n, err := s.sources[sr.src].R.ReadAt(buf, sr.off)
			results[i] = result{n: n, err: err}
		}(i, sr)
	}
	wg.Wait()

	filled := 0
	for i, sr := range plan {
		r := results[i]
		filled += r.n
		if r.err == nil && int64(r.n) < sr.n {
			r.err = io.ErrUnexpectedEOF
		}
		if r.err != nil {
			if errors.Is(r.err, io.EOF) {
				return filled, io.EOF
			}
			return filled, fmt.Errorf("stripeload: source %d read [%d,%d): %w", sr.src, sr.off, sr.off+sr.n, r.err)
		}
	}
	return filled, nil
}
