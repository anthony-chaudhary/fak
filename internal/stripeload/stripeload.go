// Package stripeload fans a single logical read across N byte-identical
// mirrors of the same file, sized by relative bandwidth (#4298).
//
// Why: the model loader streams every weight through an io.ReaderAt seam
// (internal/model/safetensors.go's safetensorsFile), so aggregate-bandwidth
// loading needs no loader changes at all -- just a ReaderAt that splits the
// requested range into contiguous sub-ranges proportional to each source's
// measured bandwidth, reads them concurrently under a bounded worker cap, and
// stitches the result. The same chunk-planner that stripes across 3 local
// NVMes also stripes across {local NVMe, peer-RAM, S3}: bounded-concurrency
// fan-out with different sources. This leaf is the hardware-free core of that
// insight; the sources are mirrors (every source returns identical bytes for
// identical ranges), never shards, which is what makes the plan a pure
// performance decision with no correctness surface beyond bit-identity.
//
// The leaf is stdlib-only and pure (tier 1 / pureRoot): it owns no I/O policy,
// no file discovery, and no bandwidth measurement -- callers hand it opened
// ReaderAts and weights.
package stripeload

import (
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
)

// DefaultMinChunk is the default split floor: reads at or below this many
// bytes are served whole by the fastest source instead of being striped.
// 1 MiB keeps per-source requests large enough that seek/dispatch overhead
// cannot eat the aggregate-bandwidth win.
const DefaultMinChunk = 1 << 20

// Source is one mirror of the same logical file, addressable by absolute
// offset. All sources MUST return byte-identical data for identical
// (off, len) ranges -- mirrors, not shards. BWWeight is a measured or
// estimated RELATIVE bandwidth used to size this source's share of a striped
// read; it must be a finite value > 0 (only ratios between sources matter).
type Source struct {
	R        io.ReaderAt
	BWWeight float64
}

// StripedReaderAt fans a single ReadAt across N mirror sources: it splits the
// requested range into contiguous sub-ranges sized proportionally to each
// source's BWWeight, reads them concurrently under a bounded worker cap, and
// stitches the result in place. Output is byte-identical to reading the whole
// range from any single source. One source (or a read no larger than the
// MinChunk floor) degrades to a direct passthrough -- no goroutines, no
// allocation.
//
// StripedReaderAt is safe for concurrent use by multiple goroutines iff every
// underlying source is (io.ReaderAt requires exactly that of implementations
// operating on a seekable stream, and bytes.Reader / os.File both satisfy it).
type StripedReaderAt struct {
	sources  []Source
	weights  []float64 // sources[i].BWWeight, extracted once for the planner
	fastest  int       // index of the highest-BWWeight source (first wins ties)
	minChunk int64
	maxConc  int
}

var _ io.ReaderAt = (*StripedReaderAt)(nil)

// Option configures a StripedReaderAt at construction time.
type Option func(*StripedReaderAt)

// WithMinChunk sets the split floor: a read (or planned sub-range) of n bytes
// or fewer is not split and goes whole to the fastest source. n must be > 0.
func WithMinChunk(n int64) Option {
	return func(s *StripedReaderAt) { s.minChunk = n }
}

// WithMaxConcurrency caps the number of sub-range reads in flight at once.
// n must be > 0. The default is len(sources) -- every planned sub-range may
// read concurrently.
func WithMaxConcurrency(n int) Option {
	return func(s *StripedReaderAt) { s.maxConc = n }
}

// New builds a StripedReaderAt over the given mirror sources. It errors on an
// empty source list, a nil source reader, a non-finite or non-positive
// BWWeight, or an invalid option value.
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

// subRange is one contiguous slice of a planned read, assigned to sources[src].
type subRange struct {
	off int64 // absolute offset of the sub-range
	n   int64 // length in bytes; always > 0
	src int   // index into the weights/sources the plan was built over
}

// carveRanges partitions [off, off+n) into at most len(weights) contiguous,
// gap-free sub-ranges whose lengths are proportional to the normalized
// weights. The last emitted range absorbs the integer-rounding remainder so
// coverage is exact. A request of n <= minChunk, or a single source, yields
// one sub-range assigned to the highest-weight source (first index wins
// ties). Zero-length sub-ranges are never emitted. n <= 0 yields nil.
//
// carveRanges is a pure function so the striping policy is testable without
// any I/O.
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
			length = left // last range absorbs the rounding remainder
		} else {
			length = int64(float64(n) * (w / total))
			if length > left {
				length = left
			}
		}
		if length == 0 {
			continue // never emit a zero-length sub-range
		}
		ranges = append(ranges, subRange{off: cur, n: length, src: i})
		cur += length
		left -= length
	}
	if left > 0 {
		// Only reachable if the final weight's share rounded to zero after
		// earlier floors; fold the tail into the last emitted range so
		// coverage stays exact and gap-free.
		ranges[len(ranges)-1].n += left
	}
	return ranges
}

// ReadAt implements io.ReaderAt. It fills p from the mirror set and follows
// io.ReaderAt semantics faithfully: a full read returns (len(p), nil); if the
// range runs past a source's end the contiguously-filled prefix length is
// returned with io.EOF; any other sub-read failure returns the filled prefix
// length and the first (offset-order) error, wrapped. It never silently
// returns partial data as success.
func (s *StripedReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("stripeload: negative offset")
	}
	// Degraded paths: one source, or a read at/below the split floor (which
	// includes every empty read, keeping (n, err) bit-identical to the
	// source's own answer). Direct passthrough -- no goroutines, no
	// allocation.
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

	// Stitch in offset order: the returned count is the contiguously-filled
	// prefix, and the first failing sub-range decides the error.
	filled := 0
	for i, sr := range plan {
		r := results[i]
		filled += r.n
		if r.err == nil && int64(r.n) < sr.n {
			// A ReaderAt must justify a short read with a non-nil error;
			// guard against a misbehaving source rather than mask it.
			r.err = io.ErrUnexpectedEOF
		}
		if r.err != nil {
			if errors.Is(r.err, io.EOF) {
				// Match single-reader semantics: bytes past every source's
				// end yield (0, io.EOF); a partially-satisfied range yields
				// the filled prefix with io.EOF.
				return filled, io.EOF
			}
			return filled, fmt.Errorf("stripeload: source %d read [%d,%d): %w", sr.src, sr.off, sr.off+sr.n, r.err)
		}
	}
	return filled, nil
}
