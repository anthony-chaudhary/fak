package ggufload

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ErrInkling2NoCheckpoint is Restore finding nothing to return to: no checkpoint
// was taken, Reset dropped it, or more tokens ran since it than its horizon.
var ErrInkling2NoCheckpoint = errors.New("inkling2_loop: no checkpoint to restore")

// ErrInkling2CheckpointUnavailable is Checkpoint running out of device memory for
// its side buffers; the session is intact and simply holds no checkpoint.
var ErrInkling2CheckpointUnavailable = errors.New("inkling2_loop: no device memory for a checkpoint")

func (s *inkling2HostSession) Pos() int { return s.h.Pos() }

// Checkpoint saves the whole host state, so any horizon is covered.
func (s *inkling2HostSession) Checkpoint(int) error {
	st := s.h.Save()
	s.ck = &st
	return nil
}

func (s *inkling2HostSession) Restore() error {
	if s.ck == nil {
		return ErrInkling2NoCheckpoint
	}
	s.h.Load(*s.ck)
	return nil
}

func (s *inkling2HostSession) Reset() error {
	s.h.Reset()
	s.ck = nil
	return nil
}

// ink2Rewind is a resident session's checkpoint. Global slots never wrap, so their
// rows below pos survive later tokens untouched. A local ring keeps its last window
// too until a later token lands on it, so only the window rows that horizon more
// tokens would overwrite need a copy; a batched ring's slack often makes that none.
type ink2Rewind struct {
	valid   bool
	pos     int
	horizon int
	// saved local positions are [lo, hi)
	lo, hi int

	hist     []ink2Ref
	zero     ink2Ref
	histFill ink2Ref // -0.0 fill: x + -0.0 == x bit for bit, so an add is an exact copy
	base     []compute.Tensor

	rows    int       // per-slot capacity of k and v, in cache rows
	k, v    []ink2Ref // per slot, set for local slots only
	rowFill ink2Ref   // -0.0 fill covering a row copy
	grown   []compute.Tensor
}

// ink2Span is one contiguous run of ring rows: ring row index, offset into the
// position-ordered side buffer, and row count.
type ink2Span struct{ ring, off, n int }

// inkling2RingSpans splits positions [lo, hi) of a ring of cap rows into
// contiguous ring runs.
func inkling2RingSpans(lo, hi, cap int) []ink2Span {
	var out []ink2Span
	for p := lo; p < hi; {
		r := p % cap
		n := min(hi-p, cap-r)
		out = append(out, ink2Span{ring: r, off: p - lo, n: n})
		p += n
	}
	return out
}

// inkling2SavedSpan is the window positions that horizon more tokens past pos
// overwrite in a ring of cap rows: those tokens land on the rows of
// [pos-cap, pos-cap+horizon).
func inkling2SavedSpan(pos, window, cap, horizon int) (lo, hi int) {
	lo = max(0, pos-window)
	hi = min(pos, pos-cap+horizon)
	return lo, max(lo, hi)
}
