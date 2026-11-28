package modelengine

// Adapted from deepseek-ai/DeepEP (MIT; Copyright (c) 2025 DeepSeek; see
// THIRD_PARTY_NOTICES.md), csrc/hybrid_ep/executor/executor.cu and
// deep_ep/hybrid_ep_buffer.py at pinned revision
// 7181fd860d182c0b9f006034651e920dab887f86 (HybridEP PR #745).
//
// Clean-room host Go contract and in-memory oracle only; no GPU or CUDA graph
// capture hardware is executed.
//
// During CUDA-graph capture, data-dependent host reads of pinned dispatch counts
// can freeze zero-row allocations into the captured graph. This contract passes
// a caller-supplied static upper bound through capture dispatch, sizes captured
// buffer allocations and copies strictly from that bound with zero host reads,
// and fails closed with an actionable sentinel when no bound is supplied.

import (
	"errors"
	"fmt"
)

var (
	// ErrGraphCaptureMissingTokenBound reports a graph-capture dispatch invoked
	// without a positive caller-supplied static token upper bound. Callers may
	// use errors.Is.
	ErrGraphCaptureMissingTokenBound = errors.New("modelengine: graph capture dispatch requires a positive caller-supplied token upper bound")
	// ErrGraphCaptureBoundExceedsCapacity reports a caller-supplied bound or
	// live replay token count that exceeds the rank/captured static bound.
	// Callers may use errors.Is.
	ErrGraphCaptureBoundExceedsCapacity = errors.New("modelengine: graph capture token count exceeds capacity bound")
	// ErrGraphCaptureInvalidShape reports invalid rank capacity or row width.
	// Callers may use errors.Is.
	ErrGraphCaptureInvalidShape = errors.New("modelengine: graph capture invalid buffer shape")
)

// GraphCaptureDispatchRequest specifies static capacity and bound inputs for a
// capture-safe or eager dispatch plan.
type GraphCaptureDispatchRequest struct {
	Capturing        bool
	MaxTokensPerRank int
	RowBytes         int
	// TokenBound is the caller-supplied static upper bound (DeepEP
	// num_of_valid_tokens). Zero or negative means no bound was supplied.
	TokenBound int
	// HostTokenCountReader simulates a data-dependent host read of a pinned
	// tensor (e.g. active token count). Capture mode MUST NEVER invoke it.
	HostTokenCountReader func() int
}

// GraphCaptureDispatchPlan captures the static allocation and copy sizing derived
// at capture time (or eager sizing outside capture).
type GraphCaptureDispatchPlan struct {
	Captured         bool
	StaticRowBound   int
	AllocatedRows    int
	AllocatedBytes   int64
	CopyRows         int
	CopyBytes        int64
	CaptureHostReads int
	stagedCopyBuffer [][]byte
	replayCount      int
	maxTokensPerRank int
	rowBytes         int
}

// PlanGraphCaptureDispatch validates the dispatch shape and builds a
// statically-sized buffer/copy plan when Capturing is true (refusing missing or
// over-capacity bounds without performing any host read), or an eager plan when
// Capturing is false.
func PlanGraphCaptureDispatch(req GraphCaptureDispatchRequest) (*GraphCaptureDispatchPlan, error) {
	if req.MaxTokensPerRank <= 0 || req.RowBytes <= 0 {
		return nil, fmt.Errorf("%w: maxTokensPerRank=%d rowBytes=%d",
			ErrGraphCaptureInvalidShape, req.MaxTokensPerRank, req.RowBytes)
	}

	if req.Capturing {
		// Never perform a data-dependent host read during graph capture; fail
		// closed if the caller did not supply a static upper bound.
		if req.TokenBound <= 0 {
			return nil, fmt.Errorf("%w (got %d; pass an explicit upper bound <= MaxTokensPerRank=%d)",
				ErrGraphCaptureMissingTokenBound, req.TokenBound, req.MaxTokensPerRank)
		}
		if req.TokenBound > req.MaxTokensPerRank {
			return nil, fmt.Errorf("%w: bound %d > maxTokensPerRank %d",
				ErrGraphCaptureBoundExceedsCapacity, req.TokenBound, req.MaxTokensPerRank)
		}
		buf := make([][]byte, req.TokenBound)
		for i := range buf {
			buf[i] = make([]byte, req.RowBytes)
		}
		bytes := int64(req.TokenBound) * int64(req.RowBytes)
		return &GraphCaptureDispatchPlan{
			Captured:         true,
			StaticRowBound:   req.TokenBound,
			AllocatedRows:    req.TokenBound,
			AllocatedBytes:   bytes,
			CopyRows:         req.TokenBound,
			CopyBytes:        bytes,
			CaptureHostReads: 0,
			stagedCopyBuffer: buf,
			maxTokensPerRank: req.MaxTokensPerRank,
			rowBytes:         req.RowBytes,
		}, nil
	}

	// Non-capture eager mode: use supplied bound if positive, otherwise consult
	// the host reader (or fall back to MaxTokensPerRank if reader is nil).
	rows := req.TokenBound
	hostReads := 0
	if rows <= 0 {
		if req.HostTokenCountReader != nil {
			rows = req.HostTokenCountReader()
			hostReads = 1
		} else {
			rows = req.MaxTokensPerRank
		}
	}
	if rows < 0 {
		return nil, fmt.Errorf("%w: negative token count %d", ErrGraphCaptureInvalidShape, rows)
	}
	if rows > req.MaxTokensPerRank {
		return nil, fmt.Errorf("%w: count %d > maxTokensPerRank %d",
			ErrGraphCaptureBoundExceedsCapacity, rows, req.MaxTokensPerRank)
	}
	buf := make([][]byte, rows)
	for i := range buf {
		buf[i] = make([]byte, req.RowBytes)
	}
	bytes := int64(rows) * int64(req.RowBytes)
	return &GraphCaptureDispatchPlan{
		Captured:         false,
		StaticRowBound:   rows,
		AllocatedRows:    rows,
		AllocatedBytes:   bytes,
		CopyRows:         rows,
		CopyBytes:        bytes,
		CaptureHostReads: hostReads,
		stagedCopyBuffer: buf,
		maxTokensPerRank: req.MaxTokensPerRank,
		rowBytes:         req.RowBytes,
	}, nil
}

// Replay executes an in-memory bounded copy into the pre-allocated static
// capture buffer without reallocating. liveValidTokens must sit in
// [0, StaticRowBound]. Unused tail rows [liveValidTokens, StaticRowBound) are
// zeroed so stale bytes never leak across replays.
func (p *GraphCaptureDispatchPlan) Replay(liveValidTokens int, src [][]byte) ([][]byte, error) {
	if p == nil {
		return nil, fmt.Errorf("%w: nil plan", ErrGraphCaptureInvalidShape)
	}
	if liveValidTokens < 0 || len(src) < liveValidTokens {
		return nil, fmt.Errorf("%w: liveValidTokens=%d srcLen=%d", ErrGraphCaptureInvalidShape, liveValidTokens, len(src))
	}
	if liveValidTokens > p.StaticRowBound {
		return nil, fmt.Errorf("%w: liveValidTokens %d > captured StaticRowBound %d",
			ErrGraphCaptureBoundExceedsCapacity, liveValidTokens, p.StaticRowBound)
	}
	for i := 0; i < liveValidTokens; i++ {
		if len(src[i]) != p.rowBytes {
			return nil, fmt.Errorf("%w: row %d bytes %d != %d", ErrGraphCaptureInvalidShape, i, len(src[i]), p.rowBytes)
		}
		copy(p.stagedCopyBuffer[i], src[i])
	}
	for i := liveValidTokens; i < p.StaticRowBound; i++ {
		clear(p.stagedCopyBuffer[i])
	}
	p.replayCount++
	out := make([][]byte, liveValidTokens)
	for i := 0; i < liveValidTokens; i++ {
		out[i] = append([]byte(nil), p.stagedCopyBuffer[i]...)
	}
	return out, nil
}
