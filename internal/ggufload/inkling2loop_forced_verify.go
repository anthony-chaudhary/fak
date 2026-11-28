package ggufload

import (
	"context"
	"fmt"
)

// Inkling2LoopForcedVerifySession is an optional interface implemented by
// sessions that can commit a grammar-forced token span [prevToken, forced...]
// in a single batched forward pass instead of serial Step calls.
type Inkling2LoopForcedVerifySession interface {
	CommitForcedSpan(ctx context.Context, prevToken int, forced []int) (next int, batched bool, err error)
}

var _ Inkling2LoopForcedVerifySession = (*inkling2ResidentSession)(nil)

// CommitForcedSpan commits prevToken followed by forced into the session KV
// cache, returning the predicted next token after the span. When batched
// prefill is active and forced is non-empty, it runs one batched Prefill pass
// over [prevToken, forced...] and notifies any attached specDrafter via
// SpecObserve.
func (r *inkling2ResidentSession) CommitForcedSpan(ctx context.Context, prevToken int, forced []int) (int, bool, error) {
	if r == nil || r.s == nil || len(r.s.owned) == 0 {
		return 0, false, fmt.Errorf("inkling2_loop: closed session")
	}
	if len(forced) == 0 {
		next, err := r.Step(prevToken)
		if err != nil {
			return 0, false, err
		}
		if err := r.SpecObserve([]int{prevToken}, 1); err != nil {
			return 0, false, err
		}
		return next, false, nil
	}
	batch := make([]int, 1+len(forced))
	batch[0] = prevToken
	copy(batch[1:], forced)
	batched := r.s.batched
	next, err := r.Prefill(ctx, batch)
	if err != nil {
		return 0, false, err
	}
	if err := r.SpecObserve(batch, len(batch)); err != nil {
		return 0, batched, err
	}
	return next, batched, nil
}
