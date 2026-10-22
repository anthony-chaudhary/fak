package model

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// v41DSparkGreedyRound is a host-reference receipt, not serving admission or a
// batch-scheduler result. Verified includes the anchor; Retained includes only
// the anchor and the contiguous greedy-accepted draft prefix. Logits belong to
// the last retained input, never to a discarded speculative suffix.
type v41DSparkGreedyRound struct {
	Proposal           *v41DSparkProposal
	Verified, Retained int
	Logits             []float32
}

// ExecuteGreedyRound consumes an externally allocated q in [1,6]. The anchor
// must already have been selected by the target. All q target inputs execute,
// including the suffix after the first rejection, before retained publication.
// Confidence is returned only as the real proposal's data, never as acceptance.
// The caller exclusively owns this reference and its Session for the whole call.
// This is exact greedy verification only, not sampled speculative decoding.
func (f *v41DSparkReference) ExecuteGreedyRound(ctx context.Context, anchor, q int) (result *v41DSparkGreedyRound, err error) {
	if q < 1 || q > 6 {
		return nil, fmt.Errorf("%w: verification allocation must be 1..6", errV41DSparkReference)
	}
	proposal, err := f.Propose(ctx, anchor)
	if err != nil {
		return nil, err
	}
	ids := append([]int{anchor}, proposal.Proposed[:q-1]...)
	round, err := f.model.beginV41StepRound(f.state, q)
	if err != nil {
		return nil, err
	}
	normal := false
	// Observe the direct abort after it drains, preserving its primary-panic
	// behavior. An incomplete drain retires this reference, not another lane.
	defer func() {
		if !normal || !round.closed || round.poisoned {
			f.mu.Lock()
			f.invalid = true
			f.mu.Unlock()
			if normal {
				err = errors.Join(err, fmt.Errorf("%w: incomplete target drain", errV41DSparkReference))
				result = nil
			}
		}
	}()
	err = func() error {
		defer round.abort()
		capture, err := newV41DSparkCapture(ctx, f.target, ids, false, round)
		if err != nil {
			return err
		}
		defer capture.close()
		keep := 1
		var preceding, retained []float32
		for i, id := range ids {
			if err := ctx.Err(); err != nil {
				return err
			}
			if i > 0 && keep == i {
				best := 0
				for token := 1; token < len(preceding); token++ {
					if preceding[token] > preceding[best] {
						best = token
					}
				}
				if id == best {
					keep++
				}
			}
			tx, err := round.beginStep(id, nil)
			if err != nil {
				return err
			}
			for tx.nextLayer < tx.layers {
				if err := ctx.Err(); err != nil {
					return err
				}
				if err := tx.advanceLayer(); err != nil {
					return err
				}
			}
			preceding, err = tx.finalize()
			if err != nil {
				return err
			}
			if len(preceding) != f.cfg.VocabSize {
				return fmt.Errorf("%w: target logits width", errV41DSparkReference)
			}
			for _, value := range preceding {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return fmt.Errorf("%w: nonfinite target logits", errV41DSparkReference)
				}
			}
			if i < keep {
				retained = append(retained[:0], preceding...)
			}
		}
		if err := f.FinishCapturedRound(ctx, capture, round, keep); err != nil {
			return err
		}
		result = &v41DSparkGreedyRound{Proposal: proposal, Verified: q, Retained: keep, Logits: retained}
		return nil
	}()
	normal = true
	return result, err
}
