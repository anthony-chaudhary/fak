package model

import (
	"context"
	"iter"
	"slices"
)

func (h *Inkling2LoopHost) project(m Inkling2LoopMatrix, x []float32) []float32 {
	if h.W.Batcher == nil {
		return inkling2MatVec(m, x)
	}
	j := inkling2Job(m, x)
	h.run([]Inkling2LoopMatVecJob{j})
	return j.Dst
}

// Prefill consumes the prompt and returns the final row's logits. All recurrent
// state is published together, so a cancelled prompt can be retried in place.
func (h *Inkling2LoopHost) Prefill(ctx context.Context, tokens []int) ([]float32, error) {
	if len(tokens) == 0 || ctx.Err() != nil {
		return nil, ctx.Err()
	}
	work := *h
	work.keys = cloneInklingSlots(h.keys)
	work.values = cloneInklingSlots(h.values)
	work.positions = cloneInklingSlots(h.positions)
	work.conv = make([][]float32, len(h.conv))
	for slot := range work.conv {
		work.conv[slot] = slices.Clone(h.conv[slot])
	}

	var logits []float32
	panel := h.prefillWidth()
	for start := 0; start < len(tokens); start += panel {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(start+panel, len(tokens))
		if end-start == 1 {
			logits = work.PrefillToken(tokens[start], end == len(tokens))
			continue
		}
		var err error
		logits, err = work.prefillPanel(ctx, tokens[start:end], end == len(tokens))
		if err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	h.keys, h.values, h.positions, h.conv = work.keys, work.values, work.positions, work.conv
	h.Features, h.pos = work.Features, work.pos
	return logits, nil
}

// Cache histories are append-only. Clipping forces the first append to reserve
// new slots while preserving all already computed K/V rows.
func cloneInklingSlots[T any](slots [][]T) [][]T {
	dst := slices.Clone(slots)
	for i := range dst {
		dst[i] = slices.Clip(dst[i])
	}
	return dst
}

func (h *Inkling2LoopHost) prefillWidth() int {
	if h.W.Batcher == nil {
		return 1
	}
	// Cap live host activation panels at one SIMD group's rows. Decoded
	// residual blocks and eight routed expert projections dominate the slab.
	hp := &h.HP
	floats := (16+h.Visits()/hp.AttnResBlock)*hp.Hidden +
		hp.ExpertsUsed*(hp.Hidden+3*hp.ExpertWidth) + 4*hp.Heads*hp.HeadDim
	return max(1, min(32, (64<<20)/(4*max(floats, 1))))
}

type inklingPanelStop struct{}
type inklingPanelYield func([]Inkling2LoopMatVecJob) bool

func (yield inklingPanelYield) MatVecBatch(jobs []Inkling2LoopMatVecJob) {
	if !yield(jobs) {
		panic(inklingPanelStop{})
	}
}

type inklingPromptRow struct {
	next func() ([]Inkling2LoopMatVecJob, bool)
	stop func()
}

func (h *Inkling2LoopHost) prefillPanel(ctx context.Context, tokens []int, final bool) ([]float32, error) {
	var logits []float32
	rows := make([]inklingPromptRow, len(tokens))
	for i, token := range tokens {
		row, weights := *h, *h.W
		row.W = &weights
		row.pos = h.pos + i
		rows[i].next, rows[i].stop = iter.Pull(func(yield func([]Inkling2LoopMatVecJob) bool) {
			defer func() {
				if r := recover(); r != nil && r != (inklingPanelStop{}) {
					panic(r)
				}
			}()
			weights.Batcher = inklingPanelYield(yield)
			result := row.PrefillToken(token, final && i == len(tokens)-1)
			if i == len(tokens)-1 {
				logits = result
				h.Features = row.Features
			}
		})
		defer rows[i].stop()
	}
	// Rows advance synchronously in token order. Each projection boundary
	// falls at the same layer stage for all rows; conv and attention thus
	// observe every causal predecessor and none of their successors.
	var batch []Inkling2LoopMatVecJob
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		batch = batch[:0]
		for _, row := range rows {
			if jobs, ok := row.next(); ok {
				batch = append(batch, jobs...)
			}
		}
		if len(batch) == 0 {
			break
		}
		h.run(batch)
	}
	h.pos += len(tokens)
	return logits, nil
}
