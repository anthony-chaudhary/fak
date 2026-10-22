package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
)

// IDs belong to the caller's requests, not their current slice positions.
// A missing previous ID is a fresh lane. Confidence must be from the previous
// round; neither allocation nor totalQ may depend on the current proposals.
type v41DSparkBatchLane struct {
	ID        string
	Reference *v41DSparkReference
	Anchor    int
}

type v41DSparkPreviousLane struct {
	ID         string
	Confidence [5]float32
}

type v41DSparkBatchLaneReceipt struct {
	ID                  string
	Allocated, Verified int
	Published           bool
	Round               *v41DSparkGreedyRound
}

// The caller owns this receipt, so earlier per-lane publication remains visible
// even on a later error or panic. This is not a batch-atomic commit contract.
// WaveWidths counts entered owned-core calls, including singleton tail waves.
// Verified counts committed target steps even if a later lane faults; these
// remain speculative until Published is true and otherwise drain on exit.
type v41DSparkBatchReceipt struct {
	Lanes      []v41DSparkBatchLaneReceipt
	WaveWidths []int
}

// executeV41DSparkGreedyBatch is a bounded host correctness path, not serving
// admission, sampled verification, GPU qualification or an adaptive Q optimizer.
// The caller exclusively owns all references/Sessions and supplies the same
// unlocked owner used by their other forwards. Allocation precedes ALL current
// proposals. Every allocated input runs even after a greedy rejection; only the
// contiguous accepted prefix is published, independently for each lane.
func executeV41DSparkGreedyBatch(ctx context.Context, owner *sync.Mutex, lanes []v41DSparkBatchLane, previous []v41DSparkPreviousLane, totalQ int, receipt *v41DSparkBatchReceipt) (err error) {
	if ctx == nil || owner == nil || receipt == nil || len(lanes) < 2 || len(lanes) > 4 || len(previous) > 4 {
		return fmt.Errorf("%w: bounded host batch requires 2..4 lanes", errV41DSparkReference)
	}
	*receipt = v41DSparkBatchReceipt{}
	prior := make(map[string]int, len(previous))
	confidence := make([][]float32, len(previous))
	for i, p := range previous {
		if _, exists := prior[p.ID]; exists || p.ID == "" {
			return fmt.Errorf("%w: duplicate or empty previous ID", errV41DSparkReference)
		}
		prior[p.ID] = i
		confidence[i] = append([]float32(nil), p.Confidence[:]...)
	}
	remap := make([]int, len(lanes))
	seen := make(map[string]bool, len(lanes))
	inputs := make([]v41StepCohortLane, len(lanes))
	var m *Model
	for i, lane := range lanes {
		f := lane.Reference
		if lane.ID == "" || seen[lane.ID] || f == nil {
			return fmt.Errorf("%w: duplicate/empty lane ID or nil reference", errV41DSparkReference)
		}
		seen[lane.ID] = true
		f.mu.Lock()
		ready := f.ready(ctx)
		f.mu.Unlock()
		if ready != nil {
			return ready
		}
		if i == 0 {
			m = f.model
		}
		if f.model != m || f.next != len(f.state.history) || f.target.backendExecutionScope != nil {
			return fmt.Errorf("%w: batch target identity or prefix differs", errV41DSparkReference)
		}
		inputs[i] = v41StepCohortLane{session: f.target, id: lane.Anchor, ctx: ctx}
		remap[i] = -1
		if row, ok := prior[lane.ID]; ok {
			remap[i] = row
		}
	}
	allocation, err := AllocateSpeculativeVerificationBudget(confidence, remap, 5, totalQ)
	if err != nil {
		return err
	}
	if !owner.TryLock() {
		return fmt.Errorf("%w: device owner is already held", errV41StepCohortDeclined)
	}
	scope := &backendExecutionScope{owner: owner, live: true}
	defer scope.finishError(&err)
	if err := m.v41StepCohortAdmit(inputs); err != nil {
		return err
	}
	for _, input := range inputs {
		scope.bind(input.session)
	}

	receipt.Lanes = make([]v41DSparkBatchLaneReceipt, len(lanes))
	proposals := make([]*v41DSparkProposal, len(lanes))
	ids := make([][]int, len(lanes))
	// No retained target journal is opened until all proposals succeed.
	for i, lane := range lanes {
		receipt.Lanes[i] = v41DSparkBatchLaneReceipt{ID: lane.ID, Allocated: allocation[i]}
		proposals[i], err = lane.Reference.Propose(ctx, lane.Anchor)
		if err != nil {
			return err
		}
		ids[i] = append([]int{lane.Anchor}, proposals[i].Proposed[:allocation[i]-1]...)
	}
	rounds := make([]*v41StepRound, len(lanes))
	captures := make([]*v41DSparkCapture, len(lanes))
	normal := false
	// Each direct abort defer runs even if another lane's cleanup panics. Its
	// paired observer sees the final drain state and retires only that reference.
	for i, lane := range lanes {
		f := lane.Reference
		rounds[i], err = m.beginV41StepRound(f.state, allocation[i])
		if err != nil {
			normal = true
			return err
		}
		r := rounds[i]
		defer func() {
			if (!normal && !receipt.Lanes[i].Published) || !r.closed || r.poisoned {
				f.mu.Lock()
				f.invalid = true
				f.mu.Unlock()
				if normal {
					err = errors.Join(err, fmt.Errorf("%w: incomplete target drain for %s", errV41DSparkReference, lane.ID))
				}
			}
		}()
		defer r.abort()
		captures[i], err = newV41DSparkCapture(ctx, f.target, ids[i], false, r)
		if err != nil {
			normal = true
			return err
		}
		defer captures[i].close()
	}
	err = func() error {
		keep := make([]int, len(lanes))
		preceding := make([][]float32, len(lanes))
		retained := make([][]float32, len(lanes))
		for i := range keep {
			keep[i] = 1
		}
		for position := 0; position < 6; position++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			var wave []v41StepCohortLane
			var indices []int
			for i, lane := range lanes {
				if position >= allocation[i] {
					continue
				}
				if position > 0 && keep[i] == position {
					best := 0
					for token := 1; token < len(preceding[i]); token++ {
						if preceding[i][token] > preceding[i][best] {
							best = token
						}
					}
					if ids[i][position] == best {
						keep[i]++
					}
				}
				wave = append(wave, v41StepCohortLane{session: lane.Reference.target, id: ids[i][position], ctx: ctx, round: rounds[i]})
				indices = append(indices, i)
			}
			if len(wave) == 0 {
				break
			}
			if err := m.v41StepCohortAdmitSize(wave, 1); err != nil {
				return err
			}
			results := make([]v41StepCohortResult, len(wave))
			coreReceipt := &v41StepCohortReceipt{faultLane: -1}
			waveErr := func() error {
				defer func() {
					receipt.WaveWidths = append(receipt.WaveWidths, len(wave))
					for j, i := range indices {
						if results[j].stats.Committed {
							receipt.Lanes[i].Verified++
						}
					}
				}()
				return m.forwardV41StepCohortOwned(scope, wave, false, results, coreReceipt)
			}()
			if waveErr != nil {
				return waveErr
			}
			for j, i := range indices {
				if results[j].err != nil {
					return results[j].err
				}
				logits := results[j].logits
				if len(logits) != m.Cfg.VocabSize {
					return fmt.Errorf("%w: target logits width", errV41DSparkReference)
				}
				finite := false
				for token, value := range logits {
					if math.IsInf(float64(value), -1) && slices.Contains(m.Cfg.SuppressTokens, token) {
						continue
					}
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						return fmt.Errorf("%w: nonfinite target logits", errV41DSparkReference)
					}
					finite = true
				}
				if !finite {
					return fmt.Errorf("%w: no finite target candidate", errV41DSparkReference)
				}
				preceding[i] = logits
				if position < keep[i] {
					retained[i] = append(retained[i][:0], logits...)
				}
			}
		}
		for i, lane := range lanes {
			if err := lane.Reference.FinishCapturedRound(ctx, captures[i], rounds[i], keep[i]); err != nil {
				return err
			}
			receipt.Lanes[i].Round = &v41DSparkGreedyRound{Proposal: proposals[i], Verified: allocation[i], Retained: keep[i], Logits: retained[i]}
			receipt.Lanes[i].Published = true
		}
		return nil
	}()
	normal = true
	return err
}
