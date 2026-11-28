// SPDX-License-Identifier: Apache-2.0
// Windowed draft KV group adapted from vLLM (vllm-project/vllm, Apache-2.0),
// vllm/v1/core/kv_cache_planning.py and
// vllm/model_executor/layers/attention/mla_attention.py, revision

package model

import (
	"errors"
	"fmt"
)

var (
	// ErrDraftWindowKVConfig marks a windowed-draft layout that cannot be
	// planned or picked.
	ErrDraftWindowKVConfig = errors.New("model: invalid windowed draft KV configuration")
	// ErrDraftWindowKVBudget marks a step the windowed draft group cannot hold
	// inside its own capacity. The group never borrows from the target's pool.
	ErrDraftWindowKVBudget = errors.New("model: windowed draft KV budget exhausted")
)

// DraftLayerBackend names the attention path one draft layer decodes through.
type DraftLayerBackend string

const (
	// DraftLayerFullContextMLA attends over the whole prefix and keeps its KV
	// in the target's pool class.
	DraftLayerFullContextMLA DraftLayerBackend = "full_context_mla"
	// DraftLayerWindowedMLA attends over a sliding window and keeps its KV in
	// the dedicated windowed draft group.
	DraftLayerWindowedMLA DraftLayerBackend = "windowed_mla"
)

// PickDraftLayerBackends chooses the backend of every draft layer.
// windows[i] is layer i's sliding attention window in tokens, 0 for a layer
// that attends over the full context.
//
// A windowed layer is admitted only under a float KV precision. Under any
// other precision it is refused instead of being routed to the full-context
// path: that fallback would silently give the layer an unbounded KV footprint
// in the wrong pool.
func PickDraftLayerBackends(windows []int, kv KVPrecision) ([]DraftLayerBackend, error) {
	backends := make([]DraftLayerBackend, len(windows))
	for i, w := range windows {
		switch {
		case w < 0:
			return nil, fmt.Errorf("%w: draft layer %d has window %d", ErrDraftWindowKVConfig, i, w)
		case w == 0:
			backends[i] = DraftLayerFullContextMLA
		case kv == KVPrecisionFP32 || kv == KVPrecisionFP16 || kv == KVPrecisionBF16:
			backends[i] = DraftLayerWindowedMLA
		default:
			return nil, fmt.Errorf("%w: draft layer %d is windowed but KV precision %q has no windowed path", ErrDraftWindowKVConfig, i, kv)
		}
	}
	return backends, nil
}

// DraftWindowKVBlocksPerSeq is an upper bound on the blocks one sequence holds
// in the windowed group: the window behind the step, the step's own tokens, and one
// block of slack because the window rarely starts on a block boundary. It does
// not grow with sequence length.
func DraftWindowKVBlocksPerSeq(window, blockTokens, maxStepTokens int) (int, error) {
	if window < 1 || blockTokens < 1 || maxStepTokens < 1 {
		return 0, fmt.Errorf("%w: window=%d block_tokens=%d max_step_tokens=%d", ErrDraftWindowKVConfig, window, blockTokens, maxStepTokens)
	}
	return (window-1+maxStepTokens+blockTokens-1)/blockTokens + 1, nil
}

// DraftWindowKVPlan is a block budget split between the target's pool and the
// windowed draft group. The two shares are disjoint.
type DraftWindowKVPlan struct {
	TargetBlocks      int
	DraftBlocks       int
	DraftBlocksPerSeq int
}

// PlanDraftWindowKV carves the windowed draft group out of totalBlocks, sized
// for draftSeqs concurrent sequences, and leaves the rest to the target.
func PlanDraftWindowKV(totalBlocks, draftSeqs, window, blockTokens, maxStepTokens int) (DraftWindowKVPlan, error) {
	perSeq, err := DraftWindowKVBlocksPerSeq(window, blockTokens, maxStepTokens)
	if err != nil {
		return DraftWindowKVPlan{}, err
	}
	if draftSeqs < 1 {
		return DraftWindowKVPlan{}, fmt.Errorf("%w: draft_seqs=%d", ErrDraftWindowKVConfig, draftSeqs)
	}
	draft := perSeq * draftSeqs
	if draft >= totalBlocks {
		return DraftWindowKVPlan{}, fmt.Errorf("%w: windowed draft group needs %d of %d blocks, leaving none for the target", ErrDraftWindowKVBudget, draft, totalBlocks)
	}
	return DraftWindowKVPlan{TargetBlocks: totalBlocks - draft, DraftBlocks: draft, DraftBlocksPerSeq: perSeq}, nil
}

// DraftWindowKVPool is the windowed draft group: a block pool with its own
// capacity and its own eviction domain. A sequence's blocks are freed only by
// that sequence's window sliding past them or by its release, never to make
// room for another sequence, and nothing outside the pool is ever evicted on
// its behalf.
//
// It is accounting only: it tracks which token ranges are retained, not their
// contents. It is not safe for concurrent use.
type DraftWindowKVPool struct {
	window      int
	blockTokens int
	capacity    int
	used        int
	evicted     int
	seqs        map[int]draftWindowSeq
}

// draftWindowSeq is one sequence's retained range: whole blocks from
// firstBlock up to the block holding token end-1.
type draftWindowSeq struct {
	firstBlock int
	end        int
}

// NewDraftWindowKVPool builds an empty windowed draft group.
func NewDraftWindowKVPool(window, blockTokens, capacityBlocks int) (*DraftWindowKVPool, error) {
	if window < 1 || blockTokens < 1 || capacityBlocks < 1 {
		return nil, fmt.Errorf("%w: window=%d block_tokens=%d capacity_blocks=%d", ErrDraftWindowKVConfig, window, blockTokens, capacityBlocks)
	}
	return &DraftWindowKVPool{
		window:      window,
		blockTokens: blockTokens,
		capacity:    capacityBlocks,
		seqs:        make(map[int]draftWindowSeq),
	}, nil
}

func (s draftWindowSeq) blocks(blockTokens int) int {
	return (s.end+blockTokens-1)/blockTokens - s.firstBlock
}

// Append advances seq by tokens draft positions and returns how many of the
// sequence's blocks the sliding window released. Blocks that lie wholly behind
// the window of the step's first new position are evicted before the step's
// own blocks are counted, so a long sequence stays inside its per-sequence
// bound. A step that does not fit the remaining capacity is refused and leaves
// the pool unchanged.
func (p *DraftWindowKVPool) Append(seq, tokens int) (int, error) {
	if p == nil || p.seqs == nil {
		return 0, fmt.Errorf("%w: uninitialized pool", ErrDraftWindowKVConfig)
	}
	if tokens < 1 {
		return 0, fmt.Errorf("%w: append of %d tokens", ErrDraftWindowKVConfig, tokens)
	}
	old := p.seqs[seq]
	next := draftWindowSeq{firstBlock: old.firstBlock, end: old.end + tokens}
	// The step's first new position attends back to old.end-(window-1).
	if oldest := old.end - (p.window - 1); oldest > 0 {
		if b := oldest / p.blockTokens; b > next.firstBlock {
			next.firstBlock = b
		}
	}
	held := old.blocks(p.blockTokens)
	need := next.blocks(p.blockTokens)
	if p.used-held+need > p.capacity {
		return 0, fmt.Errorf("%w: sequence %d needs %d blocks, %d of %d free", ErrDraftWindowKVBudget, seq, need, p.capacity-p.used+held, p.capacity)
	}
	released := next.firstBlock - old.firstBlock
	p.seqs[seq] = next
	p.used += need - held
	p.evicted += released
	return released, nil
}

// Release frees every block seq holds and returns the count.
func (p *DraftWindowKVPool) Release(seq int) int {
	if p == nil {
		return 0
	}
	s, ok := p.seqs[seq]
	if !ok {
		return 0
	}
	held := s.blocks(p.blockTokens)
	delete(p.seqs, seq)
	p.used -= held
	return held
}

// Retained returns seq's retained token range [start, end). start is
// block-aligned, so it may sit up to one block before the window.
func (p *DraftWindowKVPool) Retained(seq int) (start, end int, ok bool) {
	if p == nil {
		return 0, 0, false
	}
	s, ok := p.seqs[seq]
	if !ok {
		return 0, 0, false
	}
	return s.firstBlock * p.blockTokens, s.end, true
}

// SeqBlocks is the number of blocks seq currently holds.
func (p *DraftWindowKVPool) SeqBlocks(seq int) int {
	if p == nil {
		return 0
	}
	s, ok := p.seqs[seq]
	if !ok {
		return 0
	}
	return s.blocks(p.blockTokens)
}

// Used is the number of blocks held across all sequences.
func (p *DraftWindowKVPool) Used() int {
	if p == nil {
		return 0
	}
	return p.used
}

// Capacity is the group's own block budget.
func (p *DraftWindowKVPool) Capacity() int {
	if p == nil {
		return 0
	}
	return p.capacity
}

// Evicted is the running count of blocks the sliding window has released.
func (p *DraftWindowKVPool) Evicted() int {
	if p == nil {
		return 0
	}
	return p.evicted
}
