package modelengine

import "errors"

// errNativePrefillBudgetExhausted is the closed contract for a scheduler-step
// prefill token budget that can no longer fund a positive chunk: the request
// must wait for the next step instead of running an unbounded or sub-minimum
// chunk. Callers match it with errors.Is; the prose is not part of the API.
var errNativePrefillBudgetExhausted = errors.New("modelengine: native prefill step token budget exhausted")

// Source: sgl-project/mini-sglang, python/minisgl/scheduler/prefill.py:33-151,
// revision 9a91cfafe754aa85daee49998176275667eb58f2 (MIT):
// https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/prefill.py
// Copyright (c) 2026 sgl-project
//
// Adapted for Fak as idiomatic Go (no upstream code blocks copied): only the
// bounded-budget consumption rule is ported. Upstream, one PrefillAdder.token_budget
// is spent across the prefill requests admitted by a batch and an oversized
// request becomes a resumable ChunkedReq carrying its state into the next batch.
// On the fak side the PREFILLING lane already carries that resume state (prompt
// cursor plus resident KV), so this type owns just the shared per-step spend.
type nativePrefillStepBudget struct {
	step  uint64 // scheduler iteration this budget belongs to; 0 = never armed
	total int    // armed ceiling for the step
	left  int    // tokens not yet spent this step
}

// reset re-arms the budget for a new step. A non-positive total leaves it
// disarmed: take then refuses every draw until the next armed reset.
func (b *nativePrefillStepBudget) reset(step uint64, total int) {
	b.step = step
	if total < 0 {
		total = 0
	}
	b.total = total
	b.left = total
}

// remaining reports the tokens still available in the current step.
func (b *nativePrefillStepBudget) remaining() int { return b.left }

// spent reports how much of the step's ceiling has been consumed so far.
func (b *nativePrefillStepBudget) spent() int { return b.total - b.left }

// take funds one prefill request from the step budget. It consumes
// min(remaining, remain) and reports whether the request was only partially
// funded (resumable), mirroring upstream _add_one_req:
//
//	chunk_size = min(self.token_budget, remain_len)
//	is_chunked = chunk_size < remain_len
//	self.token_budget -= chunk_size
//
// An exhausted budget (or a non-positive remainder) refuses with
// errNativePrefillBudgetExhausted rather than funding a zero-length chunk,
// mirroring upstream try_add_one's "if self.token_budget <= 0: return None".
func (b *nativePrefillStepBudget) take(remain int) (chunk int, resumable bool, err error) {
	if b.left <= 0 || remain <= 0 {
		return 0, false, errNativePrefillBudgetExhausted
	}
	chunk = b.left
	if remain < chunk {
		chunk = remain
	}
	b.left -= chunk
	return chunk, chunk < remain, nil
}
