package grammar

import "errors"

// ErrEOSBatchTerminal marks a refused batched grammar step: an out-of-range
// row, a step on an already-finished row, or a token outside the row's live
// allowed set. A refusal leaves the whole batch unchanged.
var ErrEOSBatchTerminal = errors.New("grammar: batched EOS terminal refusal")

// RowState is one sequence's per-row grammar state inside a batched
// constrained-generation step. allowed is the live set of token IDs the row
// may consume next; it is replaced wholesale on every accepted step. finished
// is the terminal flag: once set, the row is frozen and masked to the
// terminal-only set.
type RowState struct {
	allowed  []int
	finished bool
}

// contract from vLLM's Guidance structured-output backend under speculative
// decoding, pinned at
// terminate that row only: the batched step applies EOS as a terminal
// transition scoped to that single row, freezes its state, and masks the
// finished row to the terminal-only set while every other live row keeps
// advancing. This mirrors fill_bitmask, which returns the [EOS] mask for a
// stopped matcher. The zero value refuses every step.
type BatchedGrammarStep struct {
	terminal int
	rows     []RowState
}

// NewBatchedGrammarStep builds a batch of n rows, each with its own live
// allowed set. The terminal token must be a member of every row's allowed
// set, because a finished row's mask is exactly {terminal}.
func NewBatchedGrammarStep(terminal int, allowed ...[]int) (*BatchedGrammarStep, error) {
	if len(allowed) == 0 {
		return nil, ErrEOSBatchTerminal
	}
	rows := make([]RowState, len(allowed))
	for i, a := range allowed {
		if len(a) == 0 {
			return nil, ErrEOSBatchTerminal
		}
		if !containsInt(a, terminal) {
			return nil, ErrEOSBatchTerminal
		}
		rows[i] = RowState{allowed: append([]int(nil), a...)}
	}
	return &BatchedGrammarStep{terminal: terminal, rows: rows}, nil
}

// Step applies one token to row k. A token in the row's live allowed set
// advances that row to nextAllowed; the terminal token instead freezes the
// row and marks it finished. Rows j != k are never touched. A refusal
// (out-of-range row, finished row, or token outside the live set) leaves the
// batch unchanged.
func (b *BatchedGrammarStep) Step(k int, token int, nextAllowed []int) error {
	if b == nil || k < 0 || k >= len(b.rows) {
		return ErrEOSBatchTerminal
	}
	r := &b.rows[k]
	if r.finished {
		return ErrEOSBatchTerminal
	}
	if !containsInt(r.allowed, token) {
		return ErrEOSBatchTerminal
	}
	if token == b.terminal {
		r.finished = true
		return nil
	}
	if len(nextAllowed) == 0 {
		return ErrEOSBatchTerminal
	}
	r.allowed = append([]int(nil), nextAllowed...)
	return nil
}

// Mask returns the token set row k may still consume. A finished row is
// masked to exactly {terminal} (the EOS-only mask, faithful to vLLM's
// fill_bitmask for a stopped matcher); a live row returns its live allowed
// set. An out-of-range row is refused.
func (b *BatchedGrammarStep) Mask(k int) ([]int, error) {
	if b == nil || k < 0 || k >= len(b.rows) {
		return nil, ErrEOSBatchTerminal
	}
	r := &b.rows[k]
	if r.finished {
		return []int{b.terminal}, nil
	}
	return append([]int(nil), r.allowed...), nil
}

// Finished reports whether row k has terminated. An out-of-range row is
// refused.
func (b *BatchedGrammarStep) Finished(k int) (bool, error) {
	if b == nil || k < 0 || k >= len(b.rows) {
		return false, ErrEOSBatchTerminal
	}
	return b.rows[k].finished, nil
}

// Live reports whether row k may still advance. An out-of-range row is
// refused.
func (b *BatchedGrammarStep) Live(k int) (bool, error) {
	f, err := b.Finished(k)
	if err != nil {
		return false, err
	}
	return !f, nil
}

func containsInt(set []int, v int) bool {
	for _, x := range set {
		if x == v {
			return true
		}
	}
	return false
}
