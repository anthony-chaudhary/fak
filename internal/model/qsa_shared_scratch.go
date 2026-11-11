package model

import (
	"errors"
	"fmt"
)

// -----------------------------------------------------------------------------
// QSA shared top-k scratch (shared-scratch contract)
// -----------------------------------------------------------------------------
//
// A QSA (sparse-attention indexer) layer writes its top-k position selection
// into an int32 workspace whose per-token width is the sum of the two indexer
// budgets: the top-k indexer budget plus the compressed-latent ratio slots.
// Before the upstream fix, every QSA layer of a model allocated its own copy
// of that workspace although the copies are never live simultaneously within
// one step, and the MTP drafter kept yet another private copy.
//
// This component lands the shared-scratch contract on the fak model side: one
// hoisted buffer, sized by the batched-token budget, owned once and handed out
// as aliasing views to every QSA layer and to the drafter. It is allocation
// and view contract only: no kernel, no wiring into a forward pass yet, no
// hardware claim.
//
// Provenance (ADAPT): the contract is adapted from the vLLM project's Qwen4Exp
// QSA shared top-k buffer change, PR #65619, at pin
// 9a56d4fec6b0a30a0a989f1cd06cd3c2f7325619 — see THIRD_PARTY_NOTICES.md. No
// upstream code is copied; the buffer-ownership facts are re-expressed as a Go
// type with fail-closed sizing.

// ErrQSASharedScratchDimension is the sentinel returned when a shared-scratch
// dimension is unusable: maxBatchedTokens and indexerBudget must be positive
// and compressRatio must not be negative. Callers match it with errors.Is; the
// wrapped error carries the offending values.
var ErrQSASharedScratchDimension = errors.New("model: qsa shared scratch dimension is invalid")

// ErrQSASharedScratchRow is the sentinel returned when a token row index falls
// outside the scratch's row range. Callers match it with errors.Is; the
// wrapped error carries the offending index.
var ErrQSASharedScratchRow = errors.New("model: qsa shared scratch row out of range")

// QSASharedScratch is the one hoisted top-k selection buffer shared by every
// QSA layer of a model and by its MTP drafter. The layers' uses never overlap
// within a step, so the single buffer is handed out as aliasing views instead
// of one private copy per user; the view API makes that aliasing the closed
// contract a test can pin.
type QSASharedScratch struct {
	buf   []int32 // the single backing array, rows x width
	rows  int     // max batched tokens
	width int     // per-row elements: indexer budget + compress ratio
}

// NewQSASharedScratch allocates the shared top-k selection buffer for
// maxBatchedTokens, where each token row holds indexerBudget top-k indices
// plus compressRatio compressed-latent slots. It fails closed on unusable
// dimensions so a zero-sized or negative budget cannot become the thing every
// layer derives its workspace from.
func NewQSASharedScratch(maxBatchedTokens, indexerBudget, compressRatio int) (*QSASharedScratch, error) {
	if maxBatchedTokens <= 0 || indexerBudget <= 0 || compressRatio < 0 {
		return nil, fmt.Errorf("%w: max_batched_tokens=%d indexer_budget=%d compress_ratio=%d",
			ErrQSASharedScratchDimension, maxBatchedTokens, indexerBudget, compressRatio)
	}
	width := indexerBudget + compressRatio
	return &QSASharedScratch{
		buf:   make([]int32, maxBatchedTokens*width),
		rows:  maxBatchedTokens,
		width: width,
	}, nil
}

// Rows reports the number of batched-token rows the scratch covers.
func (s *QSASharedScratch) Rows() int { return s.rows }

// Width reports the per-row element count: the sum of the two per-token
// budgets (indexer top-k budget + compress ratio).
func (s *QSASharedScratch) Width() int { return s.width }

// Len reports the total element count of the shared buffer.
func (s *QSASharedScratch) Len() int { return len(s.buf) }

// LayerView hands QSA decoder layer l its top-k selection workspace for one
// step. Every layer receives the SAME backing array: the layer index names
// the caller, it does not partition storage, because the layers' uses never
// overlap within a step. A negative layer index is a programming error and
// gets no workspace.
func (s *QSASharedScratch) LayerView(l int) []int32 {
	if l < 0 {
		return nil
	}
	return s.buf
}

// DrafterView hands the MTP drafter the very same buffer the layers use,
// replacing the drafter's former private copy with the shared scratch.
func (s *QSASharedScratch) DrafterView() []int32 { return s.buf }

// Row returns token i's row of the shared buffer — the per-token slot holding
// the summed budgets (top-k indices followed by compress-ratio slots). It
// fails closed on an out-of-range token so an oversized batch cannot walk
// past the hoisted allocation.
func (s *QSASharedScratch) Row(i int) ([]int32, error) {
	if i < 0 || i >= s.rows {
		return nil, fmt.Errorf("%w: row=%d rows=%d", ErrQSASharedScratchRow, i, s.rows)
	}
	return s.buf[i*s.width : (i+1)*s.width], nil
}
