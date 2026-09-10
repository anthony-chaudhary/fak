package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// DeviceDraftVerification is one greedy target decision plus its exact live-state
// transaction receipt. Correction is the target token after Accepted.
type DeviceDraftVerification struct {
	Accepted              []int
	Correction            int
	NextLogits            []float32
	TargetLogits          [][]float32
	Receipt               TargetVerificationReceipt
	CommittedPrefixTokens int
}

// VerifyGreedyDeviceDraft verifies a bounded linear draft with one resident
// Qwen3.8 sequence invocation. It implements raw temperature-zero argmax only;
// callers with sampling, penalties, or logit bias must keep ordinary decoding.
func (s *Session) VerifyGreedyDeviceDraft(ctx context.Context, draft []int, boundaryLogits []float32) (out DeviceDraftVerification, err error) {
	setupStarted := time.Now()
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if s == nil || s.M == nil {
		return out, ErrSpeculativeNilTarget
	}
	if len(draft) < 1 || len(draft) > qwen35DeviceVerifyMaxDraft {
		return out, targetVerificationDowngrade("greedy device draft requires 1..4 linear tokens")
	}
	if len(boundaryLogits) != s.M.Cfg.VocabSize {
		return out, targetVerificationDowngrade("greedy device boundary logits width differs from vocabulary")
	}
	for _, value := range boundaryLogits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return out, ErrSpeculativeNonFiniteLogits
		}
	}
	for _, token := range draft {
		if token < 0 || token >= s.M.Cfg.VocabSize {
			return out, targetVerificationDowngrade("greedy device draft token is outside vocabulary")
		}
	}
	correction := argmaxF32(boundaryLogits)
	if draft[0] != correction {
		out = DeviceDraftVerification{
			Correction:            correction,
			NextLogits:            append([]float32(nil), boundaryLogits...),
			CommittedPrefixTokens: sessionCommittedPrefixTokens(s),
			Receipt: TargetVerificationReceipt{
				Schema:         targetVerificationReceiptSchema,
				Engine:         targetVerificationEngine,
				Path:           targetVerificationBoundaryRejectPath,
				DraftTokens:    len(draft),
				RejectedTokens: len(draft),
				Accounting: SpeculativeCostAccounting{
					Setup:            measuredSpeculativeCost(setupStarted),
					KnownMemoryBytes: int64(len(boundaryLogits)) * int64(compute.F32.Bytes()),
				},
			},
		}
		return out, nil
	}
	tx, err := beginQwen35MTPTargetTransaction(s, boundaryLogits)
	if err != nil {
		return out, err
	}
	closed := false
	defer func() {
		if !closed {
			err = errors.Join(err, tx.Abort())
		}
	}()
	rows, err := tx.Verify(draft)
	if err != nil {
		closed = true
		return out, err
	}
	if err := ctx.Err(); err != nil {
		abortErr := tx.Abort()
		closed = true
		return out, errors.Join(err, abortErr)
	}
	argmax := make([]int, len(rows)+1)
	argmax[0] = argmaxF32(boundaryLogits)
	for i := range rows {
		argmax[i+1] = argmaxF32(rows[i])
	}
	accepted, correction, err := TripwireVerify(draft, argmax, boundaryLogits, rows)
	if err != nil {
		abortErr := tx.Abort()
		closed = true
		return out, errors.Join(err, abortErr)
	}
	nextLogits, err := tx.Commit(len(accepted))
	if err != nil {
		closed = true
		return out, err
	}
	closed = true
	out = DeviceDraftVerification{
		Accepted:              accepted,
		Correction:            correction,
		NextLogits:            append([]float32(nil), nextLogits...),
		TargetLogits:          rows,
		Receipt:               tx.VerificationReceipt(),
		CommittedPrefixTokens: sessionCommittedPrefixTokens(s),
	}
	return out, nil
}

func sessionCommittedPrefixTokens(s *Session) int {
	if s == nil {
		return 0
	}
	if s.halKV != nil {
		return s.halKV.Len()
	}
	if s.Cache != nil {
		return s.Cache.Len()
	}
	return 0
}

// SpecDecodeGreedyDevice executes a fresh-session greedy run using any linear
// ProposalGenerator, including prompt n-grams, while the target transaction owns
// device KV and recurrent commit/restore semantics.
func SpecDecodeGreedyDevice(ctx context.Context, target *Session, prompt []int, n, k int, generator ProposalGenerator) (polymodel.SpecDecodeRun, error) {
	if target == nil || target.M == nil || target.Backend == nil {
		return polymodel.SpecDecodeRun{}, ErrSpeculativeNilTarget
	}
	if generator == nil {
		return polymodel.SpecDecodeRun{}, ErrSpeculativeNilGenerator
	}
	if len(prompt) == 0 {
		return polymodel.SpecDecodeRun{}, errors.New("model: device speculative decode requires a non-empty prompt")
	}
	if k < 1 || k > qwen35DeviceVerifyMaxDraft {
		return polymodel.SpecDecodeRun{}, targetVerificationDowngrade("device speculative draft depth requires 1..4 tokens")
	}
	if target.halKV == nil || target.halKV.Len() != 0 {
		return polymodel.SpecDecodeRun{}, errors.New("model: device speculative target must be a fresh session")
	}
	if err := ctx.Err(); err != nil {
		return polymodel.SpecDecodeRun{}, err
	}
	targetLogits := target.Prefill(prompt)
	if len(targetLogits) != target.M.Cfg.VocabSize {
		return polymodel.SpecDecodeRun{}, errors.New("model: device speculative target returned malformed prompt logits")
	}
	advanceTarget := func(committed []int) error {
		resident := target.halKV.Len()
		if resident > len(committed) {
			return errors.New("model: device speculative target advanced beyond committed tokens")
		}
		if _, err := target.VerifyTokenLineage(committed[:resident]); err != nil {
			return fmt.Errorf("model: device speculative target diverged from committed prefix: %w", err)
		}
		for _, token := range committed[resident:] {
			if err := ctx.Err(); err != nil {
				return err
			}
			targetLogits = target.Step(token)
		}
		return nil
	}

	var runtimeErr error
	var pending *qwen35MTPTargetTransaction
	var pendingCommitted int
	propose := func(committed []int) []int {
		if runtimeErr != nil {
			return nil
		}
		if pending != nil {
			targetLogits, runtimeErr = pending.Commit(len(pending.draft))
			pending = nil
			pendingCommitted = 0
		}
		if runtimeErr == nil {
			runtimeErr = advanceTarget(committed)
		}
		if runtimeErr != nil {
			return nil
		}
		roundDraft := min(k, len(prompt)+n-len(committed))
		if roundDraft <= 0 {
			return nil
		}
		proposal, err := generator.Propose(ctx, committed, roundDraft)
		if err != nil {
			runtimeErr = err
			return nil
		}
		if proposal.Tree != nil {
			runtimeErr = errors.New("model: device speculative verification accepts linear proposals only")
			return nil
		}
		if len(proposal.Tokens) > roundDraft {
			proposal.Tokens = proposal.Tokens[:roundDraft]
		}
		return append([]int(nil), proposal.Tokens...)
	}
	verify := func(committed, draft []int) []int {
		if runtimeErr != nil {
			return nil
		}
		if len(draft) == 0 {
			return []int{argmaxF32(targetLogits)}
		}
		if err := advanceTarget(committed); err != nil {
			runtimeErr = err
			return nil
		}
		pending, runtimeErr = beginQwen35MTPTargetTransaction(target, targetLogits)
		if runtimeErr != nil {
			return nil
		}
		pendingCommitted = len(committed)
		rows, err := pending.Verify(draft)
		if err != nil {
			runtimeErr = err
			pending = nil
			pendingCommitted = 0
			return nil
		}
		if err := ctx.Err(); err != nil {
			runtimeErr = errors.Join(err, pending.Abort())
			pending = nil
			pendingCommitted = 0
			return nil
		}
		argmax := make([]int, len(rows)+1)
		argmax[0] = argmaxF32(targetLogits)
		for i := range rows {
			argmax[i+1] = argmaxF32(rows[i])
		}
		return argmax
	}
	run, decodeErr := polymodel.SpecDecode(prompt, propose, verify, polymodel.SpecDecodeConfig{
		MaxNewTokens: n,
		MaxDraft:     k,
		Rollback: func(evict int) {
			if pending == nil || runtimeErr != nil {
				return
			}
			if err := ctx.Err(); err != nil {
				runtimeErr = errors.Join(err, pending.Abort())
				pending = nil
				pendingCommitted = 0
				return
			}
			accepted := len(pending.draft) - evict
			targetLogits, runtimeErr = pending.Commit(accepted)
			pending = nil
			pendingCommitted = 0
		},
	})
	if pending != nil {
		if decodeErr == nil && runtimeErr == nil {
			if err := ctx.Err(); err != nil {
				runtimeErr = errors.Join(err, pending.Abort())
			} else {
				emitted := len(prompt) + len(run.Output) - pendingCommitted
				accepted := min(len(pending.draft), max(0, emitted))
				targetLogits, runtimeErr = pending.Commit(accepted)
			}
		} else {
			runtimeErr = errors.Join(runtimeErr, pending.Abort())
		}
		pending = nil
	}
	if runtimeErr != nil || decodeErr != nil {
		return run, errors.Join(runtimeErr, decodeErr)
	}
	committed := append(append([]int(nil), prompt...), run.Output...)
	if err := advanceTarget(committed); err != nil {
		return run, err
	}
	return run, nil
}
