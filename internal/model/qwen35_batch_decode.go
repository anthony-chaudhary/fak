package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// qwen35HybridQ4KBatchStep is an optional exact native batch override.
// Without an override, dispatch requires the existing backend capability.
var qwen35HybridQ4KBatchStep func(*BatchSession, []int) ([][]float32, bool)

func (bs *BatchSession) qwen35HybridQ4KBatchOK(batch int) bool {
	if bs == nil || bs.M == nil || batch < 2 || batch > 8 ||
		!bs.M.Cfg.IsQwen35Hybrid() || bs.M.Cfg.BlockTopology != PreNorm || bs.M.Cfg.IsMoE() {
		return false
	}
	if qwen35HybridQ4KBatchStep != nil {
		return true
	}
	// Keep platform overrides authoritative and unsupported scalar cohorts on
	// their established path, without materializing a batch request.
	if len(bs.Seqs) == 0 || bs.Seqs[0] == nil {
		return false
	}
	_, ok := qwen35IndependentDecodeBatchFor(bs.Seqs[0].Backend)
	return ok
}

func (bs *BatchSession) stepBatchQwen35HybridQ4K(ids []int) [][]float32 {
	step := qwen35HybridQ4KBatchStep
	if step == nil {
		step = stepBatchQwen35HybridQ4KIndependent
	}
	out, accepted := step(bs, ids)
	if !accepted {
		// A decline occurs before any model state mutation, so scalar replay is safe.
		return bs.stepBatchSerial(ids, "backend_declined")
	}
	return out
}

type qwen35IndependentDecodeBatchBackend interface {
	compute.Qwen35IndependentDecodeBatchBackend
}

func qwen35IndependentDecodeBatchFor(backend any) (qwen35IndependentDecodeBatchBackend, bool) {
	batch, ok := backend.(qwen35IndependentDecodeBatchBackend)
	return batch, ok && batch.Qwen35IndependentDecodeBatchPath() == compute.Qwen35IndependentDecodeBatchPath
}

func failQwen35IndependentDecodeBatchSessions(sessions []*Session, cause error) {
	for _, s := range sessions {
		if s == nil {
			continue
		}
		if s.halFailure == nil {
			s.halFailure = cause
		}
		if s.halClosed {
			continue
		}
		// Retire every participating session even if one backend cleanup panics.
		func() { defer func() { s.halClosed = true; _ = recover() }(); s.closeBackendOwned() }()
	}
}

func stepBatchQwen35HybridQ4KIndependent(bs *BatchSession, ids []int) ([][]float32, bool) {
	if bs == nil || bs.M == nil || len(ids) < 2 || len(ids) > 8 || len(ids) != len(bs.Seqs) {
		return nil, false
	}
	var be compute.Backend
	lanes := make([]compute.Qwen35SequencePrefillRequest, len(ids))
	finishes := make([]func(), 0, len(ids))
	for i, s := range bs.Seqs {
		// Failed cleanup can retain state pointers; they do not reopen a retired lane.
		s.ensureOpenBackendSession()
		if s == nil || s.M != bs.M || s.Backend == nil || s.qwen35HAL == nil || s.halKV == nil {
			return nil, false
		}
		if i == 0 {
			be = s.Backend
		} else if s.Backend != be {
			return nil, false
		}
		lanes[i] = s.qwen35SequencePrefillRequest([]int{ids[i]}, true)
	}
	batch, ok := qwen35IndependentDecodeBatchFor(be)
	if !ok {
		return nil, false
	}
	for i, s := range bs.Seqs {
		finishes = append(finishes, s.beginHALTokenLineageWrite([]int{ids[i]}))
	}
	defer func() {
		for i := len(finishes) - 1; i >= 0; i-- {
			finishes[i]()
		}
	}()
	var result compute.Qwen35IndependentDecodeBatchResult
	var call func(int) error
	call = func(i int) error {
		if i == len(bs.Seqs) {
			var err error
			result, err = batch.Qwen35IndependentDecodeBatch(compute.Qwen35IndependentDecodeBatchRequest{Path: compute.Qwen35IndependentDecodeBatchPath, Lanes: lanes})
			return err
		}
		_, err := bs.Seqs[i].qwen35HAL.mutateSequence(be, func(states []compute.Qwen35SequenceState) (compute.Qwen35SequencePrefillResult, error) {
			lanes[i].States = states
			return compute.Qwen35SequencePrefillResult{}, call(i + 1)
		})
		return err
	}
	if err := call(0); err != nil {
		if compute.Qwen35IndependentDecodeBatchMutated(err) {
			failQwen35IndependentDecodeBatchSessions(bs.Seqs, err)
			panic(err)
		}
		return nil, false
	}
	if result.Lanes != len(ids) || result.SharedProjectionPanels <= 0 || result.ProjectionMACs <= 0 || result.SerialFallbacks != 0 || result.Transfers.ActivationD2HBytes != 0 || result.Logits.Buf() == nil || len(result.Logits.Shape) != 2 || result.Logits.Shape[0] != len(ids) || result.Logits.Shape[1] != bs.M.Cfg.VocabSize {
		cause := fmt.Errorf("model: malformed Qwen Vulkan independent decode result: lanes=%d panels=%d fallback=%d shape=%v", result.Lanes, result.SharedProjectionPanels, result.SerialFallbacks, result.Logits.Shape)
		failQwen35IndependentDecodeBatchSessions(bs.Seqs, cause)
		panic(cause.Error()) // preserve the existing initial string-panic contract
	}
	flat := be.Read(result.Logits)
	be.Free(result.Logits)
	if len(flat) != len(ids)*bs.M.Cfg.VocabSize {
		cause := fmt.Errorf("model: malformed Qwen Vulkan independent decode logits")
		failQwen35IndependentDecodeBatchSessions(bs.Seqs, cause)
		panic(cause.Error()) // retries retain cause through the session failure latch
	}
	bs.lastStepSharedPanels = result.SharedProjectionPanels
	bs.lastStepMACs = result.ProjectionMACs
	return splitScaledLogits(nil, flat, len(ids), bs.M.Cfg.VocabSize, bs.M.Cfg), true
}
