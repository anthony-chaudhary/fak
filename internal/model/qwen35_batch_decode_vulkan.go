//go:build vulkan && (windows || linux) && cgo

package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

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

func init() { qwen35HybridQ4KBatchStep = stepBatchQwen35HybridQ4KVulkan }

func stepBatchQwen35HybridQ4KVulkan(bs *BatchSession, ids []int) ([][]float32, bool) {
	if bs == nil || bs.M == nil || len(ids) < 2 || len(ids) > 8 || len(ids) != len(bs.Seqs) {
		return nil, false
	}
	var be compute.Backend
	lanes := make([]compute.Qwen35SequencePrefillRequest, len(ids))
	finishes := make([]func(), 0, len(ids))
	for i, s := range bs.Seqs {
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
		panic(fmt.Sprintf("model: malformed Qwen Vulkan independent decode result: lanes=%d panels=%d fallback=%d shape=%v", result.Lanes, result.SharedProjectionPanels, result.SerialFallbacks, result.Logits.Shape))
	}
	flat := be.Read(result.Logits)
	be.Free(result.Logits)
	if len(flat) != len(ids)*bs.M.Cfg.VocabSize {
		panic("model: malformed Qwen Vulkan independent decode logits")
	}
	bs.lastStepSharedPanels = result.SharedProjectionPanels
	bs.lastStepMACs = result.ProjectionMACs
	return splitScaledLogits(nil, flat, len(ids), bs.M.Cfg.VocabSize, bs.M.Cfg), true
}
