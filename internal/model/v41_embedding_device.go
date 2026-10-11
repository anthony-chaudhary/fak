package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41EmbeddingGatherFunc gathers token-embedding rows on the device and returns
// an owned [len(ids), hidden] panel. A nil callback keeps the host gather; a
// selected call has no decline arm and no host retry.
// v41EmbeddingGatherBackend is the model-side view of the device embedding
// gather: byte-copied rows of a device F32
// [vocab,hidden] table, in id order, into a fresh caller-owned panel.
type v41EmbeddingGatherBackend interface {
	V41EmbeddingGatherAvailable() bool
	V41EmbeddingGather(table compute.Tensor, ids []int, hidden int) (compute.Tensor, error)
}

type v41EmbeddingGatherFunc func(ids []int) ([]float32, error)

var errV41EmbeddingGatherResult = errors.New("model: invalid selected V4.1 embedding gather result")

const v41EmbeddingTableName = "model.embed_tokens.weight"

// Refs #13668. The F32 embedding table is staged once as an immutable device
// weight (model-owned cache, shared by forked sessions); each call gathers the
// requested rows by device copies and reads the panel back, 4*len(ids)*hidden
// bytes. Packed (Q2_K/Q4_K/PQ2/Q6_K) tables and a non-unit embedding scale have
// no device gather yet, so they leave the callback nil: host gather normally,
// a typed refusal under device-only.
func (s *Session) v41EmbeddingGatherFunc() v41EmbeddingGatherFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) ||
		!s.M.has(v41EmbeddingTableName) || s.M.Cfg.embedScale() != 1 {
		return nil
	}
	operation, ok := s.Backend.(v41EmbeddingGatherBackend)
	if !ok || !operation.V41EmbeddingGatherAvailable() {
		return nil
	}
	return func(ids []int) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-embedding-gather", Layer: -1, Stage: stage, Cause: err}
			s.halFailure = closed
			s.Close()
			return closed
		}
		defer func() {
			if r := recover(); r != nil {
				if err, ok := r.(error); ok {
					var closed *BackendForwardOperationError
					if errors.As(err, &closed) {
						panic(r)
					}
					var backend *compute.BackendError
					if errors.As(err, &backend) {
						result, cause = nil, closeFailure(err)
						return
					}
				}
				if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
					if original, ok := r.(error); ok {
						err = original
					}
					result, cause = nil, closeFailure(err)
					return
				}
				err, ok := r.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
				panic(r)
			}
		}()
		cfg := s.M.Cfg
		hidden, vocab := cfg.HiddenSize, cfg.VocabSize
		size, valid := checkedMulInt(vocab, hidden)
		width, widthOK := checkedMulInt(len(ids), hidden)
		table := s.M.tensor(v41EmbeddingTableName)
		if !valid || !widthOK || hidden <= 0 || vocab <= 0 || len(ids) == 0 || len(table) < size {
			return nil, closeFailure(errV41EmbeddingGatherResult)
		}
		for _, id := range ids {
			if id < 0 || id >= vocab {
				return nil, closeFailure(errV41EmbeddingGatherResult)
			}
		}
		run := func() ([]float32, error) {
			stage = "table upload"
			key := fmt.Sprintf("v41-embedding:%s:%d:%d", v41EmbeddingTableName, vocab, hidden)
			weight := s.cachedImmutableWeight(key, key, func() compute.Tensor {
				return s.Backend.Upload(compute.NewF32(compute.Default(), []int{vocab, hidden}, table[:size]), compute.F32)
			})
			stage = "embedding-gather"
			panel, err := operation.V41EmbeddingGather(weight, ids, hidden)
			fresh := panel.Buf() != nil && panel.Backend() == s.Backend && panel.Buf() != weight.Buf()
			if fresh {
				defer s.Backend.Free(panel)
			}
			if err != nil {
				return nil, err
			}
			if !fresh || panel.Dtype != compute.F32 || panel.Layout != compute.RowMajor || panel.Quant != nil ||
				len(panel.Shape) != 2 || panel.Shape[0] != len(ids) || panel.Shape[1] != hidden {
				return nil, errV41EmbeddingGatherResult
			}
			stage = "panel readback"
			values := s.Backend.Read(panel)
			if len(values) != width {
				return nil, errV41EmbeddingGatherResult
			}
			return append([]float32(nil), values...), nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
}

// deviceGatherOps is nil-safe so a stateless Model.Forward keeps the host arms.
func (st *v41ForwardState) deviceGatherOps() *v41DeviceGatherOps {
	if st == nil {
		return nil
	}
	return st.gatherOps
}
