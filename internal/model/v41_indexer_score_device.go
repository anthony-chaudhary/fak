package model

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This session-local seam accepts already projected, normalized, rotated and
// scaled operands. Nil preserves the scalar scorer. Once selected there is no
// decline or host replay, including when downstream selection rejects NaN.
type v41IndexerScoreFunc func(layer int, q, keys, weights []float32, heads, headDim, rows int) ([]float32, error)

// Health remains bound when healthy capability absence selects the host path,
// so a later shared-backend sticky fault cannot masquerade as host eligibility.
type v41IndexerScoreHealthFunc func(layer int) error

// V41IndexerScoreOperationError excludes score-boundary health failures and
// selected score/selection failures from structural-error full-history replay.
type V41IndexerScoreOperationError struct {
	Layer int
	Cause error
}

func (e *V41IndexerScoreOperationError) Error() string {
	return fmt.Sprintf("model: V4.1 indexer score boundary failed at layer %d: %v", e.Layer, e.Cause)
}

func (e *V41IndexerScoreOperationError) Unwrap() error { return e.Cause }

var errV41IndexerScoreResult = errors.New("model: invalid selected V4.1 indexer score result")

func v41IndexerScoreOperationErr(layer int, cause error) error {
	return &V41IndexerScoreOperationError{Layer: layer, Cause: v41StageErr(v41StageIndexer, layer, cause)}
}

// A successful score dispatch can still produce NaN from finite inputs. The
// host selectors retain their rejection policy; close the selected session at
// its public boundary after the normal forward rollback, without replay.
func (s *Session) v41IndexerScoreGuard() {
	if r := recover(); r != nil {
		if err, ok := r.(error); ok && !s.halClosed {
			var selected *V41IndexerScoreOperationError
			if errors.As(err, &selected) {
				closed := &BackendForwardOperationError{
					Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"),
					Path: "v41-indexer-score", Layer: selected.Layer, Stage: "selection", Cause: err,
				}
				s.halFailure = closed
				s.Close()
				panic(v41IndexerScoreOperationErr(selected.Layer, closed))
			}
		}
		panic(r)
	}
}

// Keep the existing publication/selection code on the nil path. The selected
// path changes only the score producer; the existing selectors retain ties,
// newest-block pinning, masks, -1 sentinels, infinity handling and NaN rejection.
func v41IndexRowsWithScore(layer int, q, keys, weights []float32, heads, headDim, rows, topKBlocks, blockSize, topK int, score v41IndexerScoreFunc, health v41IndexerScoreHealthFunc) ([]int32, error) {
	if health != nil {
		if err := health(layer); err != nil {
			return nil, v41IndexerScoreOperationErr(layer, err)
		}
	}
	if score == nil {
		pub, err := NewV41IndexerPublication(layer, q, keys, weights, heads, headDim, rows, topKBlocks, blockSize, topK, 0)
		if err != nil {
			return nil, v41StageErr(v41StageIndexer, layer, err)
		}
		return pub.Rows(), nil
	}
	if layer < 0 || topK < 0 {
		return nil, v41IndexerScoreOperationErr(layer, fmt.Errorf("%w: invalid index publication layer or top-k", ErrV41ForwardStage))
	}
	values, err := score(layer, q, keys, weights, heads, headDim, rows)
	if err == nil && len(values) != rows {
		err = errV41IndexerScoreResult
	}
	if err != nil {
		return nil, v41IndexerScoreOperationErr(layer, err)
	}
	var mask []bool
	if topKBlocks > 0 {
		mask, err = V41SelectCandidateBlocks(values, rows, topKBlocks, blockSize)
		if err != nil {
			return nil, v41IndexerScoreOperationErr(layer, err)
		}
	}
	selected, err := V41SelectIndexRows(values, rows, topK, 0, mask)
	if err != nil {
		return nil, v41IndexerScoreOperationErr(layer, err)
	}
	return selected, nil
}

// Check the original host operands before any upload, even for an empty row
// set. The native ABI uses signed 32-bit element indices and host byte counts.
func v41ValidateIndexerScoreInputs(q, keys, weights []float32, heads, headDim, rows int) error {
	qSize, qOK := checkedMulInt(heads, headDim)
	keySize, keyOK := checkedMulInt(rows, headDim)
	_, qBytesOK := checkedMulInt(qSize, 4)
	_, keyBytesOK := checkedMulInt(keySize, 4)
	_, weightBytesOK := checkedMulInt(heads, 4)
	_, outBytesOK := checkedMulInt(rows, 4)
	if heads <= 0 || headDim <= 0 || rows < 0 || heads > math.MaxInt32 || headDim > math.MaxInt32 || rows > math.MaxInt32 ||
		!qOK || !keyOK || qSize > math.MaxInt32 || keySize > math.MaxInt32 ||
		!qBytesOK || !keyBytesOK || !weightBytesOK || !outBytesOK ||
		len(q) != qSize || len(keys) != keySize || len(weights) != heads {
		return fmt.Errorf("%w: invalid indexer score geometry heads=%d dim=%d rows=%d", ErrV41ForwardStage, heads, headDim, rows)
	}
	for _, operand := range []struct {
		name   string
		values []float32
	}{{"query", q}, {"keys", keys}, {"weights", weights}} {
		for i, value := range operand.values {
			if !finite32(value) {
				return fmt.Errorf("%w: indexer score %s element %d is non-finite", ErrV41ForwardStage, operand.name, i)
			}
		}
	}
	return nil
}

func (s *Session) v41IndexerScoreHealthFunc() v41IndexerScoreHealthFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	operation, ok := s.Backend.(compute.V41IndexerScoreBackend)
	if !ok {
		return nil
	}
	return func(layer int) error {
		s.ensureOpenBackendSession()
		if _, err := operation.V41IndexerScoreAdmission(); err != nil {
			closed := &BackendForwardOperationError{
				Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"),
				Path: "v41-indexer-score", Layer: layer, Stage: "admission", Cause: err,
			}
			s.halFailure = closed
			s.Close()
			return closed
		}
		return nil
	}
}

// Native production capability remains false until separately qualified. Name,
// Class, generic device memory and F32 support alone cannot select this seam.
// Scores are returned as an owned host copy; this is not device-resident top-k
// or whole-model GPU execution. Snapshots never persist the callback.
func (s *Session) v41IndexerScoreFunc() v41IndexerScoreFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	operation, ok := s.Backend.(compute.V41IndexerScoreBackend)
	if !ok {
		return nil
	}
	supported, admissionErr := operation.V41IndexerScoreAdmission()
	if admissionErr == nil && !supported {
		return nil
	}
	return func(layer int, q, keys, weights []float32, heads, headDim, rows int) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{
				Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"),
				Path: "v41-indexer-score", Layer: layer, Stage: stage, Cause: err,
			}
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
				if !s.halClosed {
					err, ok := r.(error)
					if !ok {
						err = fmt.Errorf("unclassified backend panic: %v", r)
					}
					closeFailure(err)
				}
				panic(r)
			}
		}()
		if err := v41ValidateIndexerScoreInputs(q, keys, weights, heads, headDim, rows); err != nil {
			return nil, closeFailure(err)
		}
		stage = "admission"
		if admissionErr != nil {
			return nil, closeFailure(admissionErr)
		}
		if _, err := operation.V41IndexerScoreAdmission(); err != nil {
			return nil, closeFailure(err)
		}
		if rows == 0 {
			return make([]float32, 0), nil
		}
		run := func() ([]float32, error) {
			var owned []compute.Tensor
			defer func() {
				var cleanupPanic any
				for i := len(owned) - 1; i >= 0; i-- {
					func() {
						defer func() {
							if r := recover(); r != nil && cleanupPanic == nil {
								cleanupPanic = r
							}
						}()
						s.Backend.Free(owned[i])
					}()
				}
				if cleanupPanic != nil {
					panic(cleanupPanic)
				}
			}()
			fresh := func(x compute.Tensor) bool {
				if x.Buf() == nil || x.Backend() != s.Backend {
					return false
				}
				for _, previous := range owned {
					if previous.Buf() == x.Buf() {
						return false
					}
				}
				owned = append(owned, x)
				return true
			}
			shapeOK := func(x compute.Tensor, shape []int) bool {
				return x.Dtype == compute.F32 && x.Layout == compute.RowMajor && x.Quant == nil && slices.Equal(x.Shape, shape)
			}
			upload := func(name string, shape []int, values []float32) (compute.Tensor, error) {
				stage = name + " upload"
				x := s.uploadHostF32(shape, values, compute.MemoryActivation, "V4.1 indexer score "+name)
				if !fresh(x) || !shapeOK(x, shape) {
					return compute.Tensor{}, errV41IndexerScoreResult
				}
				return x, nil
			}
			qt, err := upload("query", []int{heads, headDim}, q)
			if err != nil {
				return nil, err
			}
			kt, err := upload("keys", []int{rows, headDim}, keys)
			if err != nil {
				return nil, err
			}
			wt, err := upload("weights", []int{heads}, weights)
			if err != nil {
				return nil, err
			}
			stage = "score"
			out, err := operation.V41IndexerScore(qt, kt, wt, rows, heads, headDim)
			ownedOutput := fresh(out) // release a fresh partial output even on error
			if err != nil {
				return nil, err
			}
			stage = "output validation"
			if !ownedOutput || !shapeOK(out, []int{rows}) {
				return nil, errV41IndexerScoreResult
			}
			stage = "readback"
			// The native score primitive marks this tensor for checked Read.
			values := s.Backend.Read(out)
			if len(values) != rows {
				return nil, errV41IndexerScoreResult
			}
			// Finite inputs may produce +/-Inf or NaN. Do not clamp, rescale,
			// round, or reject here: the unchanged selector owns that policy.
			return append([]float32(nil), values...), nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
}
