package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41IndexSelectFunc selects one query's index rows from its indexer scores on
// the device: the candidate-block mask (topKBlocks > 0) and the final top-k, with
// the host selectors' tie order and -1 sentinels. A nil callback keeps the host
// selectors; a selected call has no decline arm and no host retry.
type v41IndexSelectFunc func(layer int, scores []float32, topK, topKBlocks, blockSize int) ([]int32, error)

// v41IndexSelectBackend is the model-side view of the device index selection:
// one query's candidate-block mask and final top-k over an F32 [rows] score
// tensor, returned as min(topK,rows) exact-integer row indices with -1 sentinels.
type v41IndexSelectBackend interface {
	V41IndexSelectAvailable() bool
	V41IndexSelect(scores compute.Tensor, rows, topK, topKBlocks, blockSize int) (compute.Tensor, error)
}

// v41IndexSelectMaxRows keeps every published row index an exact binary32 integer.
const v41IndexSelectMaxRows = 1 << 24

var errV41IndexSelectResult = errors.New("model: invalid selected V4.1 index selection result")

// Refs #13668. Uploads one query's scores and reads back min(topK,rows) row
// indices: 4*rows upload bytes and 4*min(topK,rows) readback bytes per call.
// The empty selections (rows == 0 or topK == 0) publish no rows and dispatch
// nothing, exactly as the host selectors return an empty row list.
func (s *Session) v41IndexSelectFunc() v41IndexSelectFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	operation, ok := s.Backend.(v41IndexSelectBackend)
	if !ok || !operation.V41IndexSelectAvailable() {
		return nil
	}
	return func(layer int, scores []float32, topK, topKBlocks, blockSize int) (result []int32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-index-select", Layer: layer, Stage: stage, Cause: err}
			s.halFailure = closed
			s.closeBackendOwned()
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
		rows := len(scores)
		if topK < 0 || topKBlocks < 0 || (topKBlocks > 0 && blockSize <= 0) || rows > v41IndexSelectMaxRows {
			return nil, closeFailure(errV41IndexSelectResult)
		}
		for i, value := range scores {
			if math.IsNaN(float64(value)) {
				return nil, closeFailure(fmt.Errorf("%w: score %d is NaN", errV41IndexSelectResult, i))
			}
		}
		if rows == 0 || topK == 0 {
			return []int32{}, nil
		}
		want := min(topK, rows)
		run := func() (result []int32, cause error) {
			var owned []compute.Tensor
			defer func() {
				// Preserve the selected operation's error or original panic,
				// while still attempting every owned release.
				primaryPanic := recover()
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
				if primaryPanic != nil {
					panic(primaryPanic)
				}
				if cause == nil && cleanupPanic != nil {
					panic(cleanupPanic)
				}
			}()
			stage = "score upload"
			st := s.uploadHostF32([]int{rows}, scores, compute.MemoryActivation, "V4.1 index-select scores")
			owned = append(owned, st)
			stage = "index-select"
			selected, err := operation.V41IndexSelect(st, rows, topK, topKBlocks, blockSize)
			fresh := selected.Buf() != nil && selected.Backend() == s.Backend && selected.Buf() != st.Buf()
			if fresh {
				owned = append(owned, selected)
			}
			if err != nil {
				return nil, err
			}
			if !fresh || selected.Dtype != compute.F32 || selected.Layout != compute.RowMajor || selected.Quant != nil ||
				len(selected.Shape) != 1 || selected.Shape[0] != want {
				return nil, errV41IndexSelectResult
			}
			stage = "selection readback"
			values := s.Backend.Read(selected)
			if len(values) != want {
				return nil, errV41IndexSelectResult
			}
			out := make([]int32, want)
			last := -1
			for i, value := range values {
				row := int(value)
				if float32(row) != value || row < -1 || row >= rows {
					return nil, errV41IndexSelectResult
				}
				// Published rows are position-sorted and distinct.
				if row >= 0 {
					if row <= last {
						return nil, errV41IndexSelectResult
					}
					last = row
				}
				out[i] = int32(row)
			}
			return out, nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
}
