package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41SharedActivationFunc func(layer int, gate, up []float32, limit float32) ([]float32, v41DenseProjectionOutcome, error)

// The shared expert's projections still return host rows. This bounded seam
// uploads both operands and retains one output row; it does not fuse projections
// or establish a device-resident FFN. A positive limit requires the actual
// backend's asymmetric clamp primitive; no device work happens on a decline.
func (s *Session) v41SharedActivationFunc() v41SharedActivationFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	limited, haveLimited := s.Backend.(interface {
		SwiGLUWithLimit(compute.Tensor, compute.Tensor, float32) compute.Tensor
	})
	return func(layer int, gate, up []float32, limit float32) (result []float32, outcome v41DenseProjectionOutcome, cause error) {
		s.ensureOpenBackendSession()
		if !finite32(limit) || (limit > 0 && !haveLimited) {
			return nil, v41ProjectionDeclined, nil
		}
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-shared-activation", Layer: layer, Stage: stage, Cause: err}
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
						result, outcome, cause = nil, v41ProjectionError, closeFailure(err)
						return
					}
				}
				if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
					if original, ok := r.(error); ok {
						err = original
					}
					result, outcome, cause = nil, v41ProjectionError, closeFailure(err)
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
		width := s.M.Cfg.MoEIntermediateSize
		if width <= 0 || len(gate) != width || len(up) != width {
			return nil, v41ProjectionError, closeFailure(errV41ProjectionResult)
		}
		for i := range gate {
			if !finite32(gate[i]) || !finite32(up[i]) {
				return nil, v41ProjectionError, closeFailure(errV41ProjectionResult)
			}
		}
		run := func() ([]float32, error) {
			stage = "gate upload"
			g := s.uploadHostF32([]int{width}, gate, compute.MemoryActivation, "V4.1 shared expert gate")
			defer s.Backend.Free(g)
			stage = "up upload"
			u := s.uploadHostF32([]int{width}, up, compute.MemoryActivation, "V4.1 shared expert up")
			defer s.Backend.Free(u)
			stage = "swiglu"
			var y compute.Tensor
			if limit > 0 {
				stage = "limited swiglu"
				y = limited.SwiGLUWithLimit(g, u, limit)
			} else {
				y = s.Backend.SwiGLU(g, u)
			}
			defer s.Backend.Free(y)
			if y.Buf() == nil || y.Dtype != compute.F32 || len(y.Shape) != 1 || y.Shape[0] != width {
				return nil, errV41ProjectionResult
			}
			stage = "readback"
			values := s.Backend.Read(y)
			if len(values) != width {
				return nil, errV41ProjectionResult
			}
			for _, v := range values {
				if !finite32(v) {
					return nil, errV41ProjectionResult
				}
			}
			return append([]float32(nil), values...), nil
		}
		result, cause = run()
		if cause != nil {
			return nil, v41ProjectionError, closeFailure(cause)
		}
		return result, v41ProjectionHandled, nil
	}
}
