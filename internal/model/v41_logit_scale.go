package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41LogitScaleFunc returns an owned host row after one exact scalar device
// multiplication. It never declines after selection. Nil keeps the host path.
type v41LogitScaleFunc func(values []float32, scale float32) ([]float32, error)

var ErrV41LogitScaleOperation = errors.New("model: selected V4.1 logit scale failed")

type V41LogitScaleOperationError struct{ Cause error }

func (e *V41LogitScaleOperationError) Error() string {
	return fmt.Sprintf("%v: %v", ErrV41LogitScaleOperation, e.Cause)
}
func (e *V41LogitScaleOperationError) Unwrap() error { return e.Cause }
func (e *V41LogitScaleOperationError) Is(target error) bool {
	return target == ErrV41LogitScaleOperation
}

var errV41LogitScaleResult = errors.New("model: invalid V4.1 logit scale result")

// Preserve the existing float64-config-to-float32 conversion and ordering:
// multiply, exact existing host softcap, then forced token suppression. The
// generic device-only logit-scale blocker remains: the last two operations and
// the model-only portable Tail path have not been moved to a device primitive.
func v41ScaleHeadLogits(logits []float32, cfg Config, scale v41LogitScaleFunc) ([]float32, error) {
	factor := float32(1)
	if cfg.LogitScale != 0 && cfg.LogitScale != 1 {
		factor = float32(cfg.LogitScale)
	}
	if scale == nil || factor == 1 || !finite32(factor) {
		logitScaleInPlace(logits, cfg)
		return logits, nil
	}
	// Head projection historically permits nonfinite raw results, including
	// infinities that a later softcap maps to finite logits. Such a row is
	// outside this primitive's finite-input domain, before any selection.
	for _, value := range logits {
		if !finite32(value) {
			logitScaleInPlace(logits, cfg)
			return logits, nil
		}
	}
	out, err := scale(logits, factor)
	if err != nil {
		return nil, &V41LogitScaleOperationError{Cause: err}
	}
	if len(out) != len(logits) {
		return nil, &V41LogitScaleOperationError{Cause: errV41LogitScaleResult}
	}
	// Config is passed by value; preserve all other host postprocessing and the
	// original model configuration without multiplying a second time.
	cfg.LogitScale = 1
	logitScaleInPlace(out, cfg)
	return out, nil
}

// This copy seam uploads and reads back 4N bytes each, with 8N live device
// payload across distinct input and output allocations. It is not a residency
// or performance claim. Only a separately linked experimental backend binds it.
func (s *Session) v41LogitScaleFunc() v41LogitScaleFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	operation, ok := s.Backend.(compute.V41LogitScaleBackend)
	if !ok || !operation.V41LogitScaleAvailable() {
		return nil
	}
	return func(values []float32, factor float32) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeSelected := func(primary error) {
			if s.halFailure == nil {
				s.halFailure = primary
			}
			if s.halClosed {
				return
			}
			// A secondary Close panic must not replace the selected failure
			// or leave this session eligible for another operation.
			defer func() { s.halClosed = true; _ = recover() }()
			s.Close()
		}
		closeFailure := func(err error) error {
			if s.halFailure != nil {
				closeSelected(s.halFailure)
				return s.halFailure
			}
			selected := &V41LogitScaleOperationError{Cause: err}
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-logit-scale", Layer: -1, Stage: stage, Cause: selected}
			closeSelected(closed)
			return closed
		}
		defer func() {
			if r := recover(); r != nil {
				if err, ok := r.(error); ok {
					var closed *BackendForwardOperationError
					if errors.As(err, &closed) {
						closeSelected(err)
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
		if len(values) == 0 || len(values) > math.MaxInt32 || len(values) > math.MaxInt/4 || !finite32(factor) {
			return nil, closeFailure(errV41LogitScaleResult)
		}
		for _, value := range values {
			if !finite32(value) {
				return nil, closeFailure(errV41LogitScaleResult)
			}
		}
		run := func() (result []float32, cause error) {
			var owned []compute.Tensor
			defer func() {
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
			stage = "input upload"
			input := s.uploadHostF32([]int{len(values)}, values, compute.MemoryActivation, "V4.1 logit scale input")
			owned = append(owned, input)
			stage = "multiply"
			out, err := operation.V41LogitScale(input, factor)
			fresh := out.Buf() != nil && out.Backend() == s.Backend && out.Buf() != input.Buf()
			if fresh {
				owned = append(owned, out)
			}
			if err != nil {
				return nil, err
			}
			if !fresh || out.Dtype != compute.F32 || out.Layout != compute.RowMajor || out.Quant != nil ||
				len(out.Shape) != 1 || out.Shape[0] != len(values) {
				return nil, errV41LogitScaleResult
			}
			stage = "multiply readback"
			read := s.Backend.Read(out)
			if len(read) != len(values) {
				return nil, errV41LogitScaleResult
			}
			for _, value := range read {
				// Finite binary32 multiplication may overflow to infinity, as
				// the existing host helper does. It cannot produce a NaN.
				if math.IsNaN(float64(value)) {
					return nil, errV41LogitScaleResult
				}
			}
			return append([]float32(nil), read...), nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
}
