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

// v41LogitHostRefusal returns the typed device-only refusal for a host arm of
// the head postprocess. Nil permits the host arms.
type v41LogitHostRefusal func(op string) error

// v41LogitFactor is the binary32 multiplier logitScaleInPlace applies; 1 is
// the identity.
func v41LogitFactor(cfg Config) float32 {
	if cfg.LogitScale != 0 && cfg.LogitScale != 1 {
		return float32(cfg.LogitScale)
	}
	return 1
}

// These mirror the guards in logitScaleInPlace and softcapInPlace, so a
// postprocess op counts as used exactly when its host arm would compute.
func v41UsesLogitScale(cfg Config) bool { return v41LogitFactor(cfg) != 1 }

func v41UsesLogitSoftcap(cfg Config) bool { return !(float32(cfg.LogitSoftcap) <= 0) }

func v41UsesLogitSuppression(cfg Config) bool { return len(cfg.SuppressTokens) > 0 }

// logitHostRefusal is non-nil when the state's owning session runs under
// ExecutionPolicyDeviceOnly. Model-only forwards have no owner and no policy.
func (st *v41ForwardState) logitHostRefusal() v41LogitHostRefusal {
	if st == nil || st.callbackOwner == nil || st.callbackOwner.executionPolicy != ExecutionPolicyDeviceOnly {
		return nil
	}
	s := st.callbackOwner
	return func(op string) error { return s.v41DeviceOnlyDeclined(-1, op) }
}

// Preserve the existing float64-config-to-float32 conversion and ordering:
// multiply, exact existing host softcap, then forced token suppression.
// Under device-only (refuse non-nil) no host arm runs: a used op without its
// device seam refuses typed before the multiply is selected, and the identity
// returns the projected row untouched. Softcap and suppression have no seam.
func v41ScaleHeadLogits(logits []float32, cfg Config, scale v41LogitScaleFunc, refuse v41LogitHostRefusal) ([]float32, error) {
	factor := v41LogitFactor(cfg)
	if refuse != nil {
		switch {
		case factor != 1 && (scale == nil || !finite32(factor) || !v41AllFinite32(logits)):
			return nil, refuse("logit-scale")
		case v41UsesLogitSoftcap(cfg):
			return nil, refuse("logit-softcap")
		case v41UsesLogitSuppression(cfg):
			return nil, refuse("logit-token-suppression")
		case factor == 1:
			return logits, nil
		}
	}
	if scale == nil || factor == 1 || !finite32(factor) {
		logitScaleInPlace(logits, cfg)
		return logits, nil
	}
	// Head projection historically permits nonfinite raw results, including
	// infinities that a later softcap maps to finite logits. Such a row is
	// outside this primitive's finite-input domain, before any selection.
	if !v41AllFinite32(logits) {
		logitScaleInPlace(logits, cfg)
		return logits, nil
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

func v41AllFinite32(values []float32) bool {
	for _, value := range values {
		if !finite32(value) {
			return false
		}
	}
	return true
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
			s.closeBackendOwned()
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
