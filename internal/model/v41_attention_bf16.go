package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41AttentionBF16Func binds the ordinary finite publications surrounding the
// existing attention RMSNorm. Missing capability retains the old route before
// selection. A selected failure retires this Session without host replay.
// Backend identity is immutable for the Session lifetime; restored state binds
// a new closure to its destination owner.
func (s *Session) v41AttentionBF16Func() v41BF16BoundaryFunc {
	publish, _, _ := s.v41AttentionBF16WithProvenance()
	return publish
}

// v41AttentionBF16WithProvenance reports the exact model/backend captured by
// the ordinary closure, after admission callbacks have run. Callers must not
// infer that provenance from Session fields before or after those callbacks.
func (s *Session) v41AttentionBF16WithProvenance() (v41BF16BoundaryFunc, *Model, compute.Backend) {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!backendHasComparableIdentity(s.Backend) ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil, nil, nil
	}
	if s.v41AttentionInputNormFunc() == nil {
		return nil, nil, nil
	}
	backend, model := s.Backend, s.M
	hidden, layers := model.Cfg.HiddenSize, model.Cfg.NumLayers
	operation, ok := backend.(compute.V41AttentionBF16Backend)
	if !ok || !operation.SupportsV41AttentionBF16(s.M.Cfg.HiddenSize) {
		return nil, nil, nil
	}
	return func(layer int, site string, values []float32) (result []float32, cause error) {
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
			selected := &V41BF16BoundaryOperationError{Layer: layer, Site: site, Cause: err}
			closed := &BackendForwardOperationError{Backend: backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-attention-bf16", Layer: layer, Stage: stage, Cause: selected}
			closeSelected(closed)
			return closed
		}
		ownedReadStarted, ownedReadReturned, ownedReadCompleted := false, false, false
		var ownedReadPrimary error
		// This outer completion fence runs after run's transient cleanup. Goexit
		// does not return the backend error; it must still retire the selected
		// session without closing it before input cleanup has been attempted.
		defer func() {
			if ownedReadStarted && !ownedReadCompleted {
				result = nil
				if cause == nil {
					cause = ownedReadPrimary
				}
				if cause == nil {
					cause = errV41BF16BoundaryResult
				}
				cause = closeFailure(cause)
			}
		}()
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
		if len(values) != hidden || len(values) == 0 || len(values) > math.MaxInt32 || len(values) > math.MaxInt/4 {
			return nil, closeFailure(errV41BF16BoundaryResult)
		}
		var siteID uint32
		if layer >= 0 && layer < layers {
			switch site {
			case "attn_norm.weight input":
				siteID = compute.V41BF16CheckedAttentionNormInput
			case "attn_norm.weight output":
				siteID = compute.V41BF16CheckedAttentionNormOutput
			}
		}
		if siteID == 0 {
			return nil, closeFailure(errV41BF16BoundaryResult)
		}
		for _, value := range values {
			if !finite32(value) {
				return nil, closeFailure(errV41BF16BoundaryResult)
			}
		}
		run := func() (result []float32, cause error) {
			var owned []compute.Tensor
			defer func() {
				primaryPanic := recover()
				// Preserve the selected primary if cleanup itself calls Goexit,
				// which prevents run's return values reaching the outer frame.
				if ownedReadStarted && !ownedReadCompleted {
					ownedReadPrimary = cause
					if ownedReadPrimary == nil && primaryPanic != nil {
						if err, ok := primaryPanic.(error); ok {
							ownedReadPrimary = err
						} else {
							ownedReadPrimary = fmt.Errorf("unclassified backend panic: %v", primaryPanic)
						}
					}
					if ownedReadPrimary == nil && !ownedReadReturned {
						ownedReadPrimary = errV41BF16BoundaryResult
						cause = ownedReadPrimary
					}
				}
				var cleanupPanic any
				cleanupInterrupted := false
				for i := len(owned) - 1; i >= 0; i-- {
					func() {
						released := false
						defer func() {
							r := recover()
							if !released && !cleanupInterrupted {
								cleanupPanic, cleanupInterrupted = r, true
							}
						}()
						backend.Free(owned[i])
						released = true
					}()
				}
				if primaryPanic != nil {
					panic(primaryPanic)
				}
				if cause == nil && cleanupInterrupted {
					if cleanupPanic != nil {
						panic(cleanupPanic)
					}
					cause = errV41BF16BoundaryResult
				}
			}()
			stage = "input upload"
			input := uploadHostF32Class(backend, []int{len(values)}, values, compute.MemoryActivation, "V4.1 attention BF16 input")
			if input.Buf() == nil || input.Backend() != backend {
				return nil, errV41BF16BoundaryResult
			}
			owned = append(owned, input)
			if input.Dtype != compute.F32 || input.Layout != compute.RowMajor || input.Quant != nil ||
				len(input.Shape) != 1 || input.Shape[0] != len(values) {
				return nil, errV41BF16BoundaryResult
			}
			stage = "checked publication"
			ownedReadStarted = true
			read, selected, readErr := compute.TryV41AttentionBF16ReadOwned(backend, input, siteID)
			ownedReadReturned = true
			if selected || readErr != nil {
				err := readErr
				if err != nil {
					return nil, err
				}
				stage = "checked publication readback"
				if len(read) != len(values) || !v41BF16Widened(read) {
					return nil, errV41BF16BoundaryResult
				}
				return read, nil
			}
			ownedReadCompleted = true // Genuine decline preserves the ordinary fallback.
			out, err := operation.V41AttentionBF16(input, siteID)
			fresh := out.Buf() != nil && out.Backend() == backend && out.Buf() != input.Buf()
			if fresh {
				owned = append(owned, out)
			}
			if err != nil {
				return nil, err
			}
			if !fresh || out.Dtype != compute.F32 || out.Layout != compute.RowMajor || out.Quant != nil ||
				len(out.Shape) != 1 || out.Shape[0] != len(values) {
				return nil, errV41BF16BoundaryResult
			}
			stage = "checked publication readback"
			read = backend.Read(out)
			if len(read) != len(values) {
				return nil, errV41BF16BoundaryResult
			}
			for _, value := range read {
				// Defense in depth after device status success. Reading happens
				// only after the checked operation has accepted its own record.
				if !finite32(value) || math.Float32bits(value)&0xffff != 0 {
					return nil, errV41BF16BoundaryResult
				}
			}
			return append([]float32(nil), read...), nil
		}
		result, cause = run()
		if ownedReadReturned {
			ownedReadCompleted = true
		} // Includes deferred input cleanup.
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}, model, backend
}
