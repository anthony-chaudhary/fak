package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41BF16BoundaryFunc publishes an owned widened-BF16 row. Nil retains the
// existing finite-only host boundary; a selected callback never declines.
type v41BF16BoundaryFunc func(layer int, site string, values []float32) ([]float32, error)

// ErrV41BF16BoundaryOperation identifies a selected BF16 copy failure, even
// when its cause is a stage sentinel that older continuations can replay.
var ErrV41BF16BoundaryOperation = errors.New("model: selected V4.1 BF16 boundary failed")

type V41BF16BoundaryOperationError struct {
	Layer int
	Site  string
	Cause error
}

func (e *V41BF16BoundaryOperationError) Error() string {
	return fmt.Sprintf("%v at layer %d (%s): %v", ErrV41BF16BoundaryOperation, e.Layer, e.Site, e.Cause)
}

func (e *V41BF16BoundaryOperationError) Unwrap() error { return e.Cause }
func (e *V41BF16BoundaryOperationError) Is(target error) bool {
	return target == ErrV41BF16BoundaryOperation
}

var errV41BF16BoundaryResult = errors.New("model: invalid V4.1 BF16 boundary result")

func v41BF16BoundaryCopy(layer int, site string, values []float32, round v41BF16BoundaryFunc) ([]float32, error) {
	if round != nil {
		// Input validation is owned by the selected session callback, so a bad
		// payload retires that session as well as refusing this publication.
		out, err := round(layer, site, values)
		if err != nil {
			return nil, &V41BF16BoundaryOperationError{Layer: layer, Site: site, Cause: err}
		}
		if len(out) != len(values) {
			return nil, &V41BF16BoundaryOperationError{Layer: layer, Site: site, Cause: errV41BF16BoundaryResult}
		}
		return out, nil
	}
	out := make([]float32, len(values))
	for i, value := range values {
		if !finite32(value) {
			return nil, errV41BF16BoundaryResult
		}
		out[i] = v41RoundBF16(value)
		if !finite32(out[i]) {
			return nil, errV41BF16BoundaryResult
		}
	}
	return out, nil
}

// This is a host-to-device copy boundary, not a residency claim: N values cost
// 4N upload bytes, 4N readback bytes, and 8N live device bytes across distinct
// input/output allocations. Two final-norm publications cost 16N transfer
// bytes in total; the existing norm's own transfers are additional.
// Ordinary Vulkan builds do not implement the experimental backend interface.
// The bf16-boundary device-only inventory entry stays until every call site is
// migrated and the separate capability/physical qualification gates are met.
func (s *Session) v41BF16BoundaryFunc() v41BF16BoundaryFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	operation, ok := s.Backend.(compute.V41BF16BoundaryBackend)
	if !ok || !operation.V41BF16BoundaryAvailable() {
		return nil
	}
	return func(layer int, site string, values []float32) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeFailure := func(err error) error {
			if s.halFailure != nil {
				return s.halFailure
			}
			selected := &V41BF16BoundaryOperationError{Layer: layer, Site: site, Cause: err}
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-bf16-boundary", Layer: layer, Stage: stage, Cause: selected}
			s.halFailure, s.halClosed = closed, true
			// The operation failure is already latched. A secondary teardown
			// panic cannot replace it or cause another attempt on this session.
			func() { defer func() { _ = recover() }(); s.Close() }()
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
		if len(values) == 0 || len(values) > math.MaxInt32 || len(values) > math.MaxInt/4 {
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
			input := s.uploadHostF32([]int{len(values)}, values, compute.MemoryActivation, "V4.1 BF16 boundary input")
			owned = append(owned, input)
			stage = "round"
			out, err := operation.V41BF16Boundary(input)
			fresh := out.Buf() != nil && out.Backend() == s.Backend && out.Buf() != input.Buf()
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
			stage = "round readback"
			read := s.Backend.Read(out)
			if len(read) != len(values) {
				return nil, errV41BF16BoundaryResult
			}
			for _, value := range read {
				if !finite32(value) || math.Float32bits(value)&0xffff != 0 {
					return nil, errV41BF16BoundaryResult
				}
			}
			// Read may expose backend scratch. Own before either Free or the
			// next normalization/rounding operation can reuse that storage.
			return append([]float32(nil), read...), nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
}
