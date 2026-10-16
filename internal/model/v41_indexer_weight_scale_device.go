package model

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41IndexWeightScaleFunc returns BF16(F32(weights[h]*scale)) for finite
// widened-BF16 index weights, computed on the device: one exact binary32
// multiply on the strict scalar module, then the ordinary checked BF16
// publication of the resident product before its single readback. It never
// declines after selection; a failure retires the session without host replay.
type v41IndexWeightScaleFunc func(layer int, weights []float32, scale float32) ([]float32, error)

var ErrV41IndexWeightScaleOperation = errors.New("model: selected V4.1 indexer weight scale failed")

type V41IndexWeightScaleOperationError struct {
	Layer int
	Cause error
}

func (e *V41IndexWeightScaleOperationError) Error() string {
	return fmt.Sprintf("%v at layer %d: %v", ErrV41IndexWeightScaleOperation, e.Layer, e.Cause)
}
func (e *V41IndexWeightScaleOperationError) Unwrap() error { return e.Cause }
func (e *V41IndexWeightScaleOperationError) Is(target error) bool {
	return target == ErrV41IndexWeightScaleOperation
}

var errV41IndexWeightScaleResult = errors.New("model: invalid V4.1 indexer weight scale result")

// v41IndexWeightScaleCalls counts completed device scale seams only while a
// test has switched the BF16 witness on.
var v41IndexWeightScaleCalls atomic.Int64

// v41IndexWeightScaleFunc binds the seam when the backend admits the strict
// scalar module and the ordinary BF16 publication at the index head count.
// Missing capability keeps the host multiply before selection. Backend
// identity is immutable for the Session lifetime.
func (s *Session) v41IndexWeightScaleFunc() v41IndexWeightScaleFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!backendHasComparableIdentity(s.Backend) ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	heads := s.M.Cfg.IndexNHeads
	if heads <= 0 || heads > math.MaxInt32 {
		return nil
	}
	backend := s.Backend
	multiply, ok := backend.(compute.V41ScaleExactBackend)
	if !ok || !multiply.V41ScaleExactAvailable() {
		return nil
	}
	publish, ok := backend.(compute.V41BF16PublishBackend)
	if !ok || !publish.SupportsV41BF16Publish(heads) {
		return nil
	}
	return func(layer int, weights []float32, scale float32) (result []float32, cause error) {
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
			selected := &V41IndexWeightScaleOperationError{Layer: layer, Cause: err}
			closed := &BackendForwardOperationError{Backend: backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-index-weight-scale", Layer: layer, Stage: stage, Cause: selected}
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
					var backendErr *compute.BackendError
					if errors.As(err, &backendErr) {
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
		// The projection boundary already published finite widened BF16; any
		// other payload is outside this seam and is never multiplied.
		if len(weights) != heads || !finite32(scale) || !v41BF16Widened(weights) {
			return nil, closeFailure(errV41IndexWeightScaleResult)
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
						backend.Free(owned[i])
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
			input := uploadHostF32Class(backend, []int{heads}, weights, compute.MemoryActivation, "V4.1 index weight scale input")
			if input.Buf() == nil || input.Backend() != backend {
				return nil, errV41IndexWeightScaleResult
			}
			owned = append(owned, input)
			if input.Dtype != compute.F32 || input.Layout != compute.RowMajor || input.Quant != nil ||
				len(input.Shape) != 1 || input.Shape[0] != heads {
				return nil, errV41IndexWeightScaleResult
			}
			// The optional combined operation retains this callback's existing
			// scalar/publication admission and the ordinary deferred-read policy.
			// A refusal precedes its dispatch; selected errors never replay.
			if fused, ok := backend.(compute.V41IndexScaleBF16Backend); ok {
				if deferred, admitted := v41DeferredBF16Publisher(backend, heads); admitted && fused.SupportsV41IndexScaleBF16(input, scale) {
					stage = "exact multiply and scaled weight publication"
					published, status, err := fused.V41IndexScaleBF16Deferred(input, scale)
					freshOutput := published.Buf() != nil && published.Backend() == backend && published.Buf() != input.Buf()
					if freshOutput {
						owned = append(owned, published)
					}
					freshStatus := status.Record.Buf() != nil && status.Record.Backend() == backend && status.Record.Buf() != input.Buf() && status.Record.Buf() != published.Buf()
					if freshStatus {
						owned = append(owned, status.Record)
					}
					if err != nil {
						return nil, err
					}
					if !freshOutput || published.Dtype != compute.F32 || published.Layout != compute.RowMajor || published.Quant != nil ||
						len(published.Shape) != 1 || published.Shape[0] != heads || !freshStatus ||
						status.Kind != compute.V41DeferredBF16Publish || status.Count != heads || status.Site != compute.V41BF16PublishIndexerScaledWeight ||
						(status.Records != 0 && status.Records != 1) || status.Record.Dtype != compute.F32 || status.Record.Layout != compute.RowMajor || status.Record.Quant != nil ||
						len(status.Record.Shape) != 1 || status.Record.Shape[0] != 4 {
						return nil, errV41IndexWeightScaleResult
					}
					if v41HostBF16Witness.Load() {
						v41BF16ResidentCalls.Add(1)
						v41BF16DeferredCalls.Add(1)
					}
					// The validated status is the last owned allocation. The read
					// consumes it even on transport failure; do not retry its Free.
					owned = owned[:len(owned)-1]
					stage = "scaled weight readback"
					read, verdict, err := v41ReadDeferredBF16(deferred, published, status)
					if err != nil {
						return nil, err
					}
					if verdict != nil {
						stage = "scaled weight publication"
						return nil, verdict
					}
					if len(read) != heads || !v41BF16Widened(read) {
						return nil, errV41IndexWeightScaleResult
					}
					if v41HostBF16Witness.Load() {
						v41IndexWeightScaleCalls.Add(1)
					}
					return read, nil
				}
			}
			stage = "exact multiply"
			product, err := multiply.V41ScaleExact(input, scale)
			fresh := product.Buf() != nil && product.Backend() == backend && product.Buf() != input.Buf()
			if fresh {
				owned = append(owned, product)
			}
			if err != nil {
				return nil, err
			}
			if !fresh || product.Dtype != compute.F32 || product.Layout != compute.RowMajor || product.Quant != nil ||
				len(product.Shape) != 1 || product.Shape[0] != heads {
				return nil, errV41IndexWeightScaleResult
			}
			// The scaled weight publishes while still resident, so the F32
			// product never reaches the host unrounded.
			stage = "scaled weight publication"
			published, status, deferred, err := s.v41PublishResidentBF16Deferred(product, compute.V41BF16PublishIndexerScaledWeight)
			if err != nil {
				return nil, err
			}
			var read []float32
			if deferred {
				owned = append(owned, published)
				stage = "scaled weight readback"
				row, verdict, err := s.v41ReadResidentBF16(published, status)
				if err != nil {
					return nil, err
				}
				if verdict != nil {
					stage = "scaled weight publication"
					return nil, verdict
				}
				read = row
			} else {
				published, selected, err := s.v41PublishResidentBF16(product, compute.V41BF16PublishIndexerScaledWeight)
				if err != nil {
					return nil, err
				}
				if !selected {
					return nil, errV41IndexWeightScaleResult
				}
				owned = append(owned, published)
				stage = "scaled weight readback"
				read = append([]float32(nil), backend.Read(published)...)
			}
			if len(read) != heads || !v41BF16Widened(read) {
				return nil, errV41IndexWeightScaleResult
			}
			if v41HostBF16Witness.Load() {
				v41IndexWeightScaleCalls.Add(1)
			}
			return read, nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
}
