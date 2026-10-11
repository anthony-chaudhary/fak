package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41CompressedPublicationRoPEFunc consumes one latent tail or two packed tails
// (index first, latent second). It returns fresh owned, already widened-BF16
// tails after both checked copybacks succeed. It never publishes destination rows.
type v41CompressedPublicationRoPEFunc func(layer int, tails, cos, sin []float32, rows, rotaryDim int) ([]float32, error)

// Session.Backend is immutable for the session lifetime. Capture its comparable
// identity for operation, transfers and ownership; restored gatherOps rebinds to
// the destination Session rather than retaining a source-session closure.
func (s *Session) v41CompressedPublicationRoPEFunc() v41CompressedPublicationRoPEFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!backendHasComparableIdentity(s.Backend) ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	backend := s.Backend
	operation, ok := backend.(compute.V41TailRoPEExactBackend)
	if !ok || !operation.V41TailRoPEExactAvailable() || !operation.V41TailRoPERowsAvailable() {
		return nil
	}
	return func(layer int, out, cos, sin []float32, rows, rotaryDim int) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeSelected := func(primary error) {
			if s.halFailure == nil {
				s.halFailure = primary
			}
			if s.halClosed {
				return
			}
			// Cleanup can panic before Close marks the session closed; retain the
			// selected primary and prohibit reuse even when cleanup is incomplete.
			defer func() {
				s.halClosed = true
				_ = recover()
			}()
			s.Close()
		}
		closeFailure := func(err error) error {
			if s.halFailure != nil {
				closeSelected(s.halFailure)
				return s.halFailure
			}
			closed := &BackendForwardOperationError{Backend: backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-compressed-publication-rope", Layer: layer, Stage: stage, Cause: err}
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
		width, valid := checkedMulInt(rows, rotaryDim)
		if !valid || (rows != 1 && rows != 2) || rotaryDim <= 0 || rotaryDim%2 != 0 ||
			len(out) != width || len(cos) != rotaryDim/2 || len(sin) != rotaryDim/2 {
			return nil, closeFailure(errV41TailRoPEResult)
		}
		for _, row := range [][]float32{out, cos, sin} {
			for _, value := range row {
				if !finite32(value) {
					return nil, closeFailure(errV41TailRoPEResult)
				}
			}
		}
		table := make([]float32, rotaryDim)
		for j := range sin {
			table[2*j], table[2*j+1] = sin[j], cos[j]
		}
		run := func() (row []float32, runErr error) {
			var owners []compute.Tensor
			defer func() {
				primaryPanic := recover()
				var cleanupPanic any
				for i := len(owners) - 1; i >= 0; i-- {
					func() {
						defer func() {
							if r := recover(); r != nil && cleanupPanic == nil {
								cleanupPanic = r
							}
						}()
						backend.Free(owners[i])
					}()
				}
				if primaryPanic != nil {
					panic(primaryPanic)
				}
				if runErr == nil && cleanupPanic != nil {
					panic(cleanupPanic)
				}
			}()
			retain := func(t compute.Tensor) {
				for _, owned := range owners {
					if owned.Buf() == t.Buf() {
						return
					}
				}
				owners = append(owners, t)
			}
			stage = "tail upload"
			xt := uploadHostF32Class(backend, []int{rows, rotaryDim}, append([]float32(nil), out...), compute.MemoryActivation, "V4.1 compressed publication RoPE")
			if xt.Buf() == nil || xt.Backend() != backend {
				return nil, errV41TailRoPEResult
			}
			retain(xt)
			stage = "table upload"
			tt := uploadHostF32Class(backend, []int{rotaryDim / 2, 2}, table, compute.MemoryActivation, "V4.1 compressed publication RoPE table")
			if tt.Buf() == nil || tt.Backend() != backend {
				return nil, errV41TailRoPEResult
			}
			retain(tt)
			stage = "tail rotation"
			rotated, err := operation.V41TailRoPERows(xt, tt, rows, rotaryDim, rotaryDim)
			fresh := rotated.Buf() != nil && rotated.Backend() == backend && rotated.Buf() != xt.Buf() && rotated.Buf() != tt.Buf()
			if fresh {
				retain(rotated)
			}
			if err != nil {
				return nil, err
			}
			if !fresh || rotated.Dtype != compute.F32 || rotated.Layout != compute.RowMajor || rotated.Quant != nil ||
				len(rotated.Shape) != 2 || rotated.Shape[0] != rows || rotated.Shape[1] != rotaryDim {
				return nil, errV41TailRoPEResult
			}
			stage = "tail readback"
			values := backend.Read(rotated)
			if len(values) != width {
				return nil, errV41TailRoPEResult
			}
			// Read may lend scratch storage invalidated by the next backend call
			// or Free. Finish both conversions before cleanup or publication.
			owned := append([]float32(nil), values...)
			if rows == 2 {
				stage = "index rotary BF16 copyback"
				index, err := v41IndexBF16Copyback(layer, "indexer.k_norm.weight", "rotary", owned[:rotaryDim])
				if err != nil {
					return nil, err
				}
				copy(owned[:rotaryDim], index)
			}
			stage = "compressed rotary BF16 copyback"
			for i := (rows - 1) * rotaryDim; i < len(owned); i++ {
				value := owned[i]
				if !finite32(value) || !finite32(v41RoundBF16(value)) {
					return nil, v41StageErr(v41StageCompress, layer, fmt.Errorf("%w: rotary latent BF16 value[%d] is non-finite or overflows", ErrV41ForwardStage, i-(rows-1)*rotaryDim))
				}
				owned[i] = v41RoundBF16(value)
			}
			return owned, nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
}
