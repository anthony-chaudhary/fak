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
	// Two exact tables cover the interleaved ratio-two/ratio-one decode
	// publications. Two admitted table extents stay live; replacement briefly
	// adds a third. For maximum admitted extent R, table storage is bounded by
	// 2*R*4 bytes steady and 3*R*4 bytes during replacement, even across geometry
	// changes. Native quarantine after uncertain completion is not recycled.
	// Payload and output tensors remain transient. Cold layer-major prefill may
	// miss this bounded cache; no position-keyed or model-global storage is kept.
	cache := newV41RoPETableCache(s, backend, "v41-compressed-publication-rope-table:0", "v41-compressed-publication-rope-table:1")
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
			s.closeBackendOwned()
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
		table := cache.hostTable(cos, sin)
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
			if xt.Buf() == nil || xt.Backend() != backend || cache.owns(xt) {
				return nil, errV41TailRoPEResult
			}
			retain(xt)
			stage = "table upload"
			tt := cache.lookup(table, rotaryDim)
			if tt.Buf() == nil {
				tt = uploadHostF32Class(backend, []int{rotaryDim / 2, 2}, table, compute.MemoryActivation, "V4.1 compressed publication RoPE table")
				fresh := tt.Buf() != nil && tt.Backend() == backend && tt.Buf() != xt.Buf() && !cache.owns(tt)
				if fresh {
					retain(tt)
				}
				if !fresh || s.Backend != backend || tt.Dtype != compute.F32 || tt.Layout != compute.RowMajor || tt.Quant != nil ||
					len(tt.Shape) != 2 || tt.Shape[0] != rotaryDim/2 || tt.Shape[1] != 2 {
					return nil, errV41TailRoPEResult
				}
				cache.install(tt, table)
				owners = owners[:len(owners)-1] // transfer the validated new table to the session
			}
			stage = "tail rotation"
			rotated, err := operation.V41TailRoPERows(xt, tt, rows, rotaryDim, rotaryDim)
			fresh := rotated.Buf() != nil && rotated.Backend() == backend && rotated.Buf() != xt.Buf() && !cache.owns(rotated)
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
			// The ordinary publication rounds both rotated rows while resident, so
			// under the bf16-boundary binding no tail is rounded on the host. Its
			// numeric verdict keeps each row's host error; a backend without the
			// publication keeps the host copybacks below, except under device-only.
			stage = "rotary BF16 publication"
			publicationErr := func(err error) error {
				var numeric *compute.V41BF16CheckedArithmeticError
				if !errors.As(err, &numeric) {
					return err
				}
				if rows == 2 && numeric.Index < rotaryDim {
					return &V41ProjectionOperationError{
						Layer: layer, Leaf: "indexer.k_norm.weight", Stage: string(v41StageIndexer),
						Cause: v41StageErr(v41StageIndexer, layer, fmt.Errorf("%w: indexer.k_norm.weight BF16 rotary value[%d] is non-finite or overflows", ErrV41ForwardStage, numeric.Index)),
					}
				}
				return v41StageErr(v41StageCompress, layer, fmt.Errorf("%w: rotary latent BF16 value[%d] is non-finite or overflows", ErrV41ForwardStage, numeric.Index-(rows-1)*rotaryDim))
			}
			if published, status, deferred, err := s.v41PublishResidentBF16Deferred(rotated, compute.V41BF16PublishTailRoPEOutput); deferred {
				if err != nil {
					return nil, publicationErr(err)
				}
				retain(published)
				stage = "tail readback"
				values, verdict, err := s.v41ReadResidentBF16(published, status)
				if err != nil {
					return nil, err
				}
				if verdict != nil {
					stage = "rotary BF16 publication"
					return nil, publicationErr(verdict)
				}
				if len(values) != width || !v41BF16Widened(values) {
					return nil, errV41TailRoPEResult
				}
				return values, nil
			}
			published, selected, err := s.v41PublishResidentBF16(rotated, compute.V41BF16PublishTailRoPEOutput)
			if selected && err != nil {
				return nil, publicationErr(err)
			}
			if err != nil {
				return nil, err
			}
			if selected {
				retain(published)
				stage = "tail readback"
				values := backend.Read(published)
				if len(values) != width || !v41BF16Widened(values) {
					return nil, errV41TailRoPEResult
				}
				return append([]float32(nil), values...), nil
			}
			if s.v41DeviceOnlyPolicy() {
				return nil, s.v41DeviceOnlyMissingOp(layer, "bf16-boundary")
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
				if finite32(owned[0]) {
					s.observeHostArm("bf16-boundary")
				}
				if err != nil {
					return nil, err
				}
				copy(owned[:rotaryDim], index)
			}
			stage = "compressed rotary BF16 copyback"
			for i := (rows - 1) * rotaryDim; i < len(owned); i++ {
				value := owned[i]
				rounded := value
				if finite32(value) {
					rounded = v41RoundBF16(value)
				}
				if finite32(value) && i == (rows-1)*rotaryDim {
					s.observeHostArm("bf16-boundary")
				}
				if !finite32(value) || !finite32(rounded) {
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
