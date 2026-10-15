package model

import (
	"errors"
	"fmt"
	"math"

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
	keys := [2]string{"v41-compressed-publication-rope-table:0", "v41-compressed-publication-rope-table:1"}
	type tableEntry struct {
		values []float32
		tensor compute.Tensor
		model  *Model
		epoch  uint64
	}
	var tables [2]tableEntry
	next := 0
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
			// Detach once before Close's flush: a secondary flush panic must not
			// strand these owners. Backend.Free retains its normal deferred-release
			// contract (Vulkan parks uncertain submissions in native quarantine).
			// A failed Free is never retried or represented as physical retirement.
			for _, key := range keys {
				if table, present := s.halW[key]; present {
					delete(s.halW, key)
					func() { defer func() { _ = recover() }(); backend.Free(table) }()
				}
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
		table := v41TailRoPEHostTable(cos, sin, [2][]float32{tables[0].values, tables[1].values})
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
			cacheOwned := func(t compute.Tensor) bool {
				for _, key := range keys {
					if table, present := s.halW[key]; present && t.Buf() == table.Buf() {
						return true
					}
				}
				return false
			}
			stage = "tail upload"
			xt := uploadHostF32Class(backend, []int{rows, rotaryDim}, append([]float32(nil), out...), compute.MemoryActivation, "V4.1 compressed publication RoPE")
			if xt.Buf() == nil || xt.Backend() != backend || cacheOwned(xt) {
				return nil, errV41TailRoPEResult
			}
			retain(xt)
			stage = "table upload"
			var tt compute.Tensor
			for i, entry := range tables {
				live, present := s.halW[keys[i]]
				if !present || live.Buf() == nil || live.Buf() != entry.tensor.Buf() ||
					live.Backend() != backend || s.Backend != backend || entry.model != s.M || entry.epoch != s.cacheGeometryEpoch ||
					live.Dtype != compute.F32 || live.Layout != compute.RowMajor || live.Quant != nil ||
					len(live.Shape) != 2 || live.Shape[0] != rotaryDim/2 || live.Shape[1] != 2 || len(entry.values) != len(table) {
					continue
				}
				equal := true
				for j, value := range table {
					if math.Float32bits(value) != math.Float32bits(entry.values[j]) {
						equal = false
						break
					}
				}
				if equal {
					tt = live
					break
				}
			}
			if tt.Buf() == nil {
				tt = uploadHostF32Class(backend, []int{rotaryDim / 2, 2}, table, compute.MemoryActivation, "V4.1 compressed publication RoPE table")
				fresh := tt.Buf() != nil && tt.Backend() == backend && tt.Buf() != xt.Buf()
				for _, key := range keys {
					if prior, present := s.halW[key]; present && prior.Buf() == tt.Buf() {
						fresh = false
					}
				}
				if fresh {
					retain(tt)
				}
				if !fresh || s.Backend != backend || tt.Dtype != compute.F32 || tt.Layout != compute.RowMajor || tt.Quant != nil ||
					len(tt.Shape) != 2 || tt.Shape[0] != rotaryDim/2 || tt.Shape[1] != 2 {
					return nil, errV41TailRoPEResult
				}
				if s.halW == nil {
					s.halW = make(map[string]compute.Tensor)
				}
				key := keys[next]
				if prior, present := s.halW[key]; present {
					// The new table is still transient if retirement panics. Delete
					// the old owner first so session failure cannot free it twice.
					delete(s.halW, key)
					backend.Free(prior)
				}
				s.halW[key] = tt
				tables[next] = tableEntry{values: table, tensor: tt, model: s.M, epoch: s.cacheGeometryEpoch}
				next = (next + 1) % len(tables)
				owners = owners[:len(owners)-1] // transfer the validated new table to the session
			}
			stage = "tail rotation"
			rotated, err := operation.V41TailRoPERows(xt, tt, rows, rotaryDim, rotaryDim)
			fresh := rotated.Buf() != nil && rotated.Backend() == backend && rotated.Buf() != xt.Buf() && !cacheOwned(rotated)
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
