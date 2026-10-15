package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41IndexRoPEFunc func(layer int, values, cos, sin []float32, heads, headDim, rotaryDim int) ([]float32, error)

func (s *Session) v41IndexRoPEFunc() v41IndexRoPEFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!backendHasComparableIdentity(s.Backend) ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	// The query is published as BF16. Raw Rows availability does not prove
	// that subnormals and separately rounded binary32 products are preserved.
	operation, ok := s.Backend.(compute.V41TailRoPEExactBackend)
	if !ok || !operation.V41TailRoPEExactAvailable() || !operation.V41TailRoPERowsAvailable() {
		return nil
	}
	backend := s.Backend
	// Eight published index sources share one exact position table during decode.
	// One R-float table stays live; upload-before-retire briefly holds two tables:
	// 4*R bytes steady and 8*R replacement payload, plus native allocation/staging.
	cache := newV41RoPETableCache(s, backend, "v41-indexer-query-rope-table:0")
	return func(layer int, out, cos, sin []float32, heads, headDim, rotaryDim int) (result []float32, cause error) {
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
			closed := &BackendForwardOperationError{Backend: backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-indexer-query-rope", Layer: layer, Stage: stage, Cause: err}
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
		width, valid := checkedMulInt(heads, headDim)
		if !valid || heads <= 0 || headDim <= 0 || rotaryDim <= 0 || rotaryDim > headDim || rotaryDim%2 != 0 ||
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
			stage = "query upload"
			xt := uploadHostF32Class(backend, []int{heads, headDim}, append([]float32(nil), out...), compute.MemoryActivation, "V4.1 indexer query RoPE")
			if xt.Buf() == nil || xt.Backend() != backend || cache.owns(xt) {
				return nil, errV41TailRoPEResult
			}
			retain(xt)
			stage = "table upload"
			tt := cache.lookup(table, rotaryDim)
			if tt.Buf() == nil {
				tt = uploadHostF32Class(backend, []int{rotaryDim / 2, 2}, table, compute.MemoryActivation, "V4.1 indexer query RoPE table")
				fresh := tt.Buf() != nil && tt.Backend() == backend && tt.Buf() != xt.Buf() && !cache.owns(tt)
				if fresh {
					retain(tt)
				}
				if !fresh || s.Backend != backend || tt.Dtype != compute.F32 || tt.Layout != compute.RowMajor || tt.Quant != nil || len(tt.Shape) != 2 || tt.Shape[0] != rotaryDim/2 || tt.Shape[1] != 2 {
					return nil, errV41TailRoPEResult
				}
				cache.install(tt, table)
				owners = owners[:len(owners)-1]
			}
			stage = "index-rope"
			rotated, err := operation.V41TailRoPERows(xt, tt, heads, headDim, rotaryDim)
			fresh := rotated.Buf() != nil && rotated.Backend() == backend && rotated.Buf() != xt.Buf() && !cache.owns(rotated)
			if fresh {
				retain(rotated)
			}
			if err != nil {
				return nil, err
			}
			if !fresh || rotated.Dtype != compute.F32 || rotated.Layout != compute.RowMajor || rotated.Quant != nil ||
				len(rotated.Shape) != 2 || rotated.Shape[0] != heads || rotated.Shape[1] != headDim {
				return nil, errV41TailRoPEResult
			}
			stage = "query readback"
			values := backend.Read(rotated)
			if len(values) != width {
				return nil, errV41TailRoPEResult
			}
			for i, value := range values {
				if !finite32(value) || (i%headDim < headDim-rotaryDim && math.Float32bits(value) != math.Float32bits(out[i])) ||
					(i%headDim >= headDim-rotaryDim && v41BF16Overflows(value)) {
					return nil, errV41TailRoPEResult
				}
			}
			// Own the row before the device allocation is released.
			return append([]float32(nil), values...), nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
}
