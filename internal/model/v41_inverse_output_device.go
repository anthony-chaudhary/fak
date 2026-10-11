package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41InverseOutputRoPEFunc rotates every head of one attention output row by
// the supplied negated-sine table and returns an owned row. A nil callback keeps
// the host inverse; a selected call has no decline arm and no host retry.
// v41TailRoPERowsBackend is the model-side view of the compute row-panel
// tail-RoPE contract: every row of x[rows,headDim]
// rotates by the supplied [sin,cos] table under the V41TailRoPEQK arithmetic,
// into a fresh caller-owned tensor. A backend that lacks it keeps the host arm.
type v41TailRoPERowsBackend interface {
	V41TailRoPERowsAvailable() bool
	V41TailRoPERows(x, sinCos compute.Tensor, rows, headDim, rotaryDim int) (compute.Tensor, error)
}

type v41InverseOutputRoPEFunc func(layer int, out, cos, conjSin []float32, heads, headDim, rotaryDim int) ([]float32, error)

// Refs #13668. The inverse output rotation reuses the tail-RoPE kernel through
// its row-panel contract with the negated-sine [-sin,cos] table. It uploads the
// output row and table, then reads one rotated row back: 4*(heads*D+R) upload
// bytes and 4*heads*D readback bytes per call. It does not claim residency.
func (s *Session) v41InverseOutputRoPEFunc() v41InverseOutputRoPEFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	operation, ok := s.Backend.(v41TailRoPERowsBackend)
	if !ok || !operation.V41TailRoPERowsAvailable() {
		return nil
	}
	return func(layer int, out, cos, conjSin []float32, heads, headDim, rotaryDim int) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-inverse-output-rope", Layer: layer, Stage: stage, Cause: err}
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
			len(out) != width || len(cos) != rotaryDim/2 || len(conjSin) != rotaryDim/2 {
			return nil, closeFailure(errV41TailRoPEResult)
		}
		for _, row := range [][]float32{out, cos, conjSin} {
			for _, value := range row {
				if !finite32(value) {
					return nil, closeFailure(errV41TailRoPEResult)
				}
			}
		}
		table := make([]float32, rotaryDim)
		for j := range conjSin {
			table[2*j], table[2*j+1] = conjSin[j], cos[j]
		}
		run := func() ([]float32, error) {
			stage = "output upload"
			xt := s.uploadHostF32([]int{heads, headDim}, out, compute.MemoryActivation, "V4.1 inverse-RoPE output")
			defer s.Backend.Free(xt)
			stage = "table upload"
			tt := s.uploadHostF32([]int{rotaryDim / 2, 2}, table, compute.MemoryActivation, "V4.1 inverse-RoPE table")
			defer s.Backend.Free(tt)
			stage = "inverse-rope"
			rotated, err := operation.V41TailRoPERows(xt, tt, heads, headDim, rotaryDim)
			fresh := rotated.Buf() != nil && rotated.Backend() == s.Backend && rotated.Buf() != xt.Buf() && rotated.Buf() != tt.Buf()
			if fresh {
				defer s.Backend.Free(rotated)
			}
			if err != nil {
				return nil, err
			}
			if !fresh || rotated.Dtype != compute.F32 || rotated.Layout != compute.RowMajor || rotated.Quant != nil ||
				len(rotated.Shape) != 2 || rotated.Shape[0] != heads || rotated.Shape[1] != headDim {
				return nil, errV41TailRoPEResult
			}
			stage = "output readback"
			values := s.Backend.Read(rotated)
			if len(values) != width {
				return nil, errV41TailRoPEResult
			}
			for i, value := range values {
				if !finite32(value) || (i%headDim < headDim-rotaryDim && math.Float32bits(value) != math.Float32bits(out[i])) {
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
