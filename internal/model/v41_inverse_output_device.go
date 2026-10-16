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
// its row-panel contract with the negated-sine [-sin,cos] table. The output row
// stays transient; at most two exact tables are session-owned. A table miss adds
// 4*R upload bytes to the unchanged 4*heads*D row upload/readback per call.
func (s *Session) v41InverseOutputRoPEFunc() v41InverseOutputRoPEFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	operation, ok := s.Backend.(v41TailRoPERowsBackend)
	if !ok || !operation.V41TailRoPERowsAvailable() {
		return nil
	}
	// Position, theta and amplitude are represented by the exact supplied table
	// bits. Session-local keys keep those mutable values out of model residency;
	// a rebound callback must also match the live handle before borrowing it.
	keys := [2]string{"v41-inverse-rope-table:0", "v41-inverse-rope-table:1"}
	type tableEntry struct {
		values []float32
		tensor compute.Tensor
		model  *Model
		epoch  uint64
	}
	var tables [2]tableEntry
	next := 0
	return func(layer int, out, cos, conjSin []float32, heads, headDim, rotaryDim int) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeSelected := func(primary error) {
			if s.halFailure == nil {
				s.halFailure = primary
			}
			if s.halClosed {
				return
			}
			// An incoming typed error can belong to another operation. Retire
			// this owner without letting a secondary Close panic replace it.
			defer func() { s.halClosed = true; _ = recover() }()
			s.closeBackendOwned()
		}
		closeFailure := func(err error) error {
			if s.halFailure != nil {
				closeSelected(s.halFailure)
				return s.halFailure
			}
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-inverse-output-rope", Layer: layer, Stage: stage, Cause: err}
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
			var tt compute.Tensor
			for i, entry := range tables {
				live, present := s.halW[keys[i]]
				if !present || live.Buf() == nil || live.Buf() != entry.tensor.Buf() ||
					live.Backend() != s.Backend || entry.model != s.M || entry.epoch != s.cacheGeometryEpoch || len(entry.values) != len(table) {
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
				if s.halW == nil {
					s.halW = make(map[string]compute.Tensor)
				}
				key := keys[next]
				if old, present := s.halW[key]; present {
					// Every prior consumer completed its readback before replacement.
					delete(s.halW, key)
					s.Backend.Free(old)
				}
				tt = s.uploadHostF32([]int{rotaryDim / 2, 2}, table, compute.MemoryActivation, "V4.1 inverse-RoPE table")
				s.halW[key] = tt
				tables[next] = tableEntry{values: table, tensor: tt, model: s.M, epoch: s.cacheGeometryEpoch}
				next = (next + 1) % len(tables)
			}
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
			values, valid := compute.ReadV41InverseRoPEOutput(s.Backend, rotated, out, headDim, rotaryDim)
			if !valid {
				return nil, errV41TailRoPEResult
			}
			return values, nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
}
