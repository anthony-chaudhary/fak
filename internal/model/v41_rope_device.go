package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A nil callback keeps the existing host rotation. Selection happens only when
// the session binds an available operation; a selected call has no decline arm.
type v41TailRoPEFunc func(layer int, q, kv, cos, sin []float32, heads, headDim, rotaryDim int) (qOut, kvOut []float32, err error)

// V41TailRoPEOperationError identifies a selected tail-RoPE failure. Continuation
// callers must not treat a nested stage sentinel as permission for host replay.
type V41TailRoPEOperationError struct {
	Layer int
	Cause error
}

func (e *V41TailRoPEOperationError) Error() string {
	return fmt.Sprintf("model: selected V4.1 tail-RoPE failed at layer %d: %v", e.Layer, e.Cause)
}

func (e *V41TailRoPEOperationError) Unwrap() error { return e.Cause }

var errV41TailRoPEResult = errors.New("model: invalid selected V4.1 tail-RoPE result")

func v41TailRoPEInPlace(layer int, q, kv, cos, sin []float32, heads, headDim, rotaryDim int, rotate v41TailRoPEFunc) error {
	if rotate == nil {
		for h := 0; h < heads; h++ {
			applyRopeTailInterleaved(q[h*headDim:(h+1)*headDim], cos, sin, rotaryDim)
		}
		applyRopeTailInterleaved(kv, cos, sin, rotaryDim)
		return nil
	}
	qOut, kvOut, err := rotate(layer, q, kv, cos, sin, heads, headDim, rotaryDim)
	if err == nil && (len(qOut) != len(q) || len(kvOut) != len(kv)) {
		err = errV41TailRoPEResult
	}
	if err != nil {
		return &V41TailRoPEOperationError{Layer: layer, Cause: v41StageErr(v41StageAttention, layer, err)}
	}
	// Publish both owned rows together, only after both readbacks succeeded.
	copy(q, qOut)
	copy(kv, kvOut)
	return nil
}

// Refs #13668. This leaf uses the exact host-generated table, including its
// current per-layer theta and amplitude. It uploads Q, KV and [sin,cos], then
// reads both outputs back: 4*((heads+1)*D+R) upload bytes and 4*(heads+1)*D
// readback bytes per call. It neither establishes residency nor claims a gain.
func (s *Session) v41TailRoPEFunc() v41TailRoPEFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	operation, ok := s.Backend.(compute.V41TailRoPEBackend)
	if !ok || !operation.SupportsV41TailRoPE() {
		return nil
	}
	return func(layer int, q, kv, cos, sin []float32, heads, headDim, rotaryDim int) (qResult, kvResult []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-tail-rope", Layer: layer, Stage: stage, Cause: err}
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
						qResult, kvResult, cause = nil, nil, closeFailure(err)
						return
					}
				}
				if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
					if original, ok := r.(error); ok {
						err = original
					}
					qResult, kvResult, cause = nil, nil, closeFailure(err)
					return
				}
				// Preserve programming panic identity, while closing the selected
				// session before the existing transactional unwind restores state.
				err, ok := r.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
				panic(r)
			}
		}()
		qSize, valid := checkedMulInt(heads, headDim)
		if !valid || heads <= 0 || headDim <= 0 || rotaryDim <= 0 || rotaryDim > headDim || rotaryDim%2 != 0 ||
			len(q) != qSize || len(kv) != headDim || len(cos) != rotaryDim/2 || len(sin) != rotaryDim/2 {
			return nil, nil, closeFailure(errV41TailRoPEResult)
		}
		for _, row := range [][]float32{q, kv, cos, sin} {
			for _, value := range row {
				if !finite32(value) {
					return nil, nil, closeFailure(errV41TailRoPEResult)
				}
			}
		}
		table := make([]float32, rotaryDim)
		for j := range sin {
			table[2*j], table[2*j+1] = sin[j], cos[j]
		}
		run := func() ([]float32, []float32, error) {
			stage = "query upload"
			qt := s.uploadHostF32([]int{heads, headDim}, q, compute.MemoryActivation, "V4.1 tail-RoPE query")
			defer s.Backend.Free(qt)
			stage = "kv upload"
			kt := s.uploadHostF32([]int{headDim}, kv, compute.MemoryActivation, "V4.1 tail-RoPE KV")
			defer s.Backend.Free(kt)
			stage = "table upload"
			tt := s.uploadHostF32([]int{rotaryDim / 2, 2}, table, compute.MemoryActivation, "V4.1 tail-RoPE table")
			defer s.Backend.Free(tt)
			stage = "tail-rope"
			qo, ko, err := operation.V41TailRoPEQK(qt, kt, tt, heads, headDim, rotaryDim)
			// Even an error may return partially allocated outputs. Free each
			// owned, fresh output once, without double-freeing malformed aliases.
			fresh := func(x compute.Tensor) bool {
				return x.Buf() != nil && x.Backend() == s.Backend && x.Buf() != qt.Buf() && x.Buf() != kt.Buf() && x.Buf() != tt.Buf()
			}
			if fresh(qo) {
				defer s.Backend.Free(qo)
			}
			if fresh(ko) && ko.Buf() != qo.Buf() {
				defer s.Backend.Free(ko)
			}
			if err != nil {
				return nil, nil, err
			}
			if !fresh(qo) || !fresh(ko) || qo.Buf() == ko.Buf() ||
				qo.Dtype != compute.F32 || ko.Dtype != compute.F32 || qo.Layout != compute.RowMajor || ko.Layout != compute.RowMajor || qo.Quant != nil || ko.Quant != nil ||
				len(qo.Shape) != 2 || qo.Shape[0] != heads || qo.Shape[1] != headDim || len(ko.Shape) != 1 || ko.Shape[0] != headDim {
				return nil, nil, errV41TailRoPEResult
			}
			read := func(output compute.Tensor, input []float32) ([]float32, error) {
				values := s.Backend.Read(output)
				if len(values) != len(input) {
					return nil, errV41TailRoPEResult
				}
				for i, value := range values {
					if !finite32(value) || (i%headDim < headDim-rotaryDim && math.Float32bits(value) != math.Float32bits(input[i])) {
						return nil, errV41TailRoPEResult
					}
				}
				// Read may expose reusable backend memory. Own Q before reading
				// KV, and own both rows before their device allocations are freed.
				return append([]float32(nil), values...), nil
			}
			stage = "query readback"
			qValues, err := read(qo, q)
			if err != nil {
				return nil, nil, err
			}
			stage = "kv readback"
			kvValues, err := read(ko, kv)
			return qValues, kvValues, err
		}
		qResult, kvResult, cause = run()
		if cause != nil {
			return nil, nil, closeFailure(cause)
		}
		return qResult, kvResult, nil
	}
}
