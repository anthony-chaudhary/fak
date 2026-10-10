// Copyright (c) 2023 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Engram projection preserves the raw output-major contraction and BF16 boundary
// in DeepSeek-V4.1-Flash at dba1be0a40aa45a94ad051997016db3960a90277 (MIT).
package model

import (
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41EngramProjectionFunc func(layer int, row []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error)

func (m *Model) v41EngramProjectionShape(l, out, in int) error {
	_, ok := checkedMulInt(out, in)
	wr, wc, present := m.residentShape(layerName(l, "engram_kv.weight"))
	if out <= 0 || in <= 0 || !ok || !present || !((wr == out && wc == in) || (wr == in && wc == out)) {
		return v41StageErr(v41StageEngram, l, fmt.Errorf("%w: invalid Engram projection shape at layer %d", ErrV41NativeUnsupported, l))
	}
	return nil
}

func (m *Model) v41EngramProjectionDimensions(l int, stage *v41EngramStage) (in, out, norm int, err error) {
	if stage == nil || stage.columns <= 0 || stage.headDim <= 0 || stage.headDim > 256 || stage.hc <= 0 || m.Cfg.HiddenSize <= 0 || stage.hc == int(^uint(0)>>1) {
		return 0, 0, 0, v41StageErr(v41StageEngram, l, fmt.Errorf("%w: invalid Engram projection geometry", ErrV41NativeUnsupported))
	}
	var ok bool
	in, ok = checkedMulInt(stage.columns, stage.headDim)
	if !ok {
		return 0, 0, 0, v41StageErr(v41StageEngram, l, ErrV41NativeUnsupported)
	}
	out, ok = checkedMulInt(stage.hc+1, m.Cfg.HiddenSize)
	if !ok {
		return 0, 0, 0, v41StageErr(v41StageEngram, l, ErrV41NativeUnsupported)
	}
	norm, ok = checkedMulInt(stage.hc, m.Cfg.HiddenSize)
	if !ok {
		return 0, 0, 0, v41StageErr(v41StageEngram, l, ErrV41NativeUnsupported)
	}
	if err = m.v41EngramProjectionShape(l, out, in); err != nil {
		return 0, 0, 0, err
	}
	return in, out, norm, nil
}

func v41EngramProjectionErr(l int, cause error) error {
	if cause == nil {
		cause = errV41ProjectionResult
	}
	return &V41ProjectionOperationError{Layer: l, Leaf: "engram_kv.weight", Stage: string(v41StageEngram), Cause: v41StageErr(v41StageEngram, l, cause)}
}

func (m *Model) v41EngramProjector(l, out, in int, project v41EngramProjectionFunc) func([]float32) ([]float32, error) {
	var weight []float32
	loaded := false
	return func(row []float32) ([]float32, error) {
		if len(row) != in {
			return nil, v41StageErr(v41StageEngram, l, ErrV41NativeUnsupported)
		}
		if project != nil {
			y, outcome, err := project(l, row, out, in)
			switch outcome {
			case v41ProjectionHandled:
				if err != nil || len(y) != out {
					return nil, v41EngramProjectionErr(l, err)
				}
				return y, nil
			case v41ProjectionError:
				return nil, v41EngramProjectionErr(l, err)
			case v41ProjectionDeclined:
			default:
				return nil, v41EngramProjectionErr(l, errV41ProjectionResult)
			}
		}
		opened := m.v41NowNanos()
		completed := 0
		var hostBytes int64
		defer func() { m.v41NoteEngramProjection(0, 1, 0, completed, 0, 0, 0, hostBytes, opened) }()
		if !loaded {
			name := layerName(l, "engram_kv.weight")
			if m.has(name) {
				weight = m.tensor(name)
			} else {
				var err error
				weight, err = m.v41ProjF32Into(l, "engram_kv.weight", nil)
				hostBytes = int64(len(weight)) * 4
				if err != nil {
					return nil, v41StageErr(v41StageEngram, l, err)
				}
			}
			hostBytes = int64(len(weight)) * 4
			n, ok := checkedMulInt(out, in)
			if !ok || len(weight) != n {
				return nil, v41StageErr(v41StageEngram, l, errV41ProjectionResult)
			}
			loaded = true
		}
		y := matRows(weight, row, out, in)
		completed = 1
		return y, nil
	}
}

func (s *Session) v41EngramProjectionFunc() v41EngramProjectionFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	return func(l int, row []float32, out, in int) (result []float32, outcome v41DenseProjectionOutcome, cause error) {
		if len(row) != in || s.M.v41EngramProjectionShape(l, out, in) != nil {
			return nil, v41ProjectionDeclined, nil
		}
		for _, v := range row {
			if !finite32(v) {
				return nil, v41ProjectionDeclined, nil
			}
		}
		name := layerName(l, "engram_kv.weight")
		descriptor, supported, _ := s.v41GroupedWeight(name, in)
		if !supported {
			return nil, v41ProjectionDeclined, nil
		}
		s.ensureOpenBackendSession()
		opened := s.M.v41NowNanos()
		completed, calls := 0, 0
		var upload, readback int64
		defer func() { s.M.v41NoteEngramProjection(1, 0, completed, 0, calls, upload, readback, 0, opened) }()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-engram-projection", Layer: l, Stage: stage, Cause: err}
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
						result, outcome, cause = nil, v41ProjectionError, closeFailure(err)
						return
					}
				}
				if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
					if original, ok := r.(error); ok {
						err = original
					}
					result, outcome, cause = nil, v41ProjectionError, closeFailure(err)
					return
				}
				// ROCm and other backends may panic with a plain error. Retire
				// the selected session while preserving that panic's identity.
				err, ok := r.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
				panic(r)
			}
		}()
		key := fmt.Sprintf("v41-engram:%s:%v:%d:%d", name, descriptor.dtype, out, in)
		var source compute.Tensor
		if _, cached := s.halW[key]; !cached {
			var err error
			source, err = descriptor.host(0, out, in)
			if err != nil {
				return nil, v41ProjectionError, closeFailure(err)
			}
		}
		stage = "weight upload"
		weight := s.cachedImmutableWeight(key, key, func() compute.Tensor { return s.Backend.Upload(source, descriptor.dtype) })
		run := func() ([]float32, error) {
			stage = "activation upload"
			x := s.uploadHostF32([]int{in}, row, compute.MemoryActivation, "V4.1 Engram projection activation")
			defer s.Backend.Free(x)
			upload += int64(len(row)) * 4
			stage = "matmul"
			calls++
			y := s.Backend.MatMul(weight, x)
			defer s.Backend.Free(y)
			stage = "readback"
			values := s.Backend.Read(y)
			readback += int64(len(values)) * 4
			if len(values) != out {
				return nil, errV41ProjectionResult
			}
			for _, v := range values {
				if !finite32(v) {
					return nil, errV41ProjectionResult
				}
			}
			return values, nil
		}
		var err error
		result, err = run()
		if err != nil {
			return nil, v41ProjectionError, closeFailure(err)
		}
		completed = 1
		return result, v41ProjectionHandled, nil
	}
}
