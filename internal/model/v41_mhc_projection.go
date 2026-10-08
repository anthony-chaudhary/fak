package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41MHCProjectionFunc func(layer int, input []float32, H int, eps float32, full, transposed bool) ([]float32, v41DenseProjectionOutcome, error)

func v41MHCProjectionErr(l int, cause error) error {
	if cause == nil {
		cause = errV41ProjectionResult
	}
	return &V41ProjectionOperationError{Layer: l, Leaf: "mhc.mixes.weight", Stage: string(v41StageMHC), Cause: v41StageErr(v41StageMHC, l, cause)}
}

func (m *Model) v41MHCProjector(l, H int, eps float32, full, transposed bool, scratch *v41ProjScratch) func([]float32) ([]float32, error) {
	var weight []float32
	loaded := false
	return func(input []float32) ([]float32, error) {
		width := H
		var valid bool
		if full {
			width, valid = checkedMulInt(4, H)
			if !valid {
				return nil, v41StageErr(v41StageMHC, l, errV41ProjectionResult)
			}
		}
		if H <= 0 || len(input) != width {
			return nil, v41StageErr(v41StageMHC, l, errV41ProjectionResult)
		}
		if scratch.mhcProjection != nil {
			y, outcome, err := scratch.mhcProjection(l, input, H, eps, full, transposed)
			switch outcome {
			case v41ProjectionHandled:
				if err != nil || len(y) != v41MHCMixWidth {
					return nil, v41MHCProjectionErr(l, err)
				}
				for _, v := range y {
					if !finite32(v) {
						return nil, v41MHCProjectionErr(l, errV41ProjectionResult)
					}
				}
				return y, nil
			case v41ProjectionError:
				return nil, v41MHCProjectionErr(l, err)
			case v41ProjectionDeclined:
			default:
				return nil, v41MHCProjectionErr(l, errV41ProjectionResult)
			}
		}
		opened := m.v41NowNanos()
		completed := 0
		var hostBytes int64
		defer func() { m.v41NoteMHCProjection(0, 1, 0, completed, 0, 0, 0, hostBytes, opened) }()
		if !loaded {
			var err error
			weight, err = m.v41MHCMixF32Into(l, scratch.mhc)
			hostBytes = int64(len(weight)) * 4
			if err != nil {
				return nil, err
			}
			scratch.mhc = weight
			loaded = true
		}
		if full {
			streams := [][]float32{input[:H], input[H : 2*H], input[2*H : 3*H], input[3*H:]}
			y, err := v41MHCProjectFull(weight, streams, H, eps, transposed)
			if err != nil {
				return nil, v41StageErr(v41StageMHC, l, err)
			}
			completed = 1
			return y, nil
		}
		n, ok := checkedMulInt(v41MHCMixWidth, H)
		if !ok || len(weight) != n {
			return nil, v41StageErr(v41StageMHC, l, errV41ProjectionResult)
		}
		y := matRows(weight, input, v41MHCMixWidth, H)
		completed = 1
		return y, nil
	}
}

func (s *Session) v41MHCProjectionFunc() v41MHCProjectionFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	return func(l int, input []float32, H int, eps float32, full, transposed bool) (result []float32, outcome v41DenseProjectionOutcome, cause error) {
		if transposed || H <= 0 || !finite32(eps) || eps <= 0 {
			return nil, v41ProjectionDeclined, nil
		}
		width := H
		var valid bool
		if full {
			width, valid = checkedMulInt(4, H)
			if !valid {
				return nil, v41ProjectionDeclined, nil
			}
		}
		elems, valid := checkedMulInt(v41MHCMixWidth, width)
		if !valid || len(input) != width {
			return nil, v41ProjectionDeclined, nil
		}
		for _, v := range input {
			if !finite32(v) {
				return nil, v41ProjectionDeclined, nil
			}
		}
		name := layerName(l, "mhc.mixes.weight")
		out, in, present := s.M.residentShape(name)
		if !present || out != v41MHCMixWidth || in != width {
			return nil, v41ProjectionDeclined, nil
		}
		w, supported, _ := s.v41GroupedWeight(name, width)
		if !supported || w.elems != elems {
			return nil, v41ProjectionDeclined, nil
		}
		opened := s.M.v41NowNanos()
		completed, calls := 0, 0
		var upload, readback int64
		defer func() { s.M.v41NoteMHCProjection(1, 0, completed, 0, calls, upload, readback, 0, opened) }()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-mhc-projection", Layer: l, Stage: stage, Cause: err}
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
				panic(r)
			}
		}()
		key := fmt.Sprintf("v41-mhc:%s:%v:%d:%d:logical", name, w.dtype, v41MHCMixWidth, width)
		var source compute.Tensor
		if _, present := s.halW[key]; !present {
			var err error
			source, err = w.host(0, v41MHCMixWidth, width)
			if err != nil {
				return nil, v41ProjectionError, closeFailure(err)
			}
		}
		stage = "weight upload"
		weight := s.cachedImmutableWeight(key, key, func() compute.Tensor { return s.Backend.Upload(source, w.dtype) })
		run := func() ([]float32, error) {
			stage = "activation upload"
			x := s.uploadHostF32([]int{width}, input, compute.MemoryActivation, "V4.1 mHC projection activation")
			defer s.Backend.Free(x)
			upload = int64(len(input)) * 4
			stage = "matmul"
			calls++
			y := s.Backend.MatMul(weight, x)
			defer s.Backend.Free(y)
			stage = "readback"
			values := s.Backend.Read(y)
			readback = int64(len(values)) * 4
			if len(values) != v41MHCMixWidth {
				return nil, errV41ProjectionResult
			}
			for _, v := range values {
				if !finite32(v) {
					return nil, errV41ProjectionResult
				}
			}
			if full {
				var ss float32
				for _, v := range input {
					ss += v * v
				}
				rsqrt := float32(1 / math.Sqrt(float64(ss/float32(len(input))+eps)))
				for i := range values {
					values[i] *= rsqrt
					if !finite32(values[i]) {
						return nil, errV41ProjectionResult
					}
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
