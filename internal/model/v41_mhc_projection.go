package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41MHCProjectionFunc func(layer int, input []float32, H int, eps float32, full, transposed bool) ([]float32, v41DenseProjectionOutcome, error)

func v41MHCProjectionErr(l int, cause error) error {
	return v41MHCProjectionErrNamed(l, "mhc.mixes.weight", cause)
}

func v41MHCProjectionErrNamed(l int, leaf string, cause error) error {
	if cause == nil {
		cause = errV41ProjectionResult
	}
	return &V41ProjectionOperationError{Layer: l, Leaf: leaf, Stage: string(v41StageMHC), Cause: v41StageErr(v41StageMHC, l, cause)}
}

func (m *Model) v41MHCProjector(l, H int, eps float32, full, transposed bool, scratch *v41ProjScratch) func([]float32) ([]float32, error) {
	return m.v41MHCProjectorNamed(l, "mhc.mixes.weight", H, eps, full, transposed, scratch)
}

func (m *Model) v41MHCProjectorNamed(l int, leaf string, H int, eps float32, full, transposed bool, scratch *v41ProjScratch) func([]float32) ([]float32, error) {
	var weight []float32
	loaded := false
	return func(input []float32) ([]float32, error) {
		if scratch == nil || !v41MHCMixLeaf(leaf) {
			return nil, v41MHCProjectionErrNamed(l, leaf, errV41ProjectionResult)
		}
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
		project := scratch.mhcProjection
		if leaf == "mhc.ffn_mixes.weight" {
			project = scratch.mhcFFNProjection
		}
		if project != nil {
			y, outcome, err := project(l, input, H, eps, full, transposed)
			switch outcome {
			case v41ProjectionHandled:
				if err != nil || len(y) != v41MHCMixWidth {
					return nil, v41MHCProjectionErrNamed(l, leaf, err)
				}
				for _, v := range y {
					if !finite32(v) {
						return nil, v41MHCProjectionErrNamed(l, leaf, errV41ProjectionResult)
					}
				}
				return y, nil
			case v41ProjectionError:
				return nil, v41MHCProjectionErrNamed(l, leaf, err)
			case v41ProjectionDeclined:
			default:
				return nil, v41MHCProjectionErrNamed(l, leaf, errV41ProjectionResult)
			}
		}
		opened := m.v41NowNanos()
		completed := 0
		var hostBytes int64
		defer func() { m.v41NoteMHCProjection(0, 1, 0, completed, 0, 0, 0, hostBytes, opened) }()
		if !loaded {
			var err error
			weight, err = m.v41MHCMixF32IntoNamed(l, leaf, scratch.mhc)
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
	return s.v41MHCProjectionFuncNamed("mhc.mixes.weight")
}

func (s *Session) v41MHCProjectionFuncNamed(leaf string) v41MHCProjectionFunc {
	if !v41MHCMixLeaf(leaf) {
		return nil
	}
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	return func(l int, input []float32, H int, eps float32, full, transposed bool) (result []float32, outcome v41DenseProjectionOutcome, cause error) {
		s.ensureOpenBackendSession()
		if H <= 0 || !finite32(eps) || eps <= 0 {
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
		name := layerName(l, leaf)
		out, in, present := s.M.residentShape(name)
		if !present {
			return nil, v41ProjectionDeclined, nil
		}
		if transposed {
			// Only the full-profile resident F32 manifest is reoriented here.
			// Packed transposes need a separate dtype/layout contract.
			if !full || out != width || in != v41MHCMixWidth || !s.M.has(name) {
				return nil, v41ProjectionDeclined, nil
			}
		} else if out != v41MHCMixWidth || in != width {
			return nil, v41ProjectionDeclined, nil
		}
		w, supported, _ := s.v41GroupedWeight(name, width)
		if !supported || w.elems != elems {
			return nil, v41ProjectionDeclined, nil
		}
		opened := s.M.v41NowNanos()
		completed, calls := 0, 0
		var upload, readback, hostBytes int64
		defer func() { s.M.v41NoteMHCProjection(1, 0, completed, 0, calls, upload, readback, hostBytes, opened) }()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-mhc-projection", Layer: l, Stage: stage, Cause: err}
			s.halFailure = closed
			s.Close()
			return closed
		}
		var payloadFailure error
		defer func() {
			if r := recover(); r != nil {
				// getOrStage has released its cache lock before this boundary
				// closes the session on a source-validation failure.
				if payloadFailure != nil {
					result, outcome, cause = nil, v41ProjectionError, closeFailure(payloadFailure)
					return
				}
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
				// Preserve unclassified panic identity, but never leave the
				// selected session reusable after an unclassified device failure.
				err, ok := r.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
				panic(r)
			}
		}()
		layout := "logical"
		if transposed {
			layout = "transposed-f32"
		}
		key := fmt.Sprintf("v41-mhc:%s:%v:%d:%d:%s", name, w.dtype, v41MHCMixWidth, width, layout)
		// Reorient only on a cache miss, including the model-owned cache shared
		// by restored/forked sessions. The host copy is bounded by 24*4H F32s.
		weight := s.cachedImmutableWeight(key, key, func() compute.Tensor {
			rows, cols := v41MHCMixWidth, width
			if transposed {
				rows, cols = width, v41MHCMixWidth
				meta := s.M.manifest[name]
				nbytes, ok := checkedMulInt(elems, 4)
				if !ok || meta.Nbytes != nbytes || meta.Offset < 0 || meta.Offset > len(s.M.raw) || nbytes > len(s.M.raw)-meta.Offset {
					payloadFailure = errV41ProjectionResult
					panic(payloadFailure)
				}
			}
			source, err := w.host(0, rows, cols)
			if err != nil {
				payloadFailure = err
				panic(payloadFailure)
			}
			if transposed {
				host, ok := source.Buf().(compute.HostBuffer)
				if !ok || len(host.F32()) != elems {
					payloadFailure = errV41ProjectionResult
					panic(payloadFailure)
				}
				stored := host.F32()
				logical := make([]float32, elems)
				for out := 0; out < v41MHCMixWidth; out++ {
					for in := 0; in < width; in++ {
						logical[out*width+in] = stored[in*v41MHCMixWidth+out]
					}
				}
				hostBytes = int64(elems) * 4
				source = compute.NewF32(compute.Default(), []int{v41MHCMixWidth, width}, logical)
			}
			stage = "weight upload"
			return s.Backend.Upload(source, w.dtype)
		})
		run := func() ([]float32, error) {
			stage = "activation upload"
			x := s.uploadHostF32([]int{width}, input, compute.MemoryActivation, "V4.1 mHC projection activation")
			defer s.Backend.Free(x)
			upload = int64(len(input)) * 4
			stage = "matmul"
			calls++
			y := s.Backend.MatMul(weight, x)
			defer s.Backend.Free(y)
			if y.Buf() == nil || y.Dtype != compute.F32 || len(y.Shape) != 1 || y.Shape[0] != v41MHCMixWidth {
				return nil, errV41ProjectionResult
			}
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
			// Read can alias backend-owned storage. Own the coefficients before
			// the host RMS scale and before the output tensor is released.
			values = append([]float32(nil), values...)
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
