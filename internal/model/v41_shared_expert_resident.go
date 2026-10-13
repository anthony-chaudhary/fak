package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41ColdSharedWeight struct {
	key     string
	dtype   compute.Dtype
	resolve func() compute.Tensor
}

// Reuse the scalar projection's resident format, optional native capability,
// cache identity and lazy staging. Admission never reads or uploads weight data.
func (s *Session) v41ColdSharedWeight(name string, out, in int) (v41ColdSharedWeight, bool) {
	dtype, ok := s.v41HeadProjectionDtype(name, out, in)
	if !ok {
		return v41ColdSharedWeight{}, false
	}
	w := v41ColdSharedWeight{dtype: dtype}
	switch {
	case s.M.has(name):
		w.key, w.resolve = name, func() compute.Tensor { return s.weightHAL(name) }
	case s.M.q8w[name] != nil:
		w.key, w.resolve = "q8:"+name, func() compute.Tensor { return s.weightHALQ8(name, s.M.q8w[name]) }
	case s.M.q4kw[name] != nil:
		w.key, w.resolve = "q4k:"+name, func() compute.Tensor { return s.weightHALQ4K(name, s.M.q4kw[name]) }
	case s.M.kqw[name] != nil:
		q := s.M.kqw[name]
		desc, supported := LookupQuantDescriptor(q.kind)
		if !supported || !desc.SupportsHAL() || (q.kind == kindQ6K && !s.useHALKQuantWeight(q)) {
			return v41ColdSharedWeight{}, false
		}
		w.key, w.resolve = desc.KeyPrefix()+name, func() compute.Tensor { return s.weightHALKQuant(name, q) }
	default:
		return v41ColdSharedWeight{}, false
	}
	return w, true
}

type v41ColdSharedResidentFunc func([]float32) ([]float32, bool, error)

// The layer takes this read-only snapshot once, after its other projections.
// Only this chain stages weights inside the subsequent shared-expert loop.
// Appending its lazy resolutions protects future tokens without rescanning the
// whole model cache for every row or temporary tensor.
func (s *Session) v41ColdSharedExpertChain(layer int) v41ColdSharedResidentFunc {
	protected := map[compute.Buffer]bool{}
	if s != nil {
		for _, weight := range s.halW {
			if weight.Buf() != nil {
				protected[weight.Buf()] = true
			}
		}
	}
	return func(input []float32) ([]float32, bool, error) {
		return s.v41ColdSharedExpertResident(layer, input, protected)
	}
}

// v41ColdSharedExpertResident retains the scalar shared expert's intermediates.
// Every publication is checked before the next weight or dependent kernel is
// touched. Only the final published row crosses the host boundary; the kernels,
// asymmetric clamp, BF16 sites and reduction order are unchanged. The cold layer
// owns selection; no callback binding or persistent activation is added here.
func (s *Session) v41ColdSharedExpertResident(layer int, input []float32, protected map[compute.Buffer]bool) (result []float32, selected bool, cause error) {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!backendHasComparableIdentity(s.Backend) || s.M.lora != nil || s.M.prism != nil {
		return nil, false, nil
	}
	cfg := s.M.Cfg
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	full, err := v41ForwardGeometry(cfg)
	limit := float32(cfg.SwigluLimit)
	if err != nil || !full || layer < 0 || layer >= cfg.NumLayers || H <= 0 || I <= 0 ||
		len(input) != H || !v41BF16Widened(input) || !finite32(limit) {
		return nil, false, nil
	}
	publication, ok := s.Backend.(compute.V41BF16PublishBackend)
	if !ok || !publication.SupportsV41BF16Publish(H) || !publication.SupportsV41BF16Publish(I) {
		return nil, false, nil
	}
	deferred, ok := v41DeferredBF16Publisher(s.Backend, H)
	if !ok || !deferred.SupportsV41DeferredStatus(I) || !compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil, false, nil
	}
	limited, haveLimited := s.Backend.(interface {
		SwiGLUWithLimit(compute.Tensor, compute.Tensor, float32) compute.Tensor
	})
	if limit > 0 && !haveLimited {
		return nil, false, nil
	}
	leaves := [3]string{"ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight", "ffn.shared_experts.w2.weight"}
	widths := [3][2]int{{I, H}, {I, H}, {H, I}}
	var weights [3]v41ColdSharedWeight
	for i, leaf := range leaves {
		weights[i], ok = s.v41ColdSharedWeight(layerName(layer, leaf), widths[i][0], widths[i][1])
		if !ok {
			return nil, false, nil
		}
	}
	s.ensureOpenBackendSession()
	selected = true
	stage, leaf, site := "weight upload", leaves[0], ""
	retire := func(err error) error {
		if s.halFailure == nil {
			path := "v41-shared-expert-resident"
			if site != "" {
				path = "v41-bf16-publish"
				err = &V41BF16BoundaryOperationError{Layer: layer, Site: site, Cause: err}
			}
			s.halFailure = &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: path, Layer: layer, Stage: stage, Cause: err}
		}
		if !s.halClosed {
			func() {
				defer func() { s.halClosed = true; _ = recover() }()
				s.Close()
			}()
		}
		if site != "" {
			return s.halFailure
		}
		return v41ProjectionOperationErr(layer, leaf, s.halFailure)
	}
	defer func() {
		if r := recover(); r != nil {
			if err, ok := r.(error); ok {
				var backend *compute.BackendError
				var closed *BackendForwardOperationError
				if errors.As(err, &backend) || errors.As(err, &closed) {
					result, cause = nil, retire(err)
					return
				}
			}
			if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
				if original, ok := r.(error); ok {
					err = original
				}
				result, cause = nil, retire(err)
				return
			}
			err, ok := r.(error)
			if !ok {
				err = fmt.Errorf("unclassified backend panic: %v", r)
			}
			retire(err)
			panic(r)
		}
		if cause != nil {
			result, cause = nil, retire(cause)
		}
	}()
	var owned []compute.Tensor
	consumed := map[compute.Buffer]bool{}
	defer func() {
		primary := recover()
		var cleanup any
		for i := len(owned) - 1; i >= 0; i-- {
			if consumed[owned[i].Buf()] {
				continue
			}
			func() {
				defer func() {
					if r := recover(); r != nil && cleanup == nil {
						cleanup = r
					}
				}()
				s.Backend.Free(owned[i])
			}()
		}
		if primary != nil {
			panic(primary)
		}
		if cause == nil && cleanup != nil {
			panic(cleanup)
		}
	}()
	own := func(t compute.Tensor) bool {
		if t.Buf() == nil || t.Backend() != s.Backend {
			return false
		}
		for _, prior := range owned {
			if prior.Buf() == t.Buf() {
				return false
			}
		}
		if protected[t.Buf()] {
			return false
		}
		owned = append(owned, t)
		return true
	}
	valid := func(t compute.Tensor, n int) bool {
		return t.Backend() == s.Backend && t.Buf() != nil && t.Dtype == compute.F32 && t.Layout == compute.RowMajor &&
			t.Quant == nil && len(t.Shape) == 1 && t.Shape[0] == n
	}
	var finalReadBytes int64
	publish := func(x compute.Tensor, n int, name string, siteID uint32, final bool) (compute.Tensor, error) {
		site, stage = name, "checked publication"
		p, status, err := deferred.V41BF16PublishDeferred(x, siteID)
		fresh, recordFresh := own(p), own(status.Record)
		if err != nil {
			return compute.Tensor{}, err
		}
		if !fresh || !valid(p, n) || !recordFresh || !valid(status.Record, 4) || status.Kind != compute.V41DeferredBF16Publish ||
			status.Count != n || status.Site != siteID || status.Records > 1 || status.Records < 0 {
			return compute.Tensor{}, errV41BF16BoundaryResult
		}
		var rows []compute.Tensor
		if final {
			rows = []compute.Tensor{p}
		}
		stage = "checked publication readback"
		consumed[status.Record.Buf()] = true // The reader consumes records on transport errors too.
		verdicts, values, err := deferred.V41ReadDeferred([]compute.V41DeferredStatus{status}, rows)
		if err != nil {
			return compute.Tensor{}, err
		}
		if final {
			for _, row := range values {
				finalReadBytes += int64(len(row)) * 4
			}
		}
		if len(verdicts) != 1 || len(values) != len(rows) {
			return compute.Tensor{}, errV41BF16BoundaryResult
		}
		if verdicts[0] != nil {
			stage = "checked publication"
			return compute.Tensor{}, verdicts[0]
		}
		if final {
			if len(values[0]) != n || !v41BF16Widened(values[0]) {
				return compute.Tensor{}, errV41BF16BoundaryResult
			}
			result = append([]float32(nil), values[0]...)
		}
		return p, nil
	}
	var xd compute.Tensor
	project := func(index int, x compute.Tensor) (compute.Tensor, error) {
		leaf, site, stage = leaves[index], "", "weight upload"
		opened := s.M.v41NowNanos()
		succeeded := 0
		var uploaded int64
		defer func() {
			var readback int64
			if index == 2 {
				readback = finalReadBytes
			}
			s.M.v41NoteDenseProjection(1, 0, succeeded, 0, uploaded, readback, opened)
		}()
		weight := weights[index].resolve()
		if weight.Buf() != nil {
			protected[weight.Buf()] = true
		}
		out, in := widths[index][0], widths[index][1]
		if weight.Backend() != s.Backend || weight.Buf() == nil || weight.Dtype != weights[index].dtype ||
			len(weight.Shape) != 2 || weight.Shape[0] != out || weight.Shape[1] != in {
			return compute.Tensor{}, errV41ProjectionResult
		}
		if index == 0 {
			stage = "activation upload"
			xd = s.uploadHostF32([]int{H}, input, compute.MemoryActivation, "V4.1 shared resident activation")
			if !own(xd) || !valid(xd, H) {
				return compute.Tensor{}, errV41ProjectionResult
			}
			x, uploaded = xd, int64(H)*4
		}
		stage = "matmul"
		y := s.Backend.MatMul(weight, x)
		if !own(y) || !valid(y, out) {
			return compute.Tensor{}, errV41ProjectionResult
		}
		succeeded = 1
		return publish(y, out, leaf+" projection", compute.V41BF16PublishSharedExpertProjection, index == 2)
	}
	gate, err := project(0, compute.Tensor{})
	if err != nil {
		return nil, true, err
	}
	up, err := project(1, xd)
	if err != nil {
		return nil, true, err
	}
	leaf, site, stage = "ffn.shared_experts.activation", "", "swiglu"
	var activation compute.Tensor
	if limit > 0 {
		stage = "limited swiglu"
		activation = limited.SwiGLUWithLimit(gate, up, limit)
	} else {
		activation = s.Backend.SwiGLU(gate, up)
	}
	if !own(activation) || !valid(activation, I) {
		return nil, true, errV41ProjectionResult
	}
	intermediate, err := publish(activation, I, leaf+" down input", compute.V41BF16PublishSharedExpertDownInput, false)
	if err != nil {
		return nil, true, err
	}
	if _, err = project(2, intermediate); err != nil {
		return nil, true, err
	}
	return result, true, nil
}
