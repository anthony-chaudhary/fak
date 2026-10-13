package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41QueryPrefixFunc func(layer int, collapse []float32) (collapsed, qLat []float32, handled bool, err error)

// ErrV41QueryPrefixOperation prevents replay of either a selected failure or a
// healthy busy refusal. Only the latter leaves its captured session open.
var ErrV41QueryPrefixOperation = errors.New("model: V4.1 query-prefix operation refused")

type V41QueryPrefixOperationError struct {
	Layer int
	Stage string
	Cause error
}

func (e *V41QueryPrefixOperationError) Error() string {
	return fmt.Sprintf("%v at layer %d (%s): %v", ErrV41QueryPrefixOperation, e.Layer, e.Stage, e.Cause)
}
func (e *V41QueryPrefixOperationError) Unwrap() error { return e.Cause }
func (e *V41QueryPrefixOperationError) Is(target error) bool {
	return target == ErrV41QueryPrefixOperation
}

var errV41QueryPrefixResult = errors.New("model: invalid V4.1 query-prefix result")

// v41QueryPrefixFunc offers only the warm canonical F32 decode prefix. A miss
// never stages a weight or edits the memo. The backend owns both host edges and
// all intermediate tensors under one lock; the model never Uploads/Reads here.
func (s *Session) v41QueryPrefixFunc() v41QueryPrefixFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!backendHasComparableIdentity(s.Backend) || !compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	backend, model := s.Backend, s.M
	operation, ok := backend.(compute.V41QueryPrefixBackend)
	if !ok || !operation.V41QueryPrefixAvailable() {
		return nil
	}
	return func(layer int, collapse []float32) (collapsed, qLat []float32, handled bool, cause error) {
		s.ensureOpenBackendSession()
		full, geometryErr := v41ForwardGeometry(model.Cfg)
		eps := float32(model.Cfg.RMSNormEps)
		if geometryErr != nil || !full || model.Cfg.LayerNorm || model.Cfg.NormGain1p || !finite32(eps) || eps <= 0 ||
			model.Cfg.QKRopeHeadDim <= 0 || model.Cfg.QKRopeHeadDim%2 != 0 || model.Cfg.QKRopeHeadDim > model.Cfg.HeadDim ||
			model.lora != nil || !s.modelWeightsHeld || s.M != model || s.halW == nil ||
			layer < 0 || layer >= model.Cfg.NumLayers {
			return nil, nil, false, nil
		}
		h, r := model.Cfg.HiddenSize, model.Cfg.QLoraRank
		projection := layerName(layer, "attn.wq_a.weight")
		if model.prism != nil && model.prism.weightWidth[projection] != 0 {
			return nil, nil, false, nil
		}
		attentionName := layerName(layer, "attn_norm.weight")
		queryName := layerName(layer, "attn.wq_a_norm.weight")
		if !v41QueryPrefixCanonicalF32(model, attentionName, h) ||
			!v41QueryPrefixCanonicalF32(model, projection, r, h) ||
			!v41QueryPrefixCanonicalF32(model, queryName, r) {
			return nil, nil, false, nil
		}
		attention, aok := s.halW[attentionName]
		weight, wok := s.halW[projection]
		query, qok := s.halW[queryName]
		if !aok || !wok || !qok || !attention.Ready() || !weight.Ready() || !query.Ready() {
			return nil, nil, false, nil
		}
		// A callback cannot migrate across backend identity. Session.Backend is
		// immutable for its lifetime; snapshot restore installs a new callback.
		handled = true
		stage := "payload"
		closeSelected := func(primary error) {
			if s.halFailure == nil {
				s.halFailure = primary
			}
			if s.halClosed {
				return
			}
			defer func() { s.halClosed = true; _ = recover() }()
			s.closeBackendOwned()
		}
		closeFailure := func(primary error) error {
			if s.halFailure != nil {
				closeSelected(s.halFailure)
				return s.halFailure
			}
			var attributed *compute.V41QueryPrefixStageError
			if errors.As(primary, &attributed) {
				stage = attributed.Stage
			}
			selected := &V41QueryPrefixOperationError{Layer: layer, Stage: stage, Cause: primary}
			closed := &BackendForwardOperationError{Backend: backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-query-prefix", Layer: layer, Stage: stage, Cause: selected}
			closeSelected(closed)
			return closed
		}
		defer func() {
			if recovered := recover(); recovered != nil {
				collapsed, qLat, handled = nil, nil, true
				if err, ok := recovered.(error); ok {
					var prior *BackendForwardOperationError
					if errors.As(err, &prior) {
						closeSelected(err)
						panic(recovered)
					}
					var backendFault *compute.BackendError
					if errors.As(err, &backendFault) {
						cause = closeFailure(err)
						return
					}
					closeFailure(err)
				} else {
					closeFailure(fmt.Errorf("unclassified backend panic: %v", recovered))
				}
				panic(recovered)
			}
		}()
		stage = "composite"
		out, err := operation.V41QueryPrefix(compute.V41QueryPrefixInputs{
			Collapse: collapse, AttentionGainHost: model.tensor(attentionName), QueryGainHost: model.tensor(queryName),
			AttentionGain: attention, WQA: weight, QueryGain: query, Epsilon: eps,
		})
		if err != nil {
			busy, isBusy := err.(*compute.V41QueryPrefixBusyError)
			if isBusy && out.Collapsed == nil && out.QLat == nil &&
				(busy.State == "batch" || busy.State == "restore") {
				return nil, nil, true, &V41QueryPrefixOperationError{Layer: layer, Stage: "busy", Cause: err}
			}
			if isBusy {
				stage = "protocol"
				err = errors.Join(errV41QueryPrefixResult, err)
			}
			return nil, nil, true, closeFailure(err)
		}
		stage = "output"
		if len(out.Collapsed) != h || len(out.QLat) != r {
			return nil, nil, true, closeFailure(errV41QueryPrefixResult)
		}
		for _, row := range [][]float32{out.Collapsed, out.QLat} {
			for _, value := range row {
				if !finite32(value) || math.Float32bits(value)&0xffff != 0 {
					return nil, nil, true, closeFailure(errV41QueryPrefixResult)
				}
			}
		}
		// Preserve both rows even if a recorder returns borrowed host storage.
		return append([]float32(nil), out.Collapsed...), append([]float32(nil), out.QLat...), true, nil
	}
}

func v41QueryPrefixCanonicalF32(m *Model, name string, shape ...int) bool {
	meta, ok := m.manifest[name]
	if !ok || meta.Dtype != "F32" || len(meta.Shape) != len(shape) || meta.Offset < 0 || meta.Offset%4 != 0 {
		return false
	}
	count := 1
	for i, width := range shape {
		if width <= 0 || meta.Shape[i] != width || count > math.MaxInt/width {
			return false
		}
		count *= width
	}
	return count <= math.MaxInt/4 && meta.Nbytes == count*4 && meta.Offset <= len(m.raw) && meta.Nbytes <= len(m.raw)-meta.Offset
}
