package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// v41AttentionNormBinding keeps both ordinary attention-input norm edges with
// their optional canonical owner. Replacing either edge discards panel authority;
// copying a binding preserves identity without granting a new lifetime.
type v41AttentionNormBinding struct {
	normalize v41AttentionInputNormFunc
	publish   v41BF16BoundaryFunc
	native    *v41BoundAttentionNorm
}

type v41BoundAttentionNorm struct {
	owner   *Session
	model   *Model
	backend compute.Backend
}

func (s *Session) v41BindAttentionNorm() *v41AttentionNormBinding {
	if s == nil {
		return nil
	}
	model, backend := s.M, s.Backend
	normalize := s.v41AttentionInputNormFunc()
	publish, publicationModel, publicationBackend := s.v41AttentionBF16WithProvenance()
	if normalize == nil && publish == nil {
		return nil
	}
	b := &v41AttentionNormBinding{normalize: normalize, publish: publish}
	if normalize != nil && publish != nil && model != nil && backend != nil && backendHasComparableIdentity(backend) && publicationModel == model && backendHasComparableIdentity(publicationBackend) && publicationBackend == backend {
		b.native = &v41BoundAttentionNorm{owner: s, model: model, backend: backend}
	}
	return b
}

func (b *v41AttentionNormBinding) normFunc() v41AttentionInputNormFunc {
	if b == nil {
		return nil
	}
	return b.normalize
}
func (b *v41AttentionNormBinding) publishFunc() v41BF16BoundaryFunc {
	if b == nil {
		return nil
	}
	return b.publish
}
func (b *v41AttentionNormBinding) replaceNorm(f v41AttentionInputNormFunc) *v41AttentionNormBinding {
	return &v41AttentionNormBinding{normalize: f, publish: b.publishFunc()}
}
func (b *v41AttentionNormBinding) replacePublish(f v41BF16BoundaryFunc) *v41AttentionNormBinding {
	return &v41AttentionNormBinding{normalize: b.normFunc(), publish: f}
}
func (s *v41ForwardState) attentionNormFunc() v41AttentionInputNormFunc {
	return s.attentionNormBinding.normFunc()
}
func (s *v41ForwardState) attentionBF16Func() v41BF16BoundaryFunc {
	return s.attentionNormBinding.publishFunc()
}
func (s *v41ForwardState) setAttentionNorm(f v41AttentionInputNormFunc) {
	s.attentionNormBinding = s.attentionNormBinding.replaceNorm(f)
}
func (s *v41ForwardState) setAttentionBF16(f v41BF16BoundaryFunc) {
	s.attentionNormBinding = s.attentionNormBinding.replacePublish(f)
}
func (s *v41ProjScratch) attentionNormFunc() v41AttentionInputNormFunc {
	return s.attentionNormBinding.normFunc()
}
func (s *v41ProjScratch) attentionBF16Func() v41BF16BoundaryFunc {
	return s.attentionNormBinding.publishFunc()
}

// coldPanel selects only a live canonical resident cold-forward binding. Input
// refusal is decided before any effect and keeps the original scalar row order.
// Within a selected chunk, numerical checks and output publications retain that
// order; physical dispatch and cleanup timing intentionally become chunk-local.
func (b *v41AttentionNormBinding) coldPanel(m *Model, st *v41ForwardState, scratch *v41ProjScratch, resident *v41MHCResident, layer int, input [][]float32, eps float32) ([][]float32, bool, error) {
	if b == nil || b.native == nil || len(input) <= 1 {
		return nil, false, nil
	}
	n := b.native
	valid := func() bool {
		return n.owner != nil && n.model == m && n.owner.M == m && backendHasComparableIdentity(n.owner.Backend) && n.owner.Backend == n.backend &&
			!n.owner.halClosed && n.owner.halFailure == nil && n.owner.v41Forward != nil && n.owner.v41Forward.attentionNormBinding == b &&
			st != nil && st.attentionNormBinding == b && st.callbackOwner == n.owner && scratch != nil && scratch.attentionNormBinding == b &&
			resident != nil && resident.d != nil && resident.d.s == n.owner && resident.rows == len(input)
	}
	if !valid() || layer < 0 || layer >= m.Cfg.NumLayers || eps != float32(m.Cfg.RMSNormEps) {
		return nil, false, nil
	}
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil || !full || m.Cfg.LayerNorm || m.Cfg.NormGain1p {
		return nil, false, nil
	}
	width := m.Cfg.HiddenSize
	if width <= 0 || !finite32(eps) || eps <= 0 {
		return nil, false, nil
	}
	for _, row := range input {
		if len(row) != width || !v41BF16Widened(row) {
			return nil, false, nil
		}
	}
	output := make([][]float32, len(input))
	for lo := 0; lo < len(input); lo += v41ProjPanelRowChunk {
		if !valid() {
			return nil, true, n.failBinding()
		}
		hi := min(lo+v41ProjPanelRowChunk, len(input))
		count, ok := checkedMulInt(hi-lo, width)
		if !ok {
			return nil, true, n.failBinding()
		}
		flat := make([]float32, 0, count)
		for _, row := range input[lo:hi] {
			flat = append(flat, row...)
		}
		publicationFailed := false
		values, cause := n.owner.v41RMSNormRows(layerName(layer, "attn_norm.weight"), flat, width, hi-lo, "v41-attention-input-norm", layer, func(row []float32) ([]float32, error) {
			rounded, err := v41BF16BoundaryCopy(layer, "attn_norm.weight output", row, b.publish)
			publicationFailed = err != nil
			return rounded, err
		})
		if cause != nil {
			if !publicationFailed {
				cause = v41ProjectionOperationErr(layer, "attn_norm.weight", cause)
			}
			return nil, true, cause
		}
		if !valid() {
			return nil, true, n.failBinding()
		}
		for row := lo; row < hi; row++ {
			start := (row - lo) * width
			output[row] = values[start : start+width : start+width]
		}
	}
	return output, true, nil
}

func (n *v41BoundAttentionNorm) failBinding() (cause error) {
	s := n.owner
	if s.halFailure != nil {
		return s.halFailure
	}
	cause = &BackendForwardOperationError{Backend: n.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-attention-input-norm", Layer: -1, Stage: "panel binding", Cause: errV41ProjectionResult}
	s.halFailure = cause
	defer func() { s.halClosed = true; _ = recover() }()
	s.closeBackendOwned()
	return cause
}

// v41ReleaseNormPanelTensors attempts every chunk free before a publication can
// retire the owner. Deferred frees also drain on Goexit. Completion is separate
// from recover's payload, since legacy panic(nil) carries no nonnil value.
func v41ReleaseNormPanelTensors(backend compute.Backend, owned *[]compute.Tensor, cause *error, completed *bool) {
	primary := recover()
	interrupted := !*completed
	var cleanup any
	cleanupPresent := false
	defer func() {
		*owned = nil
		if primary != nil {
			panic(primary)
		}
		if interrupted {
			if *cause == nil {
				*cause = errV41ProjectionResult
			}
			return
		}
		if *cause == nil && cleanupPresent {
			if cleanup != nil {
				panic(cleanup)
			}
			*cause = errV41ProjectionResult
		}
	}()
	// Register in acquisition order so execution remains reverse acquisition.
	for _, tensor := range *owned {
		defer func(t compute.Tensor) {
			returned := false
			defer func() {
				value := recover()
				if !returned && !cleanupPresent {
					cleanup, cleanupPresent = value, true
				}
			}()
			backend.Free(t)
			returned = true
		}(tensor)
	}
}
