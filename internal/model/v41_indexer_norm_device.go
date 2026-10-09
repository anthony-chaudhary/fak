package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41IndexKeyNormFunc normalizes a newly projected indexer.wk row with its
// learned gain and returns owned, finite F32 values. This seam has no BF16
// casts and no decline outcome: a selected failure must not replay on the host.
type v41IndexKeyNormFunc func(layer int, input, gain []float32, eps float32) ([]float32, error)

// v41IndexKeyNormBackend is an index-specific exact-publication qualification,
// not generic RMSNorm availability. A true result promises bit-identical host
// RMSNorm results over the full finite input and learned-gain domain at width
// and eps, including unchanged published index vectors and discontinuous top-k
// choices. A finite corpus cannot establish that full-domain promise. A narrower
// artifact/gain qualification needs a separate admission identity.
//
// Ordinary Vulkan intentionally does not implement this capability. Backend
// name/Class/Caps, compressor qualification and CPU recorders do not qualify it.
type v41IndexKeyNormBackend interface {
	SupportsV41IndexKeyNorm(width int, eps float32) bool
}

func v41IndexKeyNormOperationErr(layer int, cause error) error {
	if cause == nil {
		cause = errV41ProjectionResult
	}
	return &V41ProjectionOperationError{
		Layer: layer, Leaf: "indexer.k_norm.weight", Stage: string(v41StageIndexer),
		Cause: v41StageErr(v41StageIndexer, layer, cause),
	}
}

func (s *Session) v41IndexKeyNormFunc() v41IndexKeyNormFunc {
	normalize := s.v41RMSNormFunc()
	if normalize == nil {
		return nil
	}
	width := s.M.Cfg.IndexHeadDim
	eps := float32(s.M.Cfg.RMSNormEps)
	qualified, ok := s.Backend.(v41IndexKeyNormBackend)
	if width <= 0 || !ok || !qualified.SupportsV41IndexKeyNorm(width, eps) {
		return nil
	}
	return func(layer int, input, gain []float32, suppliedEps float32) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{
				Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"),
				Path: "v41-index-key-norm", Layer: layer, Stage: stage, Cause: err,
			}
			s.halFailure = closed
			s.Close()
			return v41IndexKeyNormOperationErr(layer, closed)
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
				// Preserve the common helper's latch and an unknown panic's
				// identity while preventing reuse of this selected session.
				if !s.halClosed {
					err, ok := r.(error)
					if !ok {
						err = fmt.Errorf("unclassified backend panic: %v", r)
					}
					closeFailure(err)
				}
				panic(r)
			}
		}()
		if len(input) != width {
			return nil, closeFailure(errV41ProjectionResult)
		}
		for _, value := range input {
			if !finite32(value) {
				return nil, closeFailure(errV41ProjectionResult)
			}
		}

		// The shared helper resolves operands from the model. Check bit identity
		// so it cannot silently substitute different gain/epsilon operands.
		stage = "canonical operands"
		name := layerName(layer, "indexer.k_norm.weight")
		meta, present := s.M.manifest[name]
		if s.M.Cfg.IndexHeadDim != width || s.M.Cfg.LayerNorm || s.M.Cfg.NormGain1p ||
			!finite32(suppliedEps) || suppliedEps <= 0 ||
			math.Float32bits(suppliedEps) != math.Float32bits(eps) ||
			math.Float32bits(float32(s.M.Cfg.RMSNormEps)) != math.Float32bits(eps) ||
			!present || len(meta.Shape) != 1 || meta.Shape[0] != width || len(gain) != width {
			return nil, closeFailure(errV41ProjectionResult)
		}
		canonical := s.M.tensor(name)
		if len(canonical) != width {
			return nil, closeFailure(errV41ProjectionResult)
		}
		for i, value := range gain {
			if !finite32(value) || !finite32(canonical[i]) || math.Float32bits(value) != math.Float32bits(canonical[i]) {
				return nil, closeFailure(errV41ProjectionResult)
			}
		}
		stage = "rmsnorm"
		values, err := normalize(name, input, width, "v41-index-key-norm", layer)
		if err != nil {
			// The shared helper already owns the close/latch and validates and
			// retains its readback. Add the marker required by token rollback.
			return nil, v41IndexKeyNormOperationErr(layer, err)
		}
		return values, nil
	}
}

func (m *Model) v41IndexKeyNorm(layer int, input, gain []float32, eps float32, normalize v41IndexKeyNormFunc) ([]float32, error) {
	if normalize == nil {
		if len(gain) == m.Cfg.IndexHeadDim {
			return rmsnormCfg(input, gain, eps, m.Cfg), nil
		}
		return input, nil
	}
	values, err := normalize(layer, input, gain, eps)
	if err == nil && len(values) != m.Cfg.IndexHeadDim {
		err = errV41ProjectionResult
	}
	if err == nil {
		for _, value := range values {
			if !finite32(value) {
				err = errV41ProjectionResult
				break
			}
		}
	}
	if err != nil {
		return nil, v41IndexKeyNormOperationErr(layer, err)
	}
	return values, nil
}
