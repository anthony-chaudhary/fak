package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41CompressorNormFunc owns the complete raw-pool -> BF16 -> learned RMSNorm
// -> BF16 tail and returns one owned, finite, widened-BF16 row. It has no decline
// outcome: once selected, any failure closes the session instead of replaying on
// the host. The gated pool is its own seam (v41CompressorPoolFunc); both BF16
// boundaries publish on the device when the backend has the ordinary operation.
type v41CompressorNormFunc func(layer int, pooled, gain []float32, eps float32) ([]float32, error)

// v41CompressorNormBackend is an explicit exact-publication qualification, not
// a generic RMSNorm availability flag. A true result promises both BF16
// boundaries and bit-identical cache publications for the full claimed input
// and learned-gain domain at width/eps. A finite corpus cannot prove this for
// arbitrary inputs or gains. A narrower artifact/gain qualification needs a
// separately designed admission identity before hardware can opt in.
//
// Ordinary Vulkan intentionally does not implement this capability: its generic
// RMSNorm reduction is not bit-exact. Vulkan binds this seam only through the
// dedicated compute.V41ExactRMSNormBackend operation, which v41ExactRMSNormFunc
// prefers over this qualification (Ref #13668).
type v41CompressorNormBackend interface {
	SupportsV41CompressorNorm(width int, eps float32) bool
}

func v41CompressorNormOperationErr(layer int, cause error) error {
	if cause == nil {
		cause = errV41ProjectionResult
	}
	return &V41ProjectionOperationError{
		Layer: layer, Leaf: "attn.compressor.norm.weight", Stage: string(v41StageCompress),
		Cause: v41StageErr(v41StageCompress, layer, cause),
	}
}

func (s *Session) v41CompressorNormFunc() v41CompressorNormFunc {
	if s == nil || s.M == nil {
		return nil
	}
	width := v41CompressorWidth(s.M.Cfg)
	eps := float32(s.M.Cfg.RMSNormEps)
	normalize := s.v41ExactRMSNormFunc(width, eps)
	if normalize == nil {
		qualified, ok := s.Backend.(v41CompressorNormBackend)
		if width <= 0 || !ok || !qualified.SupportsV41CompressorNorm(width, eps) {
			return nil
		}
		if normalize = s.v41RMSNormFunc(); normalize == nil {
			return nil
		}
	}
	publish := s.v41BF16PublishFunc()
	return func(layer int, pooled, gain []float32, suppliedEps float32) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{
				Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"),
				Path: "v41-compressor-norm", Layer: layer, Stage: stage, Cause: err,
			}
			s.halFailure = closed
			s.Close()
			return v41CompressorNormOperationErr(layer, closed)
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
				// The common RMSNorm helper may already have closed the session
				// before rethrowing an unknown panic. Keep that original latch.
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
		if len(pooled) != width {
			return nil, closeFailure(errV41ProjectionResult)
		}
		for _, value := range pooled {
			if !finite32(value) {
				return nil, closeFailure(errV41ProjectionResult)
			}
		}

		// The common helper resolves its gain and epsilon from the model. Do
		// not let that silently substitute different operands for the pool's.
		stage = "canonical operands"
		name := layerName(layer, "attn.compressor.norm.weight")
		meta, present := s.M.manifest[name]
		if v41CompressorWidth(s.M.Cfg) != width || s.M.Cfg.LayerNorm || s.M.Cfg.NormGain1p ||
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

		stage = "BF16 input"
		var input []float32
		if publish != nil {
			// The selected publication closes this session itself on failure.
			published, err := v41PublishBF16(layer, "attn.compressor.norm.weight input", pooled, publish, nil)
			if err != nil {
				return nil, v41CompressorNormOperationErr(layer, err)
			}
			input = published
		} else {
			input = append([]float32(nil), pooled...)
			for i, value := range input {
				input[i] = v41RoundBF16(value)
				if !finite32(input[i]) {
					return nil, closeFailure(errV41ProjectionResult)
				}
			}
		}
		stage = "rmsnorm"
		values, err := normalize(name, input, width, "v41-compressor-norm", layer)
		if err != nil {
			// The common helper closed/latches its exact device failure phase;
			// only add the selected compress marker required by token rollback.
			return nil, v41CompressorNormOperationErr(layer, err)
		}
		stage = "BF16 output"
		if len(values) != width {
			return nil, closeFailure(errV41ProjectionResult)
		}
		if publish != nil {
			published, err := v41PublishBF16(layer, "attn.compressor.norm.weight output", values, publish, nil)
			if err != nil {
				return nil, v41CompressorNormOperationErr(layer, err)
			}
			return published, nil
		}
		for i, value := range values {
			if !finite32(value) {
				return nil, closeFailure(errV41ProjectionResult)
			}
			values[i] = v41RoundBF16(value)
			if !finite32(values[i]) || math.Float32bits(values[i])&0xffff != 0 {
				return nil, closeFailure(errV41ProjectionResult)
			}
		}
		return values, nil
	}
}
