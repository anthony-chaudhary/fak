package model

import (
	"errors"
	"fmt"
	"os"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var ErrV4LiveExpert = errors.New("model: DeepSeek V4 live expert forward failed")

const (
	defaultV4ExpertRingBytes int64 = 480 << 30
	defaultV4ExpertOpenFiles       = 8
)

type v4LiveExpertRuntime interface {
	Close() error
	forward(layer, tokenID int, x, logits, correctionBias []float32) ([]float32, error)
	Stats() v4ExpertRuntimeStats
}

type v4LiveExpert struct {
	runtime *v4ExpertRuntime
}

func v4RuntimeLimits() (int64, int, error) {
	capBytes := defaultV4ExpertRingBytes
	maxOpen := defaultV4ExpertOpenFiles
	if raw := os.Getenv("FAK_V4_EXPERT_RING_BYTES"); raw != "" {
		v, err := strconv.ParseInt(raw, 10, 64)
		if err != nil || v <= 0 {
			return 0, 0, fmt.Errorf("%w: invalid FAK_V4_EXPERT_RING_BYTES=%q", ErrV4LiveExpert, raw)
		}
		capBytes = v
	}
	if raw := os.Getenv("FAK_V4_EXPERT_OPEN_FILES"); raw != "" {
		v, err := strconv.Atoi(raw)
		if err != nil || v <= 0 {
			return 0, 0, fmt.Errorf("%w: invalid FAK_V4_EXPERT_OPEN_FILES=%q", ErrV4LiveExpert, raw)
		}
		maxOpen = v
	}
	return capBytes, maxOpen, nil
}

func newV4LiveExpert(dir string, cfg Config, be compute.Backend) (*v4LiveExpert, error) {
	capBytes, maxOpen, err := v4RuntimeLimits()
	if err != nil {
		return nil, err
	}
	rt, err := newV4ExpertRuntime(dir, cfg, be, capBytes, maxOpen)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrV4LiveExpert, err)
	}
	return &v4LiveExpert{runtime: rt}, nil
}

func (v *v4LiveExpert) Close() error {
	if v == nil || v.runtime == nil {
		return nil
	}
	return v.runtime.Close()
}

func (v *v4LiveExpert) forward(layer, tokenID int, x, logits, correctionBias []float32) ([]float32, error) {
	if v == nil || v.runtime == nil {
		return nil, fmt.Errorf("%w: runtime is not initialized", ErrV4LiveExpert)
	}
	var (
		out []float32
		err error
	)
	if layer < 3 {
		out, err = v.runtime.forwardHash(layer, tokenID, logits, compute.NewF32(compute.Default(), []int{len(x)}, x))
	} else {
		out, err = v.runtime.forwardScored(layer, logits, correctionBias, compute.NewF32(compute.Default(), []int{len(x)}, x))
	}
	if err != nil {
		return nil, fmt.Errorf("%w: layer %d: %v", ErrV4LiveExpert, layer, err)
	}
	return out, nil
}

func (v *v4LiveExpert) Stats() v4ExpertRuntimeStats {
	if v == nil || v.runtime == nil {
		return v4ExpertRuntimeStats{}
	}
	return v.runtime.Stats()
}

func (s *Session) ensureV4LiveExpert() (v4LiveExpertRuntime, error) {
	if s == nil || s.M == nil || !s.M.Cfg.IsDeepSeekV4() {
		return nil, fmt.Errorf("%w: session architecture is not deepseek_v4", ErrV4LiveExpert)
	}
	if s.M.sourceDir == "" {
		return nil, fmt.Errorf("%w: model was not loaded from an indexed safetensors directory", ErrV4LiveExpert)
	}
	if s.Backend == nil {
		return nil, fmt.Errorf("%w: session has no compute backend", ErrV4LiveExpert)
	}
	if s.v4Expert != nil {
		return s.v4Expert, nil
	}
	v, err := newV4LiveExpert(s.M.sourceDir, s.M.Cfg, s.Backend)
	if err != nil {
		return nil, err
	}
	s.v4Expert = v
	return v, nil
}

// applyV4ExpertHAL is the live HAL seam for the V4 routed FFN. It keeps the
// residual tensor resident while the bounded expert runtime operates on one
// normalized token and returns the selected-expert contribution.
func (s *Session) applyV4ExpertHAL(layer, tokenID int, residual, postAttnNorm compute.Tensor, eps float32) error {
	normalized := s.Backend.RMSNorm(residual, postAttnNorm, eps)
	out, err := s.v4ExpertForward(layer, tokenID, normalized)
	if err != nil {
		return err
	}
	s.Backend.AddInPlace(residual, out)
	s.Backend.Free(out)
	return nil
}

func (s *Session) v4ExpertForward(layer, tokenID int, normalized compute.Tensor) (compute.Tensor, error) {
	v, err := s.ensureV4LiveExpert()
	if err != nil {
		return compute.Tensor{}, err
	}
	x := s.Backend.Read(normalized)
	if len(x) != s.M.Cfg.HiddenSize {
		return compute.Tensor{}, fmt.Errorf("%w: normalized width=%d want %d", ErrV4LiveExpert, len(x), s.M.Cfg.HiddenSize)
	}
	prefix := layerName(layer, "mlp.")
	gate := s.matWeightHAL(prefix + "gate.weight")
	logitsTensor := s.Backend.MatMul(gate, normalized)
	logits := s.Backend.Read(logitsTensor)
	s.Backend.Free(logitsTensor)
	if len(logits) != s.M.Cfg.NumExperts {
		return compute.Tensor{}, fmt.Errorf("%w: router width=%d want %d", ErrV4LiveExpert, len(logits), s.M.Cfg.NumExperts)
	}
	var bias []float32
	if layer >= 3 {
		biasName := prefix + "gate.e_score_correction_bias"
		if _, ok := s.M.manifest[biasName]; !ok {
			return compute.Tensor{}, fmt.Errorf("%w: missing %s", ErrV4LiveExpert, biasName)
		}
		bias = append([]float32(nil), s.M.tensor(biasName)...)
	}
	out, err := v.forward(layer, tokenID, x, logits, bias)
	if err != nil {
		return compute.Tensor{}, err
	}
	if len(out) != s.M.Cfg.HiddenSize {
		return compute.Tensor{}, fmt.Errorf("%w: expert output width=%d want %d", ErrV4LiveExpert, len(out), s.M.Cfg.HiddenSize)
	}
	return s.uploadHostF32([]int{s.M.Cfg.HiddenSize}, out, compute.MemoryActivation, "v4-expert-output"), nil
}
