package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The KV owner admits the cache write and returns the rotated query. A false
// result must precede effects; an error is terminal even if selected is false.
type qwen35FusedQKAppender interface {
	AppendQKNormRoPE(q, k, v, qGain, kGain compute.Tensor, layer, position, heads, rotary int, epsilon float32, theta float64) (query compute.Tensor, selected bool, err error)
}

func (s *Session) qwen35TryFusedQK(layer, kvLayer, position int, q, k, v compute.Tensor) (compute.Tensor, bool, error) {
	cfg := s.M.Cfg
	if !cfg.QKNorm || cfg.LayerNorm || cfg.QKNormPerHeadWeight ||
		cfg.RopeScaling != "" || cfg.LongRope != nil {
		return compute.Tensor{}, false, nil
	}
	appender, ok := s.halKV.(qwen35FusedQKAppender)
	if !ok {
		return compute.Tensor{}, false, nil
	}
	qName := layerName(layer, "self_attn.q_norm.weight")
	kName := layerName(layer, "self_attn.k_norm.weight")
	if !s.M.hasWeight(qName) || !s.M.hasWeight(kName) {
		// Retain the ordinary typed missing-weight error.
		return compute.Tensor{}, false, nil
	}
	out, selected, err := appender.AppendQKNormRoPE(q, k, v,
		s.normWeightHAL(qName), s.normWeightHAL(kName), kvLayer, position,
		cfg.NumHeads, cfg.rotaryDim(), cfg.qkNormEps(), cfg.ropeThetaForLayer(layer))
	if err != nil {
		return compute.Tensor{}, true, err
	}
	if !selected {
		return compute.Tensor{}, false, nil
	}
	if out.Buf() == nil || out.Backend() != q.Backend() || out.Dtype != compute.F32 ||
		len(out.Shape) != 1 || out.Numel() != q.Numel() {
		return compute.Tensor{}, true, fmt.Errorf("model: fused QK append returned an invalid resident query")
	}
	return out, true, nil
}
