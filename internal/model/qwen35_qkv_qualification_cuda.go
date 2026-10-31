//go:build cuda && cgo && fak_cuda_q4k_qkv_qualification && fak_cuda_v41_index_select

package model

import "github.com/anthony-chaudhary/fak/internal/compute"

type qwen35QKVQualificationBackend interface {
	Q4KQKVSharedInputQualification(wq, wk, wv, x compute.Tensor) (compute.Tensor, compute.Tensor, compute.Tensor, bool, error)
}

// This explicit qualification build probes only the combined Q/QG projection
// layout already used by the ordinary CUDA full-attention path. Bias, QK norm,
// query/gate splitting and attention retain their existing order and semantics.
func (s *Session) qwen35QKVQualification(layer int, x compute.Tensor) (compute.Tensor, compute.Tensor, compute.Tensor, bool, error) {
	candidate, ok := s.Backend.(qwen35QKVQualificationBackend)
	if !ok {
		return compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, false, nil
	}
	if s.M.Cfg.AttnOutputGate {
		if _, ok := s.Backend.(qwen35QueryGateSplitBackend); !ok {
			return compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, false, nil
		}
	}
	return candidate.Q4KQKVSharedInputQualification(
		s.matWeightHAL(layerName(layer, "self_attn.q_proj.weight")),
		s.matWeightHAL(layerName(layer, "self_attn.k_proj.weight")),
		s.matWeightHAL(layerName(layer, "self_attn.v_proj.weight")), x)
}
