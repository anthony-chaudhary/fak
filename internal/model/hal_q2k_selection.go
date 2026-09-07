package model

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// rawQ2KHAL admits only projection paths whose backend consumes packed Q2_K.
// The fused recurrent-attention operation retains its F32/Q8 contract.
func (s *Session) rawQ2KHAL(name string) bool {
	if !s.useHALKQuantWeights() || strings.Contains(name, ".linear_attn.") {
		return false
	}
	capable, ok := s.Backend.(interface{ SupportsQ2K() bool })
	return ok && capable.SupportsQ2K()
}

func (s *Session) preferredQ2KHAL(name string) (compute.Tensor, bool) {
	if s.rawQ2KHAL(name) {
		if qt := s.M.kqw[name]; qt != nil && qt.kind == kindQ2K {
			return s.weightHALKQuant(name, qt), true
		}
	}
	return compute.Tensor{}, false
}

func (s *Session) q2kHALHeadName() string {
	if s.M.hasWeight("lm_head.weight") {
		return "lm_head.weight"
	}
	return "model.embed_tokens.weight"
}
