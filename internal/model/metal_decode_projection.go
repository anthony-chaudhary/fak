//go:build darwin && arm64 && cgo

package model

import "github.com/anthony-chaudhary/fak/internal/metalgemm"

// metalDecodeProjection resolves one resident decode projection: the native k-quant handle when
// the loader kept the checkpoint's Q4_K/Q5_K/Q6_K rows, else the Q8 handle. ok is false when the
// projection has no resident Metal form.
func (m *Model) metalDecodeProjection(name string) (*metalgemm.Q8Weight, metalgemm.KQuantProjection, bool) {
	if qt := m.q4kw[name]; qt != nil {
		w := m.metalQ4KWeight(name, qt)
		return nil, metalgemm.KQuantProjection{Q4K: w}, w != nil
	}
	if qt := m.kqw[name]; qt != nil {
		switch qt.kind {
		case kindQ5K:
			w := m.metalQ5KWeight(name, qt)
			return nil, metalgemm.KQuantProjection{Q5K: w}, w != nil
		case kindQ6K:
			w := m.metalQ6KWeight(name, qt)
			return nil, metalgemm.KQuantProjection{Q6K: w}, w != nil
		}
		return nil, metalgemm.KQuantProjection{}, false
	}
	if qt := m.q8w[name]; qt != nil {
		w := m.metalQ8Weight(name, qt)
		return w, metalgemm.KQuantProjection{}, w != nil
	}
	return nil, metalgemm.KQuantProjection{}, false
}

// qwen35MixerDecodeWeights resolves the five linear-attention mixer projections of layer into a
// decode weight set, each served as Q8 or from its native k-quant rows.
func (m *Model) qwen35MixerDecodeWeights(layer int) (metalgemm.Qwen35DecodeWeights, bool) {
	var w metalgemm.Qwen35DecodeWeights
	q8 := [5]**metalgemm.Q8Weight{&w.InQKV, &w.InZ, &w.InB, &w.InA, &w.Out}
	for i, suffix := range [5]string{
		"linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
		"linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight", "linear_attn.out_proj.weight",
	} {
		h, native, ok := m.metalDecodeProjection(layerName(layer, suffix))
		if !ok {
			return metalgemm.Qwen35DecodeWeights{}, false
		}
		*q8[i], w.Native[i] = h, native
	}
	return w, true
}
