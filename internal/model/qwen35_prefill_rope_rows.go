package model

import "math"

// qwen35PrefillRopeRows is one host prefill invocation's immutable-after-build
// RoPE table. It reuses the existing forward rope owner and canonical row builder;
// it is never installed on a Model, Session, KV cache, or global cache.
type qwen35PrefillRopeRows struct {
	base, rows int
	table      rope
	inv        []float64
	scale      float64
}

// Only the two ordinary builtin host producers may create this owner. Legacy
// per-layer/callback entrypoints pass nil, and Metal/HAL paths retain their old
// row production. The normal hybrid admission already rejects distinct per-layer
// theta; repeat that proof here rather than relying on one model fixture.
func (s *Session) newQwen35PrefillRopeRows(base, rows int) *qwen35PrefillRopeRows {
	if !qwen35PrefillRopeRowsQualification || s.Backend != nil || s.Metal || s.MetalQ4K {
		return nil
	}
	cfg := s.M.Cfg
	if !cfg.IsQwen35Hybrid() || cfg.hasLayerSpecificRopeTheta() || rows <= 0 || base < 0 ||
		rows-1 > int(^uint(0)>>1)-base || cfg.rotaryDim() < 2 {
		return nil
	}
	full := 0
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			full++
		}
	}
	if full < 2 {
		return nil
	}
	return &qwen35PrefillRopeRows{base: base, rows: rows}
}

// prepare runs serially at the existing RoPE phase boundary. Each following
// parFor owns distinct row indices and joins before the next layer can read them.
// Canonical inverse-frequency and attention-scale bits are checked at every
// layer, so even a changed scaling configuration cannot reuse an incompatible
// table. Refusal leaves both this owner and the legacy row path untouched.
func (r *qwen35PrefillRopeRows) prepare(cfg Config, layer, base, rows int) bool {
	if r == nil || base != r.base || rows != r.rows || rows <= 0 ||
		layer < 0 || layer >= cfg.NumLayers || cfg.hasLayerSpecificRopeTheta() {
		return false
	}
	inv := cachedInvFreq(cfg, layer)
	scale := cfg.ropeAttentionFactor()
	if len(inv) == 0 {
		return false
	}
	if r.table.cos == nil {
		r.inv, r.scale = inv, scale
		r.table = rope{cos: make([][]float32, rows), sin: make([][]float32, rows)}
		return true
	}
	if len(inv) != len(r.inv) || math.Float64bits(scale) != math.Float64bits(r.scale) {
		return false
	}
	for i, v := range inv {
		if math.Float64bits(v) != math.Float64bits(r.inv[i]) {
			return false
		}
	}
	return true
}

// row is called once per index by the original joined parFor on the first full
// layer, then read-only on subsequent layers. It builds exactly [base,base+rows):
// a long cached prefix never causes prefix-sized allocation or trigonometry.
// ropeRowFromInvScaled retains the same F64 angle/trig, F32 conversion and scaled
// F32 multiplication as ropeRowForLayer. Consumers only rotate Q/K, not the table.
func (r *qwen35PrefillRopeRows) row(index int) (cos, sin []float32) {
	if r.table.cos[index] == nil {
		r.table.cos[index], r.table.sin[index] = ropeRowFromInvScaled(r.inv, r.base+index, r.scale)
	}
	return r.table.cos[index], r.table.sin[index]
}
