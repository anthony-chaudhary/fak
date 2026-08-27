//go:build darwin && arm64 && cgo

package model

import (
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func init() { qwen35HybridQ4KBatchStep = stepBatchQwen35HybridQ4KMetal }

func stepBatchQwen35HybridQ4KMetal(bs *BatchSession, ids []int) ([][]float32, bool) {
	B, m, cfg := len(ids), bs.M, bs.M.Cfg
	H, hd, nH, nKV := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	if B < 4 || B > 8 || !metalgemm.Available() || len(bs.Seqs) != B || cfg.AttnOutputGate == false {
		return nil, false
	}
	for _, s := range bs.Seqs {
		if s == nil || s.M != m || s.Backend != nil || !s.Q4K || !s.MetalQ4K || s.qwen35HAL == nil || !s.qwen35HAL.decodeAccepted || s.Cache == nil {
			return nil, false
		}
	}
	if _, err := qwen38MetalQ8RuntimeNames(cfg); err != nil {
		return nil, false
	}
	if m.promoteMetalQ8Residency() != nil {
		return nil, false
	}
	// Resolve every required resident handle before mutating any lane.
	type layerWeights struct {
		linear         *metalgemm.Qwen35DecodeWeights
		full           *metalgemm.Qwen35FullAttentionWeights
		out            *metalgemm.Q4KWeight
		gate, up, down *metalgemm.Q4KWeight
	}
	lw := make([]layerWeights, cfg.NumLayers)
	for l := range lw {
		p := func(s string) string { return layerName(l, s) }
		if cfg.isLinearAttnLayer(l) {
			n := []string{p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"), p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"), p("linear_attn.out_proj.weight")}
			h := qwen35BatchQ8Handles(m, n)
			if len(h) != 5 {
				return nil, false
			}
			w := metalgemm.Qwen35DecodeWeights{InQKV: h[0], InZ: h[1], InB: h[2], InA: h[3], Out: h[4]}
			lw[l].linear = &w
		} else {
			h := qwen35BatchQ8Handles(m, []string{p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight")})
			v := m.metalQ4KWeight(p("self_attn.v_proj.weight"), m.q4kw[p("self_attn.v_proj.weight")])
			o := m.metalQ4KWeight(p("self_attn.o_proj.weight"), m.q4kw[p("self_attn.o_proj.weight")])
			if len(h) != 2 || v == nil || o == nil {
				return nil, false
			}
			w := metalgemm.Qwen35FullAttentionWeights{Q: h[0], K: h[1], V: v}
			lw[l].full = &w
			lw[l].out = o
		}
		for _, norm := range []string{p("input_layernorm.weight"), p("post_attention_layernorm.weight")} {
			if !m.has(norm) {
				return nil, false
			}
		}
		if cfg.isLinearAttnLayer(l) {
			for _, s := range bs.Seqs {
				if qwen35BatchGDNState(s, l) == nil {
					return nil, false
				}
			}
		} else {
			for _, name := range []string{p("self_attn.q_norm.weight"), p("self_attn.k_norm.weight")} {
				if !m.has(name) {
					return nil, false
				}
			}
			kvw := nKV * hd
			for _, s := range bs.Seqs {
				pos := s.Cache.Len()
				if l >= len(s.Cache.K) || len(s.Cache.K[l]) != pos*kvw || len(s.Cache.V[l]) != pos*kvw {
					return nil, false
				}
			}
		}
		for _, item := range []struct {
			name string
			dst  **metalgemm.Q4KWeight
		}{{p("mlp.gate_proj.weight"), &lw[l].gate}, {p("mlp.up_proj.weight"), &lw[l].up}, {p("mlp.down_proj.weight"), &lw[l].down}} {
			qt := m.q4kw[item.name]
			if qt == nil {
				return nil, false
			}
			*item.dst = m.metalQ4KWeight(item.name, qt)
			if *item.dst == nil {
				return nil, false
			}
		}
	}
	if !m.has("model.norm.weight") {
		return nil, false
	}
	headName := m.headName()
	headQT := m.q4kw[headName]
	if headQT == nil {
		return nil, false
	}
	head := m.metalQ4KWeight(headName, headQT)
	if head == nil {
		return nil, false
	}

	X := make([]float32, B*H)
	m.embedRowsInto(X, ids, H, cfg)
	eps := float32(cfg.RMSNormEps)
	shared := 0
	for l, w := range lw {
		Xn := make([]float32, B*H)
		nw := m.tensor(layerName(l, "input_layernorm.weight"))
		normalizeBatchRows(Xn, X, nw, nil, B, H, eps, cfg, false)
		var attn []float32
		if w.linear != nil {
			states := make([]*metalgemm.GDNState, B)
			for i, s := range bs.Seqs {
				states[i] = qwen35BatchGDNState(s, l)
				if states[i] == nil {
					return nil, false
				}
			}
			p := func(x string) string { return layerName(l, x) }
			o, r, accepted, err := metalgemm.RunQwen35DecodeBatch(metalgemm.Qwen35DecodeBatchRequest{Input: Xn, Weights: *w.linear, States: states, Panel: metalgemm.GDNPanel{Tokens: 1, Conv1D: m.tensor(p("linear_attn.conv1d.weight")), ALog: m.tensor(p("linear_attn.A_log")), DTBias: m.tensor(p("linear_attn.dt_bias")), Norm: m.tensor(p("linear_attn.norm.weight")), RMSNormEpsilon: eps}})
			if err != nil {
				if accepted || shared > 0 {
					panic(err)
				}
				return nil, false
			}
			if !accepted {
				if shared > 0 {
					panic("model: Qwen batch linear layer declined after state mutation")
				}
				return nil, false
			}
			attn = joinQwen35Rows(o, H)
			shared += r.ProjectionDispatches
		} else {
			lanes := make([]metalgemm.Qwen35FullAttentionLane, B)
			kvw := nKV * hd
			for i, s := range bs.Seqs {
				pos := s.Cache.Len()
				lanes[i] = metalgemm.Qwen35FullAttentionLane{Position: pos, PrefixK: append([]float32(nil), s.Cache.K[l]...), PrefixV: append([]float32(nil), s.Cache.V[l]...)}
				if len(lanes[i].PrefixK) != pos*kvw {
					return nil, false
				}
			}
			max := 0
			for _, x := range lanes {
				if x.Position > max {
					max = x.Position
				}
			}
			cosv, sinv := qwen35BatchRope(cfg, l, max+1)
			p := func(x string) string { return layerName(l, x) }
			res, r, accepted, err := metalgemm.RunQwen35FullAttentionDecodeBatch(metalgemm.Qwen35FullAttentionBatchRequest{Input: Xn, Weights: *w.full, Lanes: lanes, QNorm: m.tensor(p("self_attn.q_norm.weight")), KNorm: m.tensor(p("self_attn.k_norm.weight")), Cos: cosv, Sin: sinv, NumHeads: nH, NumKVHeads: nKV, HeadDim: hd, RotaryDim: cfg.RopeDim(), Scale: cfg.attnScale(), QKNormEpsilon: eps, Gain1p: cfg.NormGain1p, QKNorm: true})
			if err != nil {
				if accepted || shared > 0 {
					panic(err)
				}
				return nil, false
			}
			if !accepted {
				if shared > 0 {
					panic("model: Qwen batch attention layer declined after state mutation")
				}
				return nil, false
			}
			shared += r.ProjectionDispatches
			flat := joinQwen35Rows(res.Output, nH*hd)
			attn = make([]float32, B*H)
			w.out.GEMVBatch(flat, B, attn)
			for i, s := range bs.Seqs {
				s.Cache.Kraw[l] = append(s.Cache.Kraw[l], res.KRaw[i]...)
				s.Cache.K[l] = append(s.Cache.K[l], res.KPost[i]...)
				s.Cache.V[l] = append(s.Cache.V[l], res.V[i]...)
			}
		}
		for i := range X {
			X[i] += attn[i]
		}
		Xn2 := make([]float32, B*H)
		normalizeBatchRows(Xn2, X, m.tensor(layerName(l, "post_attention_layernorm.weight")), nil, B, H, eps, cfg, false)
		g, u := make([]float32, B*cfg.IntermediateSize), make([]float32, B*cfg.IntermediateSize)
		w.gate.GEMVBatch(Xn2, B, g)
		w.up.GEMVBatch(Xn2, B, u)
		for i := range g {
			g[i] = silu(g[i]) * u[i]
		}
		d := make([]float32, B*H)
		w.down.GEMVBatch(g, B, d)
		for i := range X {
			X[i] += d[i]
		}
		shared += 3
	}
	for i, s := range bs.Seqs {
		s.Cache.appendPosition(s.Cache.Len(), ids[i])
	}
	norm := make([]float32, B*H)
	normalizeBatchRows(norm, X, m.tensor("model.norm.weight"), m.tensorOptional("model.norm.bias"), B, H, eps, cfg, false)
	logits := make([]float32, B*cfg.VocabSize)
	head.GEMVBatch(norm, B, logits)
	shared++
	bs.lastStepSharedPanels = shared
	bs.recordStepMACs(B)
	return splitScaledLogits(nil, logits, B, cfg.VocabSize, cfg), true
}

func qwen35BatchQ8Handles(m *Model, names []string) []*metalgemm.Q8Weight {
	metalQ4KMu.Lock()
	defer metalQ4KMu.Unlock()
	t := metalQ8KW[m]
	out := make([]*metalgemm.Q8Weight, len(names))
	for i, n := range names {
		out[i] = t[n]
		if out[i] == nil {
			return nil
		}
	}
	return out
}
func joinQwen35Rows(rows [][]float32, w int) []float32 {
	out := make([]float32, len(rows)*w)
	for i, r := range rows {
		copy(out[i*w:], r)
	}
	return out
}
func qwen35BatchRope(cfg Config, layer, n int) ([]float32, []float32) {
	rot := cfg.RopeDim()
	c := make([]float32, n*rot/2)
	s := make([]float32, len(c))
	inv := cachedInvFreq(cfg, layer)
	for p := 0; p < n; p++ {
		ropeRowInto(c[p*rot/2:(p+1)*rot/2], s[p*rot/2:(p+1)*rot/2], inv, p)
	}
	return c, s
}

func qwen35BatchGDNState(s *Session, layer int) *metalgemm.GDNState {
	b, ok := s.qwen35HAL.sequenceBackend.(*metalQwen35GDNSequenceBackend)
	if !ok {
		return nil
	}
	return b.state(s.qwen35HAL.sequenceLayers[layer])
}
