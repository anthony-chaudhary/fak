package model

import (
	"fmt"
	"strings"
)

const qwen35VerifyPanelMemoryLimit = 256 << 20

// qwen35VerifyPanel evaluates only the draft suffix through batched layer projections.
// It borrows the live session, including recurrent state. Its caller must hold a
// PrefixSnapshot until acceptance, and restore it on any execution error or panic.
// observe receives the actual normalized input row count at each layer boundary.
func (s *Session) qwen35VerifyPanel(ids []int, observe func(layer, rows int)) ([][]float32, error) {
	if err := s.admitQwen35VerifyPanel(ids); err != nil {
		return nil, err
	}
	m, cfg := s.M, s.M.Cfg
	P, H, base := len(ids), cfg.HiddenSize, s.Cache.Len()
	eps := float32(cfg.RMSNormEps)
	X := make([]float32, P*H)
	embed := m.embedRows()
	for row, id := range ids {
		copy(X[row*H:(row+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[row*H:(row+1)*H], cfg)
	}
	for layer := 0; layer < cfg.NumLayers; layer++ {
		lp := func(name string) string { return layerName(layer, name) }
		attnNorm, mlpNorm := m.attentionNorms(layer), m.mlpNorms(layer)
		Xn := make([]float32, P*H)
		for row := range ids {
			copy(Xn[row*H:(row+1)*H], normCfg(X[row*H:(row+1)*H], attnNorm.pre, attnNorm.preBias, eps, cfg))
		}
		if observe != nil {
			observe(layer, len(Xn)/H)
		}
		var O []float32
		if cfg.isLinearAttnLayer(layer) {
			rows := make([][]float32, P)
			for row := range rows {
				rows[row] = Xn[row*H : (row+1)*H]
			}
			out, err := m.linearAttnSeqBatchedState(layer, rows, &s.Cache.linear.layers[layer])
			if err != nil {
				return nil, err
			}
			O = make([]float32, P*H)
			for row := range out {
				copy(O[row*H:(row+1)*H], out[row])
			}
		} else {
			O = s.qwen35VerifyAttentionPanel(layer, Xn, P, base)
		}
		for i := range X {
			X[i] += O[i]
		}
		for row := range ids {
			copy(Xn[row*H:(row+1)*H], normCfg(X[row*H:(row+1)*H], mlpNorm.pre, mlpNorm.preBias, eps, cfg))
		}
		Down := m.batchedGatedMLP(lp, Xn, P, H, cfg.IntermediateSize, cfg)
		for i := range X {
			X[i] += Down[i]
		}
	}
	// Only complete layer execution earns new lineage and target-hidden rows.
	logits := make([][]float32, P)
	for row, id := range ids {
		x := X[row*H : (row+1)*H]
		logits[row] = s.head(m.finalNorm(x))
		s.Cache.appendPosition(base+row, id)
		s.rememberTargetHidden(base+row, id, x)
	}
	return logits, nil
}

func (s *Session) admitQwen35VerifyPanel(ids []int) error {
	if s == nil || s.M == nil || s.Cache == nil {
		return targetVerificationDowngrade("incremental panel requires a live native session")
	}
	if _, reason := qwen38OneOperationPrefix(s); reason != "" {
		return targetVerificationDowngrade(reason)
	}
	m, cfg := s.M, s.M.Cfg
	if m.lora != nil || cfg.LayerNorm || cfg.EnableResidualHook || s.activeTap() != nil || cfg.hasLayerSpecificRopeTheta() {
		return targetVerificationDowngrade("incremental panel excludes adapters, hooks and nonstandard normalization/RoPE")
	}
	if len(ids) < 1 || len(ids) > 32 {
		return targetVerificationDowngrade("incremental draft panel requires 1..32 tokens")
	}
	for _, id := range ids {
		if id < 0 || id >= cfg.VocabSize {
			return targetVerificationDowngrade("incremental draft token outside vocabulary")
		}
	}
	// Bound all arithmetic before deriving products. This is an activation/cache-growth
	// estimate, excluding resident weights, the existing cache and caller snapshots.
	for _, dimension := range []int{cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads, cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim, cfg.LinearConvKernelDim} {
		if dimension < 1 || dimension > 1<<20 {
			return targetVerificationDowngrade("incremental panel geometry outside bounded native envelope")
		}
	}
	if cfg.NumHeads%cfg.NumKVHeads != 0 {
		return targetVerificationDowngrade("incremental panel invalid GQA geometry")
	}
	keyDim := int64(cfg.LinearNumKeyHeads) * int64(cfg.LinearKeyHeadDim)
	valDim := int64(cfg.LinearNumValueHeads) * int64(cfg.LinearValueHeadDim)
	convDim := 2*keyDim + valDim
	qWidth, kvWidth := int64(cfg.NumHeads)*int64(cfg.HeadDim), int64(cfg.NumKVHeads)*int64(cfg.HeadDim)
	remaining := int64(qwen35VerifyPanelMemoryLimit / 4)
	charge := func(count, width int64) bool {
		if width <= 0 || count < 0 || count > remaining/width {
			return false
		}
		remaining -= count * width
		return true
	}
	perRow := 16*int64(cfg.HiddenSize) + 8*int64(cfg.IntermediateSize) + 12*convDim + 8*valDim + int64(cfg.VocabSize) + 8*qWidth + 6*int64(cfg.NumLayers)*kvWidth
	if !charge(int64(len(ids)), perRow) ||
		!charge(int64(s.Cache.Len()), 6*int64(cfg.NumLayers)*kvWidth) ||
		!charge(int64(s.Cache.Len())+int64(len(ids)), int64(currentWorkerCount())) {
		return targetVerificationDowngrade("incremental panel activation/cache-growth estimate exceeds 256 MiB")
	}
	if len(s.Cache.K) != cfg.NumLayers || len(s.Cache.Kraw) != cfg.NumLayers || len(s.Cache.V) != cfg.NumLayers || s.Cache.linear == nil || len(s.Cache.linear.layers) != cfg.NumLayers {
		return targetVerificationDowngrade("incremental panel cache layer geometry mismatch")
	}
	for layer := 0; layer < cfg.NumLayers; layer++ {
		projections := []struct {
			name    string
			out, in int64
		}{
			{"mlp.gate_proj.weight", int64(cfg.IntermediateSize), int64(cfg.HiddenSize)},
			{"mlp.up_proj.weight", int64(cfg.IntermediateSize), int64(cfg.HiddenSize)},
			{"mlp.down_proj.weight", int64(cfg.HiddenSize), int64(cfg.IntermediateSize)},
		}
		if cfg.isLinearAttnLayer(layer) {
			projections = append(projections, []struct {
				name    string
				out, in int64
			}{
				{"linear_attn.in_proj_qkv.weight", convDim, int64(cfg.HiddenSize)},
				{"linear_attn.in_proj_z.weight", valDim, int64(cfg.HiddenSize)},
				{"linear_attn.in_proj_b.weight", int64(cfg.LinearNumValueHeads), int64(cfg.HiddenSize)},
				{"linear_attn.in_proj_a.weight", int64(cfg.LinearNumValueHeads), int64(cfg.HiddenSize)},
				{"linear_attn.out_proj.weight", int64(cfg.HiddenSize), valDim},
			}...)
		} else {
			projections = append(projections, []struct {
				name    string
				out, in int64
			}{
				{"self_attn.q_proj.weight", 2 * qWidth, int64(cfg.HiddenSize)},
				{"self_attn.k_proj.weight", kvWidth, int64(cfg.HiddenSize)},
				{"self_attn.v_proj.weight", kvWidth, int64(cfg.HiddenSize)},
				{"self_attn.o_proj.weight", int64(cfg.HiddenSize), qWidth},
			}...)
		}
		for _, projection := range projections {
			name := layerName(layer, projection.name)
			meta, ok := m.manifest[name]
			if !ok || !strings.EqualFold(meta.Dtype, "f32") || len(meta.Shape) != 2 ||
				int64(meta.Shape[0]) != projection.out || int64(meta.Shape[1]) != projection.in ||
				int64(meta.Nbytes) != 4*projection.out*projection.in || meta.Offset < 0 ||
				meta.Offset > len(m.raw) || meta.Nbytes > len(m.raw)-meta.Offset {
				return targetVerificationDowngrade("incremental panel requires complete resident F32 projection: " + name)
			}
		}
		if m.has(layerName(layer, "self_attn.sinks")) {
			return targetVerificationDowngrade("incremental panel does not admit attention sinks")
		}
		if cfg.isLinearAttnLayer(layer) {
			if _, err := m.linearAttnSeqBatchedState(layer, nil, &s.Cache.linear.layers[layer]); err != nil {
				return targetVerificationDowngrade(err.Error())
			}
		} else if int64(len(s.Cache.K[layer])) != int64(s.Cache.Len())*kvWidth || len(s.Cache.Kraw[layer]) != len(s.Cache.K[layer]) || len(s.Cache.V[layer]) != len(s.Cache.K[layer]) {
			return targetVerificationDowngrade(fmt.Sprintf("incremental panel KV prefix mismatch at layer %d", layer))
		}
	}
	return nil
}

func (s *Session) qwen35VerifyAttentionPanel(layer int, Xn []float32, P, base int) []float32 {
	m, cfg := s.M, s.M.Cfg
	H, hd, nH, nKV := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	qw, kw := nH*hd, nKV*hd
	lp := func(name string) string { return layerName(layer, name) }
	qf := m.residentMatMulBatch(lp("self_attn.q_proj.weight"), Xn, 2*qw, H, P)
	K := m.residentMatMulBatch(lp("self_attn.k_proj.weight"), Xn, kw, H, P)
	V := m.residentMatMulBatch(lp("self_attn.v_proj.weight"), Xn, kw, H, P)
	Q, gates := make([]float32, P*qw), make([]float32, P*qw)
	for row := 0; row < P; row++ {
		q, gate := splitPackedQueryGate(qf[row*2*qw:(row+1)*2*qw], nH, hd)
		copy(Q[row*qw:(row+1)*qw], q)
		copy(gates[row*qw:(row+1)*qw], gate)
		m.applyProjBias(layer, Q[row*qw:(row+1)*qw], K[row*kw:(row+1)*kw], V[row*kw:(row+1)*kw])
		m.applyLayerQKNorm(layer, Q[row*qw:(row+1)*qw], K[row*kw:(row+1)*kw])
	}
	raw := append([]float32(nil), K...)
	for row := 0; row < P; row++ {
		cos, sin := ropeRowForLayer(cfg, layer, base+row)
		ropeRowQKInto(Q[row*qw:(row+1)*qw], K[row*kw:(row+1)*kw], cos, sin, hd, nH, nKV)
	}
	keys, values := appendLayerKV(s.Cache, layer, raw, K, V)
	attn := make([]float32, P*qw)
	attnPrefillInto(attn, Q, keys, values, P, base, nH, hd, kw, cfg.GroupSize(), cfg.windowForLayer(layer), layer, cfg.attnScale(), float32(cfg.AttnSoftcap), dot, m.attnObs)
	for i := range attn {
		attn[i] *= sigmoidf(gates[i])
	}
	out := m.residentMatMulBatch(lp("self_attn.o_proj.weight"), attn, H, qw, P)
	for row := 0; row < P; row++ {
		m.addBiasIfPresent(out[row*H:(row+1)*H], lp("self_attn.o_proj.bias"))
	}
	return out
}
