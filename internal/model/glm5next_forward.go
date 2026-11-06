package model

// GLM-5.3-Flash (glm5_next) native text forward — the weights-bound engine rung
// of the epic #9433 native track. Adapted from HuggingFace Transformers
// (transformers/models/glm5_next, pinned at 19876312341f42cf49467bb24d67271cf28cb599)
// under Apache-2.0.
//
// What this file owns (and the oracle leaves glm5next_kda_*/glm5next_cadence do
// NOT): the full decoder executed against REAL checkpoint tensor names, so a
// recognized glm5_next checkpoint loads, prefills and decodes through the native
// engine instead of dying in the typed refusal. Structure per the pinned
// checkpoint (zai-org/GLM-5.3-Flash@04c4e9e9):
//
//   - 45 layers, 3:1 KDA:DSA cadence (layer%4==3 is DSA); dense MLP on 0..2,
//     288-expert top-8 sigmoid-routed MoE after
//   - mHC hyper-connections: hc_mult parallel residual streams per site
//     (hc_attn_* / hc_ffn_*), Sinkhorn-projected stream combiner, unweighted
//     stream-mean head before the final norm
//   - KDA: separate q/k/v projections + per-branch causal depthwise conv (silu),
//     f_a/f_b forget gate (A_log/dt_bias, lower-bound sigmoid), b_proj beta,
//     l2-normalized delta-rule recurrence, gated RMSNorm output + o_proj
//   - DSA: nope-only MLA (qk_rope_head_dim=0) + k-pool indexer (pooled key
//     scoring, top-k pool select, always-select tail) + sparse attention
//   - clamped SwiGLU (swiglu_limit) on every MLP branch; fp32 router math
//
// Decode is stepwise by construction: prefill advances the same per-token
// session the decode loop uses, so Forward(seq) and a session fed the same ids
// token-by-token are the same computation (glm5next_forward_test pins it).

import (
	"fmt"
	"math"
	"sort"
)

// glm5NextTensor resolves a GLM-5.3-Flash tensor by its model-internal suffix,
// trying the multimodal-wrapper prefix first (model.language_model.*) then the
// bare model.* form — the two layouts ValidateGLM5NextManifest already accepts.
func glm5NextTensor(m *Model, suffix string) []float32 {
	if v := m.tensor("model.language_model." + suffix); v != nil {
		return v
	}
	return m.tensor("model." + suffix)
}

func glm5NextHas(m *Model, suffix string) bool {
	return m.has("model.language_model."+suffix) || m.has("model."+suffix)
}

// glm5NextDims carries the resolved per-forward geometry. Standard axes come
// from the parsed Config; the glm5-specific knobs fall back to the pinned
// canonical values when the config does not state them (the exact-identity
// gate already guarantees those values for a recognized checkpoint).
type glm5NextDims struct {
	hidden, layers         int
	heads, kvHeads         int
	qLora, kvLora          int
	qkNope, qkRope, vHead  int
	vocab                  int
	eps                    float64
	linearHeads, linearDim int
	convWindow             int
	gateLowerBound         float64
	indexHeads, indexDim   int
	indexTopk, indexKpool  int
	indexAlwaysTail        bool
	hcMult, hcSinkhorn     int
	hcEps                  float64
	moeInter               int
	numExperts, topK       int
	nGroup, topKGroup      int
	nShared, firstDense    int
	routedScaling          float64
	normTopk               bool
	swigluLimit            float64
	inter                  int
}

func glm5NextDimsFromCfg(cfg Config) glm5NextDims {
	def := DefaultGLM5NextConfig()
	d := glm5NextDims{
		hidden:      cfg.HiddenSize,
		layers:      cfg.NumLayers,
		heads:       cfg.NumHeads,
		kvHeads:     cfg.NumKVHeads,
		qLora:       cfg.QLoraRank,
		kvLora:      cfg.KVLoraRank,
		qkNope:      cfg.QKNopeHeadDim,
		qkRope:      cfg.QKRopeHeadDim,
		vHead:       cfg.VHeadDim,
		vocab:       cfg.VocabSize,
		eps:         cfg.RMSNormEps,
		linearHeads: cfg.LinearNumKeyHeads,
		linearDim:   cfg.LinearKeyHeadDim,
		convWindow:  cfg.LinearConvKernelDim,
		indexHeads:  cfg.IndexNHeads,
		indexDim:    cfg.IndexHeadDim,
		indexTopk:   cfg.IndexTopK,
		indexKpool:  cfg.IndexKpool,
		hcMult:      cfg.HCMult,
		hcSinkhorn:  cfg.HCSinkhornIters,
		hcEps:       cfg.HCEps,
		moeInter:    cfg.MoEIntermediateSize,
		numExperts:  cfg.NumExperts,
		topK:        cfg.NumExpertsPerTok,
		nGroup:      cfg.NGroup,
		topKGroup:   cfg.TopKGroup,
		nShared:     cfg.NSharedExperts,
		firstDense:  cfg.FirstKDenseReplace,
		normTopk:    cfg.NormTopKProb,
		swigluLimit: cfg.SwigluLimit,
		inter:       cfg.IntermediateSize,
	}
	if cfg.LinearNumValueHeads > 0 && d.linearHeads == 0 {
		d.linearHeads = cfg.LinearNumValueHeads
	}
	if cfg.LinearValueHeadDim > 0 && d.linearDim == 0 {
		d.linearDim = cfg.LinearValueHeadDim
	}
	if d.linearHeads == 0 {
		d.linearHeads = def.KDANumHeads
	}
	if d.linearDim == 0 {
		d.linearDim = def.KDAHeadDim
	}
	if d.convWindow == 0 {
		d.convWindow = def.ConvWindowSize
	}
	if d.gateLowerBound == 0 {
		// The pinned checkpoint gates with lower_bound=-5 (decay factor in
		// (e^-5, 1)); a literal 0 means the HF no-bound softplus path.
		d.gateLowerBound = -5.0
	}
	if d.indexHeads == 0 {
		d.indexHeads = def.IndexHeads
	}
	if d.indexDim == 0 {
		d.indexDim = def.IndexHeadDim
	}
	if d.indexTopk == 0 {
		d.indexTopk = def.IndexTopK
	}
	if d.indexKpool == 0 {
		d.indexKpool = def.IndexKPool
	}
	d.indexAlwaysTail = cfg.IndexKpoolAlwaysSelectTail
	if d.hcMult == 0 {
		d.hcMult = def.HCMult
	}
	if d.hcSinkhorn == 0 {
		d.hcSinkhorn = 20
	}
	if d.hcEps == 0 {
		d.hcEps = 1e-6
	}
	if d.numExperts == 0 {
		d.numExperts = def.NRoutedExperts
	}
	if d.topK == 0 {
		d.topK = def.ExpertsPerToken
	}
	if d.firstDense == 0 {
		d.firstDense = def.FirstDenseLayers
	}
	if d.moeInter == 0 {
		d.moeInter = def.MoEIntermediate
	}
	if d.routedScaling == 0 {
		d.routedScaling = cfg.RoutedScalingFactor
	}
	if d.routedScaling == 0 {
		d.routedScaling = 2.5
	}
	if d.nGroup == 0 {
		d.nGroup = 1
	}
	if d.topKGroup == 0 {
		d.topKGroup = 1
	}
	if d.eps == 0 {
		d.eps = 1e-5
	}
	return d
}

// validateGLM5NextEngineManifest fails the load closed when any tensor the
// native forward reads is absent, naming the first missing key per class so an
// operator sees the seam instead of a mid-request panic. Runs at admission,
// after the shape-level ValidateGLM5NextManifest.
func validateGLM5NextEngineManifest(cfg Config, man map[string]tensorMeta) error {
	pick := func(suffix string) bool {
		if _, ok := man["model.language_model."+suffix]; ok {
			return true
		}
		_, ok := man["model."+suffix]
		return ok
	}
	missing := func(what, suffix string) error {
		return fmt.Errorf("model: glm5_next native forward missing %s tensor %s (or model.language_model.%s)", what, suffix, suffix)
	}
	for _, s := range []string{"embed_tokens.weight", "norm.weight"} {
		if !pick(s) {
			return missing("global", s)
		}
	}
	// The pinned checkpoint keeps the untied LM head at the top level
	// (lm_head.weight, no model.* prefix); accept that form or the prefixed
	// variants, else a tied-embedding fallback.
	if _, ok := man["lm_head.weight"]; !ok && !pick("lm_head.weight") && !cfg.TieWordEmbeddings {
		return missing("output head", "lm_head.weight")
	}
	for l := 0; l < cfg.NumLayers; l++ {
		lp := fmt.Sprintf("layers.%d.", l)
		for _, s := range []string{"input_layernorm.weight", "post_attention_layernorm.weight",
			"hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"} {
			if !pick(lp + s) {
				return missing("layer", lp+s)
			}
		}
		if l%4 == 3 {
			for _, s := range []string{"self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight",
				"self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
				"self_attn.kv_a_layernorm.weight", "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight",
				"self_attn.indexer.wq_b.weight", "self_attn.indexer.wk.weight",
				"self_attn.indexer.k_norm.weight", "self_attn.indexer.k_norm.bias",
				"self_attn.indexer.weights_proj.weight", "self_attn.indexer.index_kpool_compress_ape",
				"self_attn.indexer.index_kpool_compress_gate"} {
				if !pick(lp + s) {
					return missing("DSA layer", lp+s)
				}
			}
		} else {
			for _, s := range []string{"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
				"self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight",
				"self_attn.f_a_proj.weight", "self_attn.f_b_proj.weight", "self_attn.dt_bias", "self_attn.A_log",
				"self_attn.b_proj.weight", "self_attn.g_a_proj.weight", "self_attn.g_b_proj.weight",
				"self_attn.o_norm.weight", "self_attn.o_proj.weight"} {
				if !pick(lp + s) {
					return missing("KDA layer", lp+s)
				}
			}
		}
		if l < cfg.FirstKDenseReplace {
			for _, s := range []string{"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"} {
				if !pick(lp + s) {
					return missing("dense MLP layer", lp+s)
				}
			}
		} else {
			for _, s := range []string{"mlp.gate.weight", "mlp.gate.e_score_correction_bias",
				"mlp.experts.0.gate_proj.weight", "mlp.experts.0.up_proj.weight", "mlp.experts.0.down_proj.weight",
				"mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight", "mlp.shared_experts.down_proj.weight"} {
				if !pick(lp + s) {
					return missing("MoE layer", lp+s)
				}
			}
		}
	}
	return nil
}

// GLM5NextSession advances a GLM-5.3-Flash text model one token at a time,
// owning the per-layer recurrent (KDA) and cached (DSA) state between steps.
// It is the native decode seam: prefill is N steps, decode continues the same
// session. Snapshot/Restore support prefix reuse.
type GLM5NextSession struct {
	m          *Model
	d          glm5NextDims
	kda        map[int]*glm5NextKDARuntime
	dsa        map[int]*glm5NextDSARuntime
	tokens     int
	lastInput  []float32                       // current collapsed input, read by the indexer's weights_proj
	layerTrace func(layer int, mean []float32) // optional per-layer diagnostic sink
}

type glm5NextKDARuntime struct {
	conv       [][]float32 // per q/k/v branch: rolling last (W-1) mixed inputs
	convFilled int         // inputs seen since reset (caps at W-1)
	S          [][]float32 // per head: [kDim*vDim] recurrent matrix, row-major kDim x vDim
}

type glm5NextDSARuntime struct {
	K      []float32 // appended per token: heads*qkNope
	V      []float32 // appended per token: heads*vHead
	IndexK []float32 // appended per token: indexDim
	Gate   []float32 // appended per token: indexDim (kpool compress gate scores)
}

// NewGLM5NextSession builds a session over an admitted glm5_next model.
func NewGLM5NextSession(m *Model) (*GLM5NextSession, error) {
	if !m.Cfg.GLM5Next {
		return nil, fmt.Errorf("model: glm5_next session on non-glm5_next model")
	}
	d := glm5NextDimsFromCfg(m.Cfg)
	// The pinned checkpoint is nope-only MLA (qk_rope_head_dim=0, mla_use_nope);
	// a rope-carrying variant needs the interleaved-rope seam wired before it
	// can run — refuse it here rather than silently dropping the rotation.
	if d.qkRope != 0 {
		return nil, fmt.Errorf("model: glm5_next native forward requires qk_rope_head_dim=0 (mla_use_nope), got %d", d.qkRope)
	}
	if d.hidden <= 0 || d.layers <= 0 || d.heads <= 0 || d.linearHeads <= 0 || d.linearDim <= 0 {
		return nil, fmt.Errorf("model: glm5_next geometry incomplete: hidden=%d layers=%d heads=%d linear=%dx%d",
			d.hidden, d.layers, d.heads, d.linearHeads, d.linearDim)
	}
	s := &GLM5NextSession{m: m, d: d,
		kda: map[int]*glm5NextKDARuntime{}, dsa: map[int]*glm5NextDSARuntime{}}
	for l := 0; l < s.d.layers; l++ {
		if l%4 != 3 {
			s.kda[l] = &glm5NextKDARuntime{}
		} else {
			s.dsa[l] = &glm5NextDSARuntime{}
		}
	}
	return s, nil
}

// Reset clears all recurrent and cached state (a fresh context).
func (s *GLM5NextSession) Reset() {
	for l, k := range s.kda {
		k.conv = nil
		k.convFilled = 0
		k.S = nil
		s.kda[l] = k
	}
	for l, c := range s.dsa {
		c.K = c.K[:0]
		c.V = c.V[:0]
		c.IndexK = c.IndexK[:0]
		c.Gate = c.Gate[:0]
		s.dsa[l] = c
	}
	s.tokens = 0
}

// TokensProcessed reports how many tokens the session has consumed.
func (s *GLM5NextSession) TokensProcessed() int { return s.tokens }

// Snapshot deep-copies the session state; Restore overwrites in place.
func (s *GLM5NextSession) Snapshot() *GLM5NextSession {
	clone := &GLM5NextSession{m: s.m, d: s.d, kda: map[int]*glm5NextKDARuntime{},
		dsa: map[int]*glm5NextDSARuntime{}, tokens: s.tokens}
	for l, k := range s.kda {
		kc := &glm5NextKDARuntime{convFilled: k.convFilled}
		for _, c := range k.conv {
			kc.conv = append(kc.conv, append([]float32(nil), c...))
		}
		for _, mat := range k.S {
			kc.S = append(kc.S, append([]float32(nil), mat...))
		}
		clone.kda[l] = kc
	}
	for l, c := range s.dsa {
		clone.dsa[l] = &glm5NextDSARuntime{
			K: append([]float32(nil), c.K...), V: append([]float32(nil), c.V...),
			IndexK: append([]float32(nil), c.IndexK...), Gate: append([]float32(nil), c.Gate...),
		}
	}
	return clone
}

// Restore overwrites the receiver's state from snap (same model + geometry).
func (s *GLM5NextSession) Restore(snap *GLM5NextSession) {
	if s == nil || snap == nil || s.m != snap.m {
		return
	}
	s.tokens = snap.tokens
	for l := 0; l < s.d.layers; l++ {
		if l%4 != 3 {
			src, ok := snap.kda[l]
			if !ok {
				s.kda[l] = &glm5NextKDARuntime{}
				continue
			}
			dst := s.kda[l]
			if dst == nil {
				dst = &glm5NextKDARuntime{}
				s.kda[l] = dst
			}
			dst.convFilled = src.convFilled
			dst.conv = nil
			for _, c := range src.conv {
				dst.conv = append(dst.conv, append([]float32(nil), c...))
			}
			dst.S = nil
			for _, mat := range src.S {
				dst.S = append(dst.S, append([]float32(nil), mat...))
			}
		} else {
			src, ok := snap.dsa[l]
			if !ok {
				s.dsa[l] = &glm5NextDSARuntime{}
				continue
			}
			s.dsa[l] = &glm5NextDSARuntime{
				K: append([]float32(nil), src.K...), V: append([]float32(nil), src.V...),
				IndexK: append([]float32(nil), src.IndexK...), Gate: append([]float32(nil), src.Gate...),
			}
		}
	}
}

// Step advances one token (already-embedded input row) through all layers and
// returns the final stream-mean hidden the logits tail consumes.
func (s *GLM5NextSession) Step(embed []float32) []float32 {
	d := s.d
	hc := d.hcMult
	streams := make([][]float32, hc)
	for h := range streams {
		streams[h] = append([]float32(nil), embed...)
	}
	if s.layerTrace != nil {
		mean := make([]float32, d.hidden)
		copy(mean, embed)
		s.layerTrace(-1, mean)
	}
	for l := 0; l < d.layers; l++ {
		// --- attention site ---
		residual := make([][]float32, hc)
		for h := range streams {
			residual[h] = streams[h]
		}
		post, comb, collapsed := s.hcProject(streams, l, true)
		xn := s.norm(collapsed, fmt.Sprintf("layers.%d.input_layernorm.weight", l))
		var attnOut []float32
		if l%4 != 3 {
			attnOut = s.kdaLayer(l, xn)
		} else {
			attnOut = s.dsaLayer(l, xn)
		}
		streams = s.hcCombine(residual, post, comb, attnOut)

		// --- MLP site ---
		residual = make([][]float32, hc)
		for h := range streams {
			residual[h] = streams[h]
		}
		post, comb, collapsed = s.hcProject(streams, l, false)
		xn = s.norm(collapsed, fmt.Sprintf("layers.%d.post_attention_layernorm.weight", l))
		var mlpOut []float32
		if l < d.firstDense {
			mlpOut = s.denseMLP(l, xn)
		} else {
			mlpOut = s.moeMLP(l, xn)
		}
		streams = s.hcCombine(residual, post, comb, mlpOut)

		if s.layerTrace != nil {
			mean := make([]float32, d.hidden)
			for _, st := range streams {
				for i, v := range st {
					mean[i] += v / float32(hc)
				}
			}
			s.layerTrace(l, mean)
		}
	}
	// hc_head: unweighted mean over the streams.
	out := make([]float32, d.hidden)
	for _, st := range streams {
		for i, v := range st {
			out[i] += v / float32(hc)
		}
	}
	s.tokens++
	return out
}

// Logits runs the final norm + LM head over a Step() output.
func (s *GLM5NextSession) Logits(hidden []float32) []float32 {
	normW := glm5NextTensor(s.m, "norm.weight")
	xf := rmsnorm(hidden, normW, float32(s.d.eps))
	mat := residentKernel{s.m}
	head := "lm_head.weight"
	if !s.m.has(head) {
		head = s.m.headName()
	}
	return mat.mul(head, mat.prep(xf), s.d.vocab, s.d.hidden)
}

// hcProject runs one mHC site: pre-collapse weights, post placement weights and
// the Sinkhorn-projected stream combiner, plus the collapsed input the sublayer
// consumes. attnSite selects the hc_attn_* vs hc_ffn_* parameter set.
func (s *GLM5NextSession) hcProject(streams [][]float32, layer int, attnSite bool) (post []float32, comb [][]float32, collapsed []float32) {
	d := s.d
	hc := d.hcMult
	hidden := d.hidden
	kind := "ffn"
	if attnSite {
		kind = "attn"
	}
	fn := glm5NextTensor(s.m, fmt.Sprintf("layers.%d.hc_%s_fn", layer, kind))
	base := glm5NextTensor(s.m, fmt.Sprintf("layers.%d.hc_%s_base", layer, kind))
	scale := glm5NextTensor(s.m, fmt.Sprintf("layers.%d.hc_%s_scale", layer, kind))
	mix := (2 + hc) * hc
	flat := make([]float32, 0, hc*hidden)
	for _, st := range streams {
		flat = append(flat, st...)
	}
	// Unweighted RMSNorm over the flattened stream vector (no learnable weight).
	var ss float64
	for _, v := range flat {
		ss += float64(v) * float64(v)
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(len(flat))+d.hcEps))
	for i := range flat {
		flat[i] *= inv
	}
	logits := glm5MatVec(fn, flat, mix, hc*hidden)
	pre := make([]float32, hc)
	post = make([]float32, hc)
	for h := 0; h < hc; h++ {
		pre[h] = sigmoidf(logits[h]*float32(scale[0])+base[h]) + float32(d.hcEps)
		post[h] = 2 * sigmoidf(logits[hc+h]*float32(scale[1])+base[hc+h])
	}
	comb = make([][]float32, hc)
	for h := 0; h < hc; h++ {
		row := make([]float32, hc)
		for j := 0; j < hc; j++ {
			row[j] = logits[2*hc+h*hc+j]*float32(scale[2]) + base[2*hc+h*hc+j]
		}
		sm := glm5Softmax(row)
		for j := range sm {
			sm[j] += float32(d.hcEps)
		}
		comb[h] = sm
	}
	// Column-normalize, then alternate row/column normalizes for the remaining
	// Sinkhorn iterations (each division re-adds hc_eps, per the reference).
	colNorm := func() {
		for j := 0; j < hc; j++ {
			sum := float32(0)
			for h := 0; h < hc; h++ {
				sum += comb[h][j]
			}
			inv := float32(1) / (sum + float32(d.hcEps))
			for h := 0; h < hc; h++ {
				comb[h][j] *= inv
			}
		}
	}
	rowNorm := func() {
		for h := 0; h < hc; h++ {
			sum := float32(0)
			for j := 0; j < hc; j++ {
				sum += comb[h][j]
			}
			inv := float32(1) / (sum + float32(d.hcEps))
			for j := 0; j < hc; j++ {
				comb[h][j] *= inv
			}
		}
	}
	colNorm()
	for it := 1; it < d.hcSinkhorn; it++ {
		rowNorm()
		colNorm()
	}
	collapsed = make([]float32, hidden)
	for h := 0; h < hc; h++ {
		for i := 0; i < hidden; i++ {
			collapsed[i] += pre[h] * streams[h][i]
		}
	}
	return post, comb, collapsed
}

// hcCombine folds a sublayer output back into the residual streams:
// new[h] = post[h]*out + sum_j comb[j][h]*residual[j].
func (s *GLM5NextSession) hcCombine(residual [][]float32, post []float32, comb [][]float32, out []float32) [][]float32 {
	hc := len(residual)
	next := make([][]float32, hc)
	for h := 0; h < hc; h++ {
		row := append([]float32(nil), out...)
		for i := range row {
			row[i] *= post[h]
		}
		for j := 0; j < hc; j++ {
			w := comb[j][h]
			if w == 0 {
				continue
			}
			for i := range row {
				row[i] += w * residual[j][i]
			}
		}
		next[h] = row
	}
	return next
}

// norm applies the checkpoint RMSNorm weight for a layer-local suffix.
func (s *GLM5NextSession) norm(x []float32, suffix string) []float32 {
	return rmsnorm(x, glm5NextTensor(s.m, suffix), float32(s.d.eps))
}

// kdaLayer runs one KDA linear-attention layer for the current token.
func (s *GLM5NextSession) kdaLayer(l int, x []float32) []float32 {
	d := s.d
	lp := fmt.Sprintf("layers.%d.self_attn.", l)
	qkvDim := d.linearHeads * d.linearDim
	rt := s.kda[l]

	qRaw := glm5MatVec(glm5NextTensor(s.m, lp+"q_proj.weight"), x, qkvDim, d.hidden)
	kRaw := glm5MatVec(glm5NextTensor(s.m, lp+"k_proj.weight"), x, qkvDim, d.hidden)
	vRaw := glm5MatVec(glm5NextTensor(s.m, lp+"v_proj.weight"), x, qkvDim, d.hidden)

	// Causal depthwise conv (kernel W, silu) per branch; decode shifts one input in.
	q := s.convStep(rt, 0, qRaw, glm5NextTensor(s.m, lp+"q_conv1d.weight"))
	k := s.convStep(rt, 1, kRaw, glm5NextTensor(s.m, lp+"k_conv1d.weight"))
	v := s.convStep(rt, 2, vRaw, glm5NextTensor(s.m, lp+"v_conv1d.weight"))

	// Forget gate: f_b(f_a(x)) + dt_bias, decay = lower_bound*sigmoid(exp(A_log)*g).
	fa := glm5MatVec(glm5NextTensor(s.m, lp+"f_a_proj.weight"), x, d.linearDim, d.hidden)
	fb := glm5MatVec(glm5NextTensor(s.m, lp+"f_b_proj.weight"), fa, qkvDim, d.linearDim)
	dtBias := glm5NextTensor(s.m, lp+"dt_bias")
	aLog := glm5NextTensor(s.m, lp+"A_log")
	gExp := make([]float32, qkvDim)
	for i := 0; i < qkvDim; i++ {
		g := fb[i] + dtBias[i]
		h := i / d.linearDim
		decayRate := float32(math.Exp(float64(aLog[h])))
		if d.gateLowerBound != 0 {
			gExp[i] = float32(math.Exp(float64(d.gateLowerBound) * float64(sigmoidf(decayRate*g))))
		} else {
			gf := float64(g)
			sp := gf
			if gf <= 20 {
				sp = math.Log(1 + math.Exp(gf))
			}
			gExp[i] = float32(-float64(decayRate) * sp)
		}
	}
	// beta = sigmoid(b_proj(x)) per head.
	beta := glm5MatVec(glm5NextTensor(s.m, lp+"b_proj.weight"), x, d.linearHeads, d.hidden)
	for i := range beta {
		beta[i] = sigmoidf(beta[i])
	}

	// Delta-rule recurrence per head with l2-normalized q/k.
	if rt.S == nil {
		rt.S = make([][]float32, d.linearHeads)
	}
	out := make([]float32, qkvDim)
	dk, dv := d.linearDim, d.linearDim
	scale := float32(1 / math.Sqrt(float64(dk)))
	for h := 0; h < d.linearHeads; h++ {
		qh := glm5L2Norm(q[h*dk : (h+1)*dk])
		kh := glm5L2Norm(k[h*dk : (h+1)*dk])
		vh := v[h*dk : (h+1)*dk]
		for i := range qh {
			qh[i] *= scale
		}
		mat := rt.S[h]
		if mat == nil {
			mat = make([]float32, dk*dv)
			rt.S[h] = mat
		}
		for a := 0; a < dk; a++ {
			ga := gExp[h*dk+a]
			if ga != 1 {
				row := mat[a*dv : (a+1)*dv]
				for b := 0; b < dv; b++ {
					row[b] *= ga
				}
			}
		}
		kvMem := make([]float32, dv)
		for a := 0; a < dk; a++ {
			ka := kh[a]
			if ka == 0 {
				continue
			}
			row := mat[a*dv : (a+1)*dv]
			for b := 0; b < dv; b++ {
				kvMem[b] += row[b] * ka
			}
		}
		for b := 0; b < dv; b++ {
			kvMem[b] = (vh[b] - kvMem[b]) * beta[h]
		}
		for a := 0; a < dk; a++ {
			ka := kh[a]
			if ka == 0 {
				continue
			}
			row := mat[a*dv : (a+1)*dv]
			for b := 0; b < dv; b++ {
				row[b] += ka * kvMem[b]
			}
		}
		dst := out[h*dk : (h+1)*dk]
		for a := 0; a < dk; a++ {
			qa := qh[a]
			if qa == 0 {
				continue
			}
			row := mat[a*dv : (a+1)*dv]
			for b := 0; b < dv; b++ {
				dst[b] += row[b] * qa
			}
		}
	}

	// Gated output norm + o_proj.
	ga := glm5MatVec(glm5NextTensor(s.m, lp+"g_a_proj.weight"), x, d.linearDim, d.hidden)
	gb := glm5MatVec(glm5NextTensor(s.m, lp+"g_b_proj.weight"), ga, qkvDim, d.linearDim)
	oNormW := glm5NextTensor(s.m, lp+"o_norm.weight")
	normed := make([]float32, qkvDim)
	for h := 0; h < d.linearHeads; h++ {
		n := rmsnorm(out[h*dk:(h+1)*dk], oNormW, float32(d.eps))
		for j := 0; j < dk; j++ {
			normed[h*dk+j] = n[j] * sigmoidf(gb[h*dk+j])
		}
	}
	return glm5MatVec(glm5NextTensor(s.m, lp+"o_proj.weight"), normed, d.hidden, qkvDim)
}

// convStep runs the causal depthwise short conv for one branch, shifting the
// current mixed input into the rolling state, and applies the silu activation.
func (s *GLM5NextSession) convStep(rt *glm5NextKDARuntime, branch int, in []float32, weight []float32) []float32 {
	w := s.d.convWindow
	qkvDim := len(in)
	if rt.conv == nil {
		rt.conv = make([][]float32, 3)
	}
	st := rt.conv[branch]
	have := len(st) / qkvDim
	out := make([]float32, qkvDim)
	for c := 0; c < qkvDim; c++ {
		var sum float32
		wBase := c * w
		// st holds the last `have` inputs, most recent last: input from k steps
		// ago is st[have-k]; missing history reads as the left zero pad.
		for u := 0; u < w-1; u++ {
			k := w - 1 - u
			if k > have {
				continue
			}
			sum += weight[wBase+u] * st[(have-k)*qkvDim+c]
		}
		sum += weight[wBase+w-1] * in[c]
		out[c] = silu(sum)
	}
	if w > 1 {
		st = append(st, append([]float32(nil), in...)...)
		if n := len(st) / qkvDim; n > w-1 {
			st = st[(n-(w-1))*qkvDim:]
		}
		rt.conv[branch] = st
		rt.convFilled++
	}
	return out
}

// dsaLayer runs one DSA sparse-attention layer for the current token: nope-only
// MLA projections, the k-pool indexer over the cached keys, and sparse attention
// restricted to the selected token indices.
func (s *GLM5NextSession) dsaLayer(l int, x []float32) []float32 {
	d := s.d
	lp := fmt.Sprintf("layers.%d.self_attn.", l)
	rt := s.dsa[l]

	qResid := rmsnorm(glm5MatVec(glm5NextTensor(s.m, lp+"q_a_proj.weight"), x, d.qLora, d.hidden),
		glm5NextTensor(s.m, lp+"q_a_layernorm.weight"), float32(d.eps))
	qFlat := glm5MatVec(glm5NextTensor(s.m, lp+"q_b_proj.weight"), qResid, d.heads*d.qkNope, d.qLora)

	kv := glm5MatVec(glm5NextTensor(s.m, lp+"kv_a_proj_with_mqa.weight"), x, d.kvLora+d.qkRope, d.hidden)
	kLat := rmsnorm(kv[:d.kvLora], glm5NextTensor(s.m, lp+"kv_a_layernorm.weight"), float32(d.eps))
	kvFlat := glm5MatVec(glm5NextTensor(s.m, lp+"kv_b_proj.weight"), kLat, d.heads*(d.qkNope+d.vHead), d.kvLora)

	kRow := make([]float32, d.heads*d.qkNope)
	vRow := make([]float32, d.heads*d.vHead)
	for h := 0; h < d.heads; h++ {
		base := h * (d.qkNope + d.vHead)
		copy(kRow[h*d.qkNope:(h+1)*d.qkNope], kvFlat[base:base+d.qkNope])
		copy(vRow[h*d.vHead:(h+1)*d.vHead], kvFlat[base+d.qkNope:base+d.qkNope+d.vHead])
	}
	rt.K = append(rt.K, kRow...)
	rt.V = append(rt.V, vRow...)

	// Indexer state: pooled-key scoring over cached compressed keys.
	qIdx := glm5MatVec(glm5NextTensor(s.m, lp+"indexer.wq_b.weight"), qResid, d.indexHeads*d.indexDim, d.qLora)
	kIdxRaw := glm5MatVec(glm5NextTensor(s.m, lp+"indexer.wk.weight"), x, d.indexDim, d.hidden)
	kIdx := glm5LayerNorm(kIdxRaw,
		glm5NextTensor(s.m, lp+"indexer.k_norm.weight"),
		glm5NextTensor(s.m, lp+"indexer.k_norm.bias"), 1e-6)
	gate := glm5MatVec(glm5NextTensor(s.m, lp+"indexer.index_kpool_compress_gate"), x, d.indexDim, d.hidden)
	rt.IndexK = append(rt.IndexK, kIdx...)
	rt.Gate = append(rt.Gate, gate...)
	tokens := len(rt.IndexK) / d.indexDim

	s.lastInput = x
	topk := s.dsaSelect(l, rt, qIdx, tokens)
	attn := glm5DSASparseAttend(qFlat, rt.K, rt.V, topk, tokens, d.heads, d.qkNope+d.qkRope, d.vHead)
	return glm5MatVec(glm5NextTensor(s.m, lp+"o_proj.weight"), attn, d.hidden, d.heads*d.vHead)
}

// dsaSelect scores pooled keys and returns the selected token indices for the
// current query (length indexTopk + kpool-1 when the tail is always selected).
func (s *GLM5NextSession) dsaSelect(l int, rt *glm5NextDSARuntime, qIdx []float32, tokens int) []int {
	d := s.d
	kp := d.indexKpool
	dim := d.indexDim
	ape := glm5NextTensor(s.m, fmt.Sprintf("layers.%d.self_attn.indexer.index_kpool_compress_ape", l))
	wproj := glm5NextTensor(s.m, fmt.Sprintf("layers.%d.self_attn.indexer.weights_proj.weight", l))

	nPools := tokens / kp // only complete pools are selectable
	if nPools == 0 {
		// Incomplete first pool (tokens < kp): the reference drops the
		// incomplete pool and attends the always-selected tail, which for
		// tokens <= kp-1 is every visible token 0..t.
		out := make([]int, 0, tokens+d.indexTopk)
		for i := 0; i < tokens; i++ {
			out = append(out, i)
		}
		target := d.indexTopk
		if d.indexAlwaysTail {
			target += kp - 1
		}
		for len(out) < target {
			out = append(out, -1)
		}
		return out[:target]
	}
	poolKeys := make([][]float32, nPools)
	for p := 0; p < nPools; p++ {
		keys := make([]float32, dim)
		// Weighted average over the pool's tokens: softmax over the dim axis of
		// (gate_i[j] + ape[j]) per token i, then sum prob_i[j]*kIdx_i[j].
		for i := p * kp; i < (p+1)*kp; i++ {
			logits := make([]float32, dim)
			for j := 0; j < dim; j++ {
				logits[j] = rt.Gate[i*dim+j] + ape[j]
			}
			probs := glm5Softmax(logits)
			for j := 0; j < dim; j++ {
				keys[j] += probs[j] * rt.IndexK[i*dim+j]
			}
		}
		poolKeys[p] = keys
	}

	// Per-head ReLU scores against pooled keys, head-weighted sum per query.
	// qIdx is the CURRENT token's indexer query: [indexHeads x indexDim].
	t := tokens - 1
	if len(qIdx) < d.indexHeads*d.indexDim {
		return []int{t}
	}
	qv := qIdx[:d.indexHeads*d.indexDim]
	scale := float32(1 / math.Sqrt(float64(dim)))
	wv := glm5MatVec(wproj, s.lastInput, d.indexHeads, d.hidden)
	inv := float32(1 / math.Sqrt(float64(d.indexHeads)))
	scores := make([][]float32, d.indexHeads)
	for h := 0; h < d.indexHeads; h++ {
		row := make([]float32, nPools)
		qh := qv[h*dim : (h+1)*dim]
		for p := 0; p < nPools; p++ {
			var sum float32
			pk := poolKeys[p]
			for j := 0; j < dim; j++ {
				sum += qh[j] * pk[j]
			}
			row[p] = float32(math.Max(float64(sum*scale), 0))
		}
		scores[h] = row
	}
	idxScores := make([]float32, nPools)
	for p := 0; p < nPools; p++ {
		var sum float32
		for h := 0; h < d.indexHeads; h++ {
			sum += wv[h] * inv * scores[h][p]
		}
		idxScores[p] = sum
	}
	// Causal visibility: a pool is selectable only when its LAST token is
	// visible to the query (HF valid_candidates = pool_visible & pool_valid).
	// The attention gather re-checks per token; this keeps the SELECTION
	// identical to the reference instead of wasting slots on future pools.
	for p := 0; p < nPools; p++ {
		if p*kp+kp-1 > tokens-1 {
			idxScores[p] = -float32(math.MaxFloat32)
		}
	}
	selectK := d.indexTopk / kp
	if selectK > nPools {
		selectK = nPools
	}
	order := make([]int, nPools)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return idxScores[order[a]] > idxScores[order[b]] })
	selected := order[:selectK]
	sort.Ints(selected)

	out := make([]int, 0, d.indexTopk+2*kp)
	for _, p := range selected {
		for i := 0; i < kp; i++ {
			out = append(out, p*kp+i)
		}
	}
	if d.indexAlwaysTail {
		visible := t + 1
		tailCount := visible % kp
		if tailCount > 0 {
			tailStart := visible - tailCount
			for i := 0; i < kp-1; i++ {
				if i < tailCount && tailStart+i < tokens {
					out = append(out, tailStart+i)
				}
			}
		}
	}
	target := d.indexTopk
	if d.indexAlwaysTail {
		target += kp - 1
	}
	for len(out) < target {
		out = append(out, -1)
	}
	return out[:target]
}

// glm5DSASparseAttend computes the sparse attention output for the current
// query over the selected token indices (softmax restricted to valid entries).
func glm5DSASparseAttend(qFlat, K, V []float32, topk []int, tokens, heads, qkDim, vHead int) []float32 {
	out := make([]float32, heads*vHead)
	scale := float32(1 / math.Sqrt(float64(qkDim)))
	t := tokens - 1
	// qFlat is the CURRENT token's query row: [heads x qkDim].
	for h := 0; h < heads; h++ {
		qh := qFlat[h*qkDim : (h+1)*qkDim]
		sel := make([]float32, 0, len(topk))
		pos := make([]int32, 0, len(topk))
		for _, idx := range topk {
			if idx < 0 || idx > t {
				continue
			}
			var sum float32
			kOff := idx*heads*qkDim + h*qkDim
			for j := 0; j < qkDim; j++ {
				sum += qh[j] * K[kOff+j]
			}
			sel = append(sel, sum*scale)
			pos = append(pos, int32(idx))
		}
		if len(sel) == 0 {
			continue
		}
		wts := glm5Softmax(sel)
		dst := out[h*vHead : (h+1)*vHead]
		for i, idx := range pos {
			vOff := int(idx)*heads*vHead + h*vHead
			w := wts[i]
			for j := 0; j < vHead; j++ {
				dst[j] += w * V[vOff+j]
			}
		}
	}
	return out
}

// denseMLP runs the clamped SwiGLU dense FFN (layers 0..2).
func (s *GLM5NextSession) denseMLP(l int, x []float32) []float32 {
	lp := fmt.Sprintf("layers.%d.mlp.", l)
	return glm5ClampedSwiglu(x,
		glm5NextTensor(s.m, lp+"gate_proj.weight"),
		glm5NextTensor(s.m, lp+"up_proj.weight"),
		glm5NextTensor(s.m, lp+"down_proj.weight"),
		s.d.inter, s.d.hidden, s.d.swigluLimit)
}

// moeMLP runs the 288-expert top-8 sigmoid-routed MoE plus the shared expert.
func (s *GLM5NextSession) moeMLP(l int, x []float32) []float32 {
	d := s.d
	lp := fmt.Sprintf("layers.%d.mlp.", l)
	routerW := glm5NextTensor(s.m, lp+"gate.weight")
	bias := glm5NextTensor(s.m, lp+"gate.e_score_correction_bias")

	logits := make([]float32, d.numExperts)
	for e := 0; e < d.numExperts; e++ {
		row := routerW[e*d.hidden : (e+1)*d.hidden]
		var sum float32
		for j, v := range x {
			sum += row[j] * v
		}
		logits[e] = sum
	}
	scores := make([]float32, d.numExperts)
	choice := make([]float32, d.numExperts)
	for e := range logits {
		scores[e] = sigmoidf(logits[e])
		choice[e] = scores[e] + bias[e]
	}
	order := make([]int, d.numExperts)
	for i := range order {
		order[i] = i
	}
	// noaux_tc grouped selection; n_group==1 degenerates to the plain top-k
	// (the single group always wins its topk_group slot).
	if d.nGroup > 1 && d.numExperts%d.nGroup == 0 {
		g := d.nGroup
		per := d.numExperts / g
		groupScore := make([]float32, g)
		for gi := 0; gi < g; gi++ {
			best, second := float32(-math.MaxFloat32), float32(-math.MaxFloat32)
			for e := gi * per; e < (gi+1)*per; e++ {
				if choice[e] > best {
					second = best
					best = choice[e]
				} else if choice[e] > second {
					second = choice[e]
				}
			}
			groupScore[gi] = best + second
		}
		gorder := make([]int, g)
		for i := range gorder {
			gorder[i] = i
		}
		sort.SliceStable(gorder, func(a, b int) bool { return groupScore[gorder[a]] > groupScore[gorder[b]] })
		keep := make([]bool, d.numExperts)
		for i := 0; i < d.topKGroup && i < g; i++ {
			for e := gorder[i] * per; e < (gorder[i]+1)*per; e++ {
				keep[e] = true
			}
		}
		filtered := order[:0]
		for _, e := range order {
			if keep[e] {
				filtered = append(filtered, e)
			}
		}
		order = filtered
	}
	sort.SliceStable(order, func(a, b int) bool { return choice[order[a]] > choice[order[b]] })
	if len(order) > d.topK {
		order = order[:d.topK]
	}
	scaling := float32(d.routedScaling)
	if d.normTopk {
		wsum := float32(0)
		for _, e := range order {
			wsum += scores[e]
		}
		scaling *= float32(1) / (wsum + 1e-20)
	}

	out := make([]float32, d.hidden)
	prefix := lp + "experts."
	for _, e := range order {
		ep := fmt.Sprintf("%s%d.", prefix, e)
		contrib := glm5ClampedSwiglu(x,
			glm5NextTensor(s.m, ep+"gate_proj.weight"),
			glm5NextTensor(s.m, ep+"up_proj.weight"),
			glm5NextTensor(s.m, ep+"down_proj.weight"),
			d.moeInter, d.hidden, d.swigluLimit)
		w := scores[e] * scaling
		for i := range out {
			out[i] += w * contrib[i]
		}
	}
	shared := glm5ClampedSwiglu(x,
		glm5NextTensor(s.m, lp+"shared_experts.gate_proj.weight"),
		glm5NextTensor(s.m, lp+"shared_experts.up_proj.weight"),
		glm5NextTensor(s.m, lp+"shared_experts.down_proj.weight"),
		d.moeInter*d.nShared, d.hidden, d.swigluLimit)
	for i := range out {
		out[i] += shared[i]
	}
	return out
}

// forwardGLM5Next is the Model-level entry: full prefill over already-embedded
// rows, collecting per-layer collapsed means and per-position logits.
func (m *Model) forwardGLM5Next(x [][]float32) *Activations {
	s, err := NewGLM5NextSession(m)
	if err != nil {
		panic(err)
	}
	cfg := m.Cfg
	act := &Activations{Seq: len(x)}
	trace := make([][][]float32, len(x))
	act.Logits = make([][]float32, len(x))
	for t, row := range x {
		rows := make([][]float32, 0, cfg.NumLayers+1)
		s.layerTrace = func(layer int, mean []float32) {
			rows = append(rows, append([]float32(nil), mean...))
		}
		hidden := s.Step(row)
		act.Logits[t] = s.Logits(hidden)
		trace[t] = rows
	}
	s.layerTrace = nil
	if len(trace) > 0 && len(trace[0]) == cfg.NumLayers+1 {
		act.Hidden = make([][]float32, 0, cfg.NumLayers+1)
		for l := 0; l <= cfg.NumLayers; l++ {
			flat := make([]float32, 0, len(x)*cfg.HiddenSize)
			for t := range x {
				flat = append(flat, trace[t][l]...)
			}
			act.Hidden = append(act.Hidden, flat)
		}
	}
	return act
}

// glm5NextEmbed looks up token embeddings through the glm5 prefix fallbacks.
func (m *Model) glm5NextEmbed(ids []int) [][]float32 {
	embed := glm5NextTensor(m, "embed_tokens.weight")
	H := m.Cfg.HiddenSize
	x := make([][]float32, len(ids))
	for t, id := range ids {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x[t], m.Cfg)
	}
	return x
}

func glm5Softmax(xs []float32) []float32 {
	out := make([]float32, len(xs))
	if len(xs) == 0 {
		return out
	}
	max := xs[0]
	for _, v := range xs[1:] {
		if v > max {
			max = v
		}
	}
	var sum float64
	for i, v := range xs {
		e := math.Exp(float64(v - max))
		out[i] = float32(e)
		sum += e
	}
	inv := float32(1 / sum)
	for i := range out {
		out[i] *= inv
	}
	return out
}

func glm5L2Norm(x []float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(ss+1e-6))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv
	}
	return out
}

func glm5LayerNorm(x, w, b []float32, eps float64) []float32 {
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	var vari float64
	for _, v := range x {
		dv := float64(v) - mean
		vari += dv * dv
	}
	vari /= float64(len(x))
	inv := float32(1 / math.Sqrt(vari+eps))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = (float32(float64(v)-mean))*inv*w[i] + b[i]
	}
	return out
}

func glm5MatVec(w, x []float32, outDim, inDim int) []float32 {
	out := make([]float32, outDim)
	if outDim == 0 || inDim == 0 || len(w) < outDim*inDim {
		return out
	}
	for i := 0; i < outDim; i++ {
		row := w[i*inDim : (i+1)*inDim]
		var sum float32
		for j := 0; j < inDim; j++ {
			sum += row[j] * x[j]
		}
		out[i] = sum
	}
	return out
}

// glm5ClampedSwiglu is the GLM-5.3-Flash MLP activation: gate clamped to
// (−inf, limit], up clamped to [−limit, limit], then silu(gate)*up, down-proj.
func glm5ClampedSwiglu(x, gateW, upW, downW []float32, inter, hidden int, limit float64) []float32 {
	gate := glm5MatVec(gateW, x, inter, hidden)
	up := glm5MatVec(upW, x, inter, hidden)
	for i := range gate {
		if float64(gate[i]) > limit {
			gate[i] = float32(limit)
		}
		if float64(up[i]) > limit {
			up[i] = float32(limit)
		}
		if float64(up[i]) < -limit {
			up[i] = float32(-limit)
		}
		gate[i] = silu(gate[i]) * up[i]
	}
	return glm5MatVec(downW, gate, hidden, inter)
}
