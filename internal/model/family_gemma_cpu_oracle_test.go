package model

// family_gemma_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU numeric
// oracle for the Gemma2/Gemma3 family (#1271 Lane 1, support-maturity epic #1243).
// Companion to family_cpu_oracle_test.go, which states the doctrine and carries the
// OLMo2 (PostNorm) and Qwen2/3 (PreNorm) precedents; Gemma is the first SandwichNorm
// family to get a reference, so the block dataflow below is transcribed rather than
// adapted from either precedent.
//
// Independence discipline (family_cpu_oracle_test.go:13-24): the reference is a plain
// scalar transcription of HuggingFace transformers/models/gemma2/{modeling,
// configuration}_gemma2.py and .../gemma3/{modeling,configuration}_gemma3.py. It
// reuses NONE of the production machinery — tensors come straight out of the manifest
// bytes via cpuOracleTensor, every matmul / norm / softmax / GELU / soft-cap is a naive
// in-order scalar loop, and the block dataflow, the local/global cadence, the per-layer
// RoPE base and the attention scale are hardcoded from the HF RULES in gemmaOracleSpec
// rather than read back out of cfg.BlockTopology / cfg.Window / cfg.RopeThetaPerLayer /
// cfg.attnScale(). That last point is what makes this a witness for the DERIVATION and
// not just for the kernel: if config.go lowers a published Gemma config to the wrong
// per-layer window or the wrong rope base, the reference still uses the HF rule and the
// two sides diverge.
//
// What Gemma actually does, transcribed from the HF source (not from the family's
// reputation):
//
//   - SANDWICH NORM (Gemma2DecoderLayer.forward / Gemma3DecoderLayer.forward). BOTH
//     sub-layers carry a norm before AND after the body, and all four norms are
//     DISTINCT tensors:
//     x = x + post_attention_layernorm(attn(input_layernorm(x)))
//     x = x + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))
//     Note the naming trap: post_attention_layernorm is the POST norm of the ATTENTION
//     sub-layer here, not (as in Llama) the pre-norm of the MLP. A reference that used
//     it as the MLP pre-norm, or that dropped either post-norm, diverges by O(0.1..1).
//   - NORM GAIN (1+w) (Gemma2RMSNorm/Gemma3RMSNorm.forward: `output * (1.0 + weight)`,
//     computed in fp32 and cast back). The published weights initialize at ~0, so the
//     fixture below draws them in (-0.25, 0.25): a kernel that applied a plain `w` gain
//     would be off by roughly a factor of four per norm, four norms per layer.
//   - EMBEDDING SCALE. Gemma multiplies the embedding row by
//     `normalizer = torch.tensor(config.hidden_size**0.5, dtype=hidden_states.dtype)`
//     — sqrt(hidden_size), NOT sqrt(head_dim) and NOT 1/sqrt(hidden_size).
//   - GeGLU with gelu_pytorch_tanh: down(gelu_tanh(gate(x)) * up(x)). The activation is
//     the TANH APPROXIMATION 0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z^3))), not SiLU and
//     not the exact erf GELU. hidden_activation (not hidden_act) is the config key.
//   - ALTERNATING LOCAL/GLOBAL ATTENTION. Gemma2: `is_sliding = not bool(layer_idx % 2)`
//     (Gemma2DecoderLayer.__init__) — equivalently the layer_types list newer configs
//     carry, `"sliding_attention" if bool((i+1) % 2) else "full_attention"` — so EVEN
//     layers are windowed and ODD layers are FULL causal. Gemma3 generalizes it to
//     `(i+1) % sliding_window_pattern`, pattern 6, i.e. 5 local layers then 1 global.
//     The sliding mask keeps key k for query q iff `q - k < window`, i.e. the window is
//     `window` positions INCLUSIVE of the query.
//   - TWO ROPE BASES (Gemma3 only): rope_local_base_freq (10000) on the sliding layers,
//     rope_theta (1e6) on the full-attention layers. Gemma2 uses one rope_theta for both.
//   - SOFT-CAPS (Gemma2 only): attn_logit_softcapping applied to the raw scores BEFORE
//     the causal mask and the softmax, and final_logit_softcapping applied AFTER the LM
//     head, both as `c * tanh(z/c)`. Gemma3 dropped both (they are absent from its config).
//   - QUERY PRE-ATTENTION SCALAR: `scaling = config.query_pre_attn_scalar**-0.5`, which
//     is NOT always head_dim**-0.5 — Gemma2-27B uses 144 against head_dim 128 and
//     Gemma3-27B uses 168 against head_dim 128. The fixtures below therefore set a
//     query_pre_attn_scalar that DIFFERS from head_dim on both lineages, so a kernel that
//     silently fell back to 1/sqrt(head_dim) is caught.
//   - PER-HEAD QK-NORM (Gemma3 only): Gemma3RMSNorm(head_dim) on q and on k, applied
//     after projection and before RoPE, with the same (1+w) gain.
//   - TIED EMBEDDINGS: tie_word_embeddings is true for every published Gemma, so the LM
//     head is the embedding matrix and no lm_head.weight tensor exists.
//
// Fixture scaling note (honest, and load-bearing for non-vacuity): the published cap
// values are 50 (attention) and 30 (logits). At the fixture's weight scale the scores
// are O(0.2) and the logits O(1), so tanh(z/50) == z/50 to ~7 decimal places and BOTH
// caps would be numerically inert — the oracle would stay green with soft-capping
// deleted. The fixtures therefore set caps that are on the same scale as the fixture's
// own activations. This changes only the SCALAR, never the semantics or the placement,
// and TestGemmaCPUNumericOracleAxesAreLive asserts every Gemma-specific axis actually
// moves the compared logits by more than the tolerance in this fixture — so no axis is
// "witnessed" vacuously.

import (
	"encoding/binary"
	"math"
	"testing"
)

// gemmaOracleRMSNorm is Gemma2RMSNorm/Gemma3RMSNorm: x*rsqrt(mean(x^2)+eps) * (1+w).
// The +1 is the whole point — see the header.
func gemmaOracleRMSNorm(x, w []float32, eps float32) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * (1 + w[i])
	}
	return out
}

// gemmaOracleGeluTanh is HF ACT2FN["gelu_pytorch_tanh"] (nn.functional.gelu with
// approximate="tanh"): 0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z^3))).
func gemmaOracleGeluTanh(z float32) float32 {
	z64 := float64(z)
	inner := math.Sqrt(2/math.Pi) * (z64 + 0.044715*z64*z64*z64)
	return float32(0.5 * z64 * (1 + math.Tanh(inner)))
}

// gemmaOracleSoftcap is Gemma2's tanh soft-cap: z -> c*tanh(z/c). c<=0 means "absent"
// (Gemma3), which is the identity.
func gemmaOracleSoftcap(z, c float32) float32 {
	if c <= 0 {
		return z
	}
	return c * float32(math.Tanh(float64(z/c)))
}

// gemmaOracleSpec is the family semantics the reference hardcodes. Every field is
// derived HERE from the published HF rule for the lineage, never read out of Config —
// that is what lets the numeric comparison witness config.go's derivation as well as
// the kernel.
type gemmaOracleSpec struct {
	layers int
	// pattern is HF's sliding_window_pattern: layer l is sliding unless (l+1)%pattern==0.
	// Gemma2's `is_sliding = not bool(layer_idx % 2)` is exactly pattern 2.
	pattern int
	// window is the sliding-window width in positions, INCLUSIVE of the query.
	window     int
	localTheta float64
	fullTheta  float64
	// qpas is query_pre_attn_scalar; the attention scale is qpas**-0.5.
	qpas     int
	attnCap  float32
	logitCap float32
	qkNorm   bool
}

func (s gemmaOracleSpec) isSliding(l int) bool { return (l+1)%s.pattern != 0 }

// windowFor is the per-layer attention span: the sliding width on a local layer, -1
// (unbounded causal) on a global layer.
func (s gemmaOracleSpec) windowFor(l int) int {
	if s.isSliding(l) {
		return s.window
	}
	return -1
}

func (s gemmaOracleSpec) thetaFor(l int) float64 {
	if s.isSliding(l) {
		return s.localTheta
	}
	return s.fullTheta
}

// gemmaOracleIDs is the prompt. len(ids)=7 is deliberately larger than the fixtures'
// window of 3: with a shorter prompt every layer would attend its full prefix anyway
// and the local/global cadence would be numerically invisible.
var gemmaOracleIDs = []int{3, 17, 5, 23, 41, 2, 19}

// gemma2OracleCfg is the tiny Gemma2 fixture config. nH*hd (32) deliberately differs
// from HiddenSize (24) — that is Gemma's real shape (Gemma2-2B: hidden 2304, 8 heads,
// head_dim 256) and it stops a projection-width/hidden-width conflation from cancelling.
// QueryPreAttnScalar (12) differs from HeadDim (8) exactly as 27B's 144 differs from its
// head_dim 128. The caps are scaled to the fixture's activations (see the header note).
// NO layer_types and NO sliding_window_pattern are set: a published Gemma2 config.json
// carries neither, only `sliding_window`, so this is the shape the derivation must handle.
func gemma2OracleCfg() Config {
	return Config{
		HiddenSize:         24,
		NumLayers:          4,
		NumHeads:           4,
		NumKVHeads:         2,
		HeadDim:            8,
		IntermediateSize:   40,
		VocabSize:          53,
		ModelType:          "gemma2",
		Architectures:      []string{"Gemma2ForCausalLM"},
		HiddenActivation:   "gelu_pytorch_tanh",
		RMSNormEps:         1e-6,
		RopeTheta:          10000,
		QueryPreAttnScalar: 12,
		AttnSoftcap:        0.4,
		LogitSoftcap:       0.5,
		TieWordEmbeddings:  true,
	}
}

// gemma3OracleCfg is the tiny Gemma3 fixture config: 6 layers so the pattern-6 cadence
// has exactly one global layer (index 5), the two published rope bases, per-head qk-norm,
// and no soft-caps (Gemma3 dropped them). QueryPreAttnScalar 18 vs HeadDim 8 mirrors
// Gemma3-27B's 168 vs 128.
func gemma3OracleCfg() Config {
	return Config{
		HiddenSize:           24,
		NumLayers:            6,
		NumHeads:             4,
		NumKVHeads:           2,
		HeadDim:              8,
		IntermediateSize:     40,
		VocabSize:            53,
		ModelType:            "gemma3",
		Architectures:        []string{"Gemma3ForCausalLM"},
		HiddenActivation:     "gelu_pytorch_tanh",
		RMSNormEps:           1e-6,
		RopeTheta:            1000000,
		RopeLocalBaseFreq:    10000,
		SlidingWindowPattern: 6,
		QueryPreAttnScalar:   18,
		TieWordEmbeddings:    true,
	}
}

// gemmaOracleWindow is the fixtures' sliding_window, passed the way a real config.json
// passes it (the "sliding_window" key), so the per-layer Window slice has to be DERIVED.
const gemmaOracleWindow = 3

// gemmaOracleSpecFor builds the HF spec for a lineage. This is the transcription of the
// published rule; nothing here consults the derived Config.
func gemmaOracleSpecFor(gemma3 bool) gemmaOracleSpec {
	if gemma3 {
		return gemmaOracleSpec{
			layers:     6,
			pattern:    6, // 5 local : 1 global; layer 5 is the global one
			window:     gemmaOracleWindow,
			localTheta: 10000,
			fullTheta:  1000000,
			qpas:       18,
			qkNorm:     true,
		}
	}
	return gemmaOracleSpec{
		layers:     4,
		pattern:    2, // is_sliding = not bool(layer_idx % 2): even sliding, ODD full
		window:     gemmaOracleWindow,
		localTheta: 10000,
		fullTheta:  10000, // Gemma2 has ONE rope base for both regimes
		qpas:       12,
		attnCap:    0.4,
		logitCap:   0.5,
	}
}

// newGemmaOracleModel builds the fixture on Gemma's REAL tensor roster — all FOUR
// sandwich norms per layer, plus the per-head qk-norms on gemma3 — and loads it through
// newModel with the config decoded the way a config.json would be (deriveConfigAxes over
// a `sliding_window` hint), so the production side runs the whole derivation.
//
// Norm weights are drawn in (-0.25, 0.25): Gemma's published norm weights sit near 0
// because the kernel adds the 1, so this is both faithful AND distinct/non-zero, which
// keeps norm-tensor routing and the (1+w) gain numerically live.
func newGemmaOracleModel(t *testing.T, gemma3 bool) *Model {
	t.Helper()
	cfg := gemma2OracleCfg()
	if gemma3 {
		cfg = gemma3OracleCfg()
	}
	window := gemmaOracleWindow
	if err := cfg.deriveConfigAxes(configJSONHints{SlidingWindow: &window}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize

	type ts = synthTensor
	var tensors []ts
	tensors = append(tensors, ts{"model.embed_tokens.weight", []int{V, H}})
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		tensors = append(tensors,
			ts{p + "input_layernorm.weight", []int{H}},
			ts{p + "self_attn.q_proj.weight", []int{nH * hd, H}},
			ts{p + "self_attn.k_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.v_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.o_proj.weight", []int{H, nH * hd}},
		)
		if gemma3 {
			// Gemma3RMSNorm(head_dim) per head — NOT OLMo2's full-projection-width norm.
			tensors = append(tensors,
				ts{p + "self_attn.q_norm.weight", []int{hd}},
				ts{p + "self_attn.k_norm.weight", []int{hd}},
			)
		}
		tensors = append(tensors,
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{p + "pre_feedforward_layernorm.weight", []int{H}},
			ts{p + "post_feedforward_layernorm.weight", []int{H}},
			ts{p + "mlp.gate_proj.weight", []int{I, H}},
			ts{p + "mlp.up_proj.weight", []int{I, H}},
			ts{p + "mlp.down_proj.weight", []int{H, I}},
		)
	}
	tensors = append(tensors, ts{"model.norm.weight", []int{H}})
	// No lm_head.weight: Gemma ties the head to the embedding.

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		if isCPUOracleNormWeight(name) {
			return 0.25 * next() // (1+w) lands in (0.75, 1.25): non-unit, distinct, well-conditioned
		}
		return synthMatmulFill(name, next)
	})

	m, err := newModel(cfg, man, raw)
	if err != nil {
		t.Fatalf("newModel: %v", err)
	}
	return m
}

// gemmaReference runs the independent Gemma forward: per-position logits for ids under
// spec. Every step is the HF Gemma2/Gemma3 dataflow, hardcoded — NOT routed through
// cfg.BlockTopology, normCfg, ffnFor, cfg.attnScale, cfg.windowForLayer,
// cfg.ropeThetaForLayer or any other production helper.
func gemmaReference(t *testing.T, m *Model, ids []int, spec gemmaOracleSpec) [][]float32 {
	t.Helper()
	cfg := m.Cfg
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	grp := nH / nKV
	eps := float32(cfg.RMSNormEps)
	seq := len(ids)

	embed := cpuOracleTensor(t, m, "model.embed_tokens.weight")
	// GemmaModel.forward: inputs_embeds * tensor(hidden_size**0.5, dtype=embeds.dtype).
	normalizer := float32(math.Sqrt(float64(H)))
	x := make([][]float32, seq)
	for tt, id := range ids {
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		for i := range x[tt] {
			x[tt][i] *= normalizer
		}
	}

	// scaling = query_pre_attn_scalar**-0.5 (NOT head_dim**-0.5 in general).
	scale := float32(1.0 / math.Sqrt(float64(spec.qpas)))

	for l := 0; l < spec.layers; l++ {
		p := layerPrefix(l)
		inLN := cpuOracleTensor(t, m, p+"input_layernorm.weight")
		wq := cpuOracleTensor(t, m, p+"self_attn.q_proj.weight")
		wk := cpuOracleTensor(t, m, p+"self_attn.k_proj.weight")
		wv := cpuOracleTensor(t, m, p+"self_attn.v_proj.weight")
		wo := cpuOracleTensor(t, m, p+"self_attn.o_proj.weight")
		postAttnLN := cpuOracleTensor(t, m, p+"post_attention_layernorm.weight")
		preFFLN := cpuOracleTensor(t, m, p+"pre_feedforward_layernorm.weight")
		postFFLN := cpuOracleTensor(t, m, p+"post_feedforward_layernorm.weight")
		wg := cpuOracleTensor(t, m, p+"mlp.gate_proj.weight")
		wu := cpuOracleTensor(t, m, p+"mlp.up_proj.weight")
		wd := cpuOracleTensor(t, m, p+"mlp.down_proj.weight")

		theta := spec.thetaFor(l)
		W := spec.windowFor(l)

		// --- attention sub-layer, SANDWICH: x += postAttnLN(attn(inLN(x))) ---
		q := make([][]float32, seq)
		k := make([][]float32, seq)
		v := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			xn := gemmaOracleRMSNorm(x[tt], inLN, eps)
			q[tt] = cpuOracleMatVec(wq, xn, nH*hd, H)
			k[tt] = cpuOracleMatVec(wk, xn, nKV*hd, H)
			v[tt] = cpuOracleMatVec(wv, xn, nKV*hd, H)
			if spec.qkNorm {
				// Gemma3Attention: per-head RMSNorm(head_dim) after projection, before RoPE.
				qnw := cpuOracleTensor(t, m, p+"self_attn.q_norm.weight")
				knw := cpuOracleTensor(t, m, p+"self_attn.k_norm.weight")
				for h := 0; h < nH; h++ {
					copy(q[tt][h*hd:(h+1)*hd], gemmaOracleRMSNorm(q[tt][h*hd:(h+1)*hd], qnw, eps))
				}
				for h := 0; h < nKV; h++ {
					copy(k[tt][h*hd:(h+1)*hd], gemmaOracleRMSNorm(k[tt][h*hd:(h+1)*hd], knw, eps))
				}
			}
			for h := 0; h < nH; h++ {
				cpuOracleRope(q[tt][h*hd:(h+1)*hd], tt, hd, theta)
			}
			for h := 0; h < nKV; h++ {
				cpuOracleRope(k[tt][h*hd:(h+1)*hd], tt, hd, theta)
			}
		}
		for tt := 0; tt < seq; tt++ {
			// Sliding mask: key j is kept iff tt-j < window, i.e. j >= tt-window+1.
			lo := 0
			if W >= 0 {
				if lo = tt - W + 1; lo < 0 {
					lo = 0
				}
			}
			concat := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				kvh := h / grp
				qh := q[tt][h*hd : (h+1)*hd]
				scores := make([]float32, tt+1-lo)
				for j := lo; j <= tt; j++ {
					kh := k[j][kvh*hd : (kvh+1)*hd]
					var s float32
					for d := 0; d < hd; d++ {
						s += qh[d] * kh[d]
					}
					// Gemma2 soft-caps the RAW scores, before the mask and the softmax.
					scores[j-lo] = gemmaOracleSoftcap(s*scale, spec.attnCap)
				}
				cpuOracleSoftmax(scores)
				o := concat[h*hd : (h+1)*hd]
				for j := lo; j <= tt; j++ {
					vh := v[j][kvh*hd : (kvh+1)*hd]
					for d := 0; d < hd; d++ {
						o[d] += scores[j-lo] * vh[d]
					}
				}
			}
			attnOut := cpuOracleMatVec(wo, concat, H, nH*hd)
			nout := gemmaOracleRMSNorm(attnOut, postAttnLN, eps)
			for i := 0; i < H; i++ {
				x[tt][i] += nout[i]
			}
		}

		// --- MLP sub-layer, SANDWICH: x += postFFLN(GeGLU(preFFLN(x))) ---
		for tt := 0; tt < seq; tt++ {
			xn := gemmaOracleRMSNorm(x[tt], preFFLN, eps)
			gate := cpuOracleMatVec(wg, xn, I, H)
			up := cpuOracleMatVec(wu, xn, I, H)
			for i := 0; i < I; i++ {
				gate[i] = gemmaOracleGeluTanh(gate[i]) * up[i]
			}
			mlpOut := cpuOracleMatVec(wd, gate, H, I)
			nout := gemmaOracleRMSNorm(mlpOut, postFFLN, eps)
			for i := 0; i < H; i++ {
				x[tt][i] += nout[i]
			}
		}
	}

	norm := cpuOracleTensor(t, m, "model.norm.weight")
	logits := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		xf := gemmaOracleRMSNorm(x[tt], norm, eps)
		row := cpuOracleMatVec(embed, xf, V, H) // tied head: logits = E @ xf
		for i := range row {
			row[i] = gemmaOracleSoftcap(row[i], spec.logitCap)
		}
		logits[tt] = row
	}
	return logits
}

// gemmaOracleAssertDerivedAxes checks that config.go lowered the published config to the
// axes the reference hardcodes. This is the structural half of the witness — it names the
// broken axis directly instead of leaving only a logit delta — but it is NOT the gate:
// the numeric comparison below stands on its own.
func gemmaOracleAssertDerivedAxes(t *testing.T, cfg Config, spec gemmaOracleSpec) {
	t.Helper()
	if cfg.BlockTopology != SandwichNorm {
		t.Fatalf("derived topology = %v, want SandwichNorm", cfg.BlockTopology)
	}
	if !cfg.NormGain1p {
		t.Fatal("derived NormGain1p = false, want true (Gemma RMSNorm scales by 1+weight)")
	}
	if !cfg.ActGeluTanh {
		t.Fatal("derived ActGeluTanh = false, want true (hidden_activation gelu_pytorch_tanh)")
	}
	if want := math.Sqrt(float64(cfg.HiddenSize)); cfg.EmbedScale != want {
		t.Fatalf("derived EmbedScale = %v, want sqrt(hidden_size) = %v", cfg.EmbedScale, want)
	}
	if cfg.QKNorm != spec.qkNorm {
		t.Fatalf("derived QKNorm = %v, want %v", cfg.QKNorm, spec.qkNorm)
	}
	for l := 0; l < spec.layers; l++ {
		if got, want := cfg.windowForLayer(l), spec.windowFor(l); got != want {
			t.Errorf("layer %d: derived window = %d, want %d (HF cadence: layer %d is %s)",
				l, got, want, l, map[bool]string{true: "sliding_attention", false: "full_attention"}[spec.isSliding(l)])
		}
		if got, want := cfg.ropeThetaForLayer(l), spec.thetaFor(l); got != want {
			t.Errorf("layer %d: derived rope theta = %v, want %v", l, got, want)
		}
	}
}

// runGemmaOracle compares the production cacheless Forward AND the cached Prefill/Step
// decode path against the independent reference at every position.
func runGemmaOracle(t *testing.T, gemma3 bool) {
	t.Helper()
	m := newGemmaOracleModel(t, gemma3)
	spec := gemmaOracleSpecFor(gemma3)
	gemmaOracleAssertDerivedAxes(t, m.Cfg, spec)

	ids := gemmaOracleIDs
	ref := gemmaReference(t, m, ids, spec)

	act := m.Forward(ids)
	for tt := range ids {
		if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > cpuOracleTol {
			t.Errorf("Forward logits pos %d: max|delta| vs reference = %.3e (tol %.0e)", tt, d, cpuOracleTol)
		}
	}

	// Cached decode path: Prefill then Step must match the reference at the same
	// positions (the reference is cacheless, so Step(id) at position len(ids) is
	// compared against a reference run over the extended prompt).
	s := m.NewSession()
	pf := s.Prefill(ids)
	if d := cpuOracleMaxAbsDiff(pf, ref[len(ids)-1]); d > cpuOracleTol {
		t.Errorf("Prefill last logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
	next := 11
	st := s.Step(next)
	extRef := gemmaReference(t, m, append(append([]int(nil), ids...), next), spec)
	if d := cpuOracleMaxAbsDiff(st, extRef[len(ids)]); d > cpuOracleTol {
		t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
}

// TestGemmaCPUNumericOracle is the Gemma2/3×cpu M4 witness: one fixture per lineage,
// each gated against the independent HF-semantics reference on the full cacheless
// prefill (every position) and on the cached decode path. If this test reds, the
// honesty fence demotes the cell back to M3 (drop the covmatrix OracleInCI bit with it).
func TestGemmaCPUNumericOracle(t *testing.T) {
	for _, lineage := range []struct {
		name   string
		gemma3 bool
	}{
		{"gemma2", false},
		{"gemma3", true},
	} {
		t.Run(lineage.name, func(t *testing.T) { runGemmaOracle(t, lineage.gemma3) })
	}
}

// gemmaOracleAssertAxisLive proves one Gemma-specific axis is not numerically inert in
// this fixture: the reference under the HF spec must differ from the reference under a
// deliberately WRONG spec by more than the tolerance. Without this, "production matches
// the reference" could be green simply because the axis does nothing at this scale — a
// vacuous witness. This compares reference-to-reference, so it cannot be satisfied by
// anything production does.
func gemmaOracleAssertAxisLive(t *testing.T, m *Model, base [][]float32, wrong gemmaOracleSpec, axis string) {
	t.Helper()
	alt := gemmaReference(t, m, gemmaOracleIDs, wrong)
	var worst float64
	for tt := range gemmaOracleIDs {
		if d := cpuOracleMaxAbsDiff(base[tt], alt[tt]); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Errorf("axis %q is inert in this fixture (max|delta| = %.3e <= tol %.0e): the oracle would stay green with that axis wrong",
			axis, worst, cpuOracleTol)
	}
}

// TestGemmaCPUNumericOracleAxesAreLive is the anti-vacuity gate for the family axes that
// a size/scale choice could accidentally switch off — the local/global cadence, the dual
// rope base, the soft-caps, the query_pre_attn_scalar and the per-head qk-norm. Each one
// is mutated to the "wrong but plausible" value and the reference must move.
func TestGemmaCPUNumericOracleAxesAreLive(t *testing.T) {
	t.Run("gemma2", func(t *testing.T) {
		m := newGemmaOracleModel(t, false)
		spec := gemmaOracleSpecFor(false)
		base := gemmaReference(t, m, gemmaOracleIDs, spec)

		// Every layer sliding — i.e. `sliding_window` applied uniformly, with no
		// full-attention layers at all. This is the shape a derivation that ignores
		// Gemma2's alternation produces.
		allSliding := spec
		allSliding.pattern = spec.layers + 1
		gemmaOracleAssertAxisLive(t, m, base, allSliding, "gemma2 local/global alternation")

		// Every layer full — the window ignored entirely.
		noWindow := spec
		noWindow.window = -1
		gemmaOracleAssertAxisLive(t, m, base, noWindow, "gemma2 sliding window")

		noAttnCap := spec
		noAttnCap.attnCap = 0
		gemmaOracleAssertAxisLive(t, m, base, noAttnCap, "attn_logit_softcapping")

		noLogitCap := spec
		noLogitCap.logitCap = 0
		gemmaOracleAssertAxisLive(t, m, base, noLogitCap, "final_logit_softcapping")

		headDimScale := spec
		headDimScale.qpas = m.Cfg.HeadDim
		gemmaOracleAssertAxisLive(t, m, base, headDimScale, "query_pre_attn_scalar")
	})

	t.Run("gemma3", func(t *testing.T) {
		m := newGemmaOracleModel(t, true)
		spec := gemmaOracleSpecFor(true)
		base := gemmaReference(t, m, gemmaOracleIDs, spec)

		allSliding := spec
		allSliding.pattern = spec.layers + 1
		gemmaOracleAssertAxisLive(t, m, base, allSliding, "gemma3 5:1 local/global cadence")

		oneTheta := spec
		oneTheta.localTheta = spec.fullTheta
		gemmaOracleAssertAxisLive(t, m, base, oneTheta, "rope_local_base_freq vs rope_theta")

		noQKNorm := spec
		noQKNorm.qkNorm = false
		gemmaOracleAssertAxisLive(t, m, base, noQKNorm, "gemma3 per-head qk-norm")

		headDimScale := spec
		headDimScale.qpas = m.Cfg.HeadDim
		gemmaOracleAssertAxisLive(t, m, base, headDimScale, "query_pre_attn_scalar")
	})
}

// TestGemmaCPUNumericOracleIsSensitive proves the comparison is non-vacuous on the
// WEIGHT axis: perturbing ONE raw fixture element must move the compared logits far
// beyond the tolerance. All four sandwich norms are listed separately because a
// reference that collapsed the sandwich to PreNorm — or that reused
// post_attention_layernorm as the MLP pre-norm, the Llama reading of that name — would
// stay green if only one of them were covered. The tied head (model.embed_tokens.weight)
// is listed because an untied reference would ignore a head perturbation.
func TestGemmaCPUNumericOracleIsSensitive(t *testing.T) {
	for _, tc := range []struct {
		name   string
		gemma3 bool
		tensor string
	}{
		{"input_layernorm", false, "model.layers.0.input_layernorm.weight"},
		{"post_attention_layernorm", false, "model.layers.0.post_attention_layernorm.weight"},
		{"pre_feedforward_layernorm", false, "model.layers.0.pre_feedforward_layernorm.weight"},
		{"post_feedforward_layernorm", false, "model.layers.0.post_feedforward_layernorm.weight"},
		{"final_norm", false, "model.norm.weight"},
		{"tied_head_embed", false, "model.embed_tokens.weight"},
		{"o_proj", false, "model.layers.0.self_attn.o_proj.weight"},
		{"gemma3_q_norm", true, "model.layers.0.self_attn.q_norm.weight"},
		{"gemma3_k_norm", true, "model.layers.0.self_attn.k_norm.weight"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newGemmaOracleModel(t, tc.gemma3)
			spec := gemmaOracleSpecFor(tc.gemma3)
			ids := gemmaOracleIDs
			ref := gemmaReference(t, m, ids, spec)

			meta, ok := m.manifest[tc.tensor]
			if !ok {
				t.Fatalf("fixture tensor %q missing", tc.tensor)
			}
			orig := math.Float32frombits(binary.LittleEndian.Uint32(m.raw[meta.Offset:]))
			binary.LittleEndian.PutUint32(m.raw[meta.Offset:], math.Float32bits(orig+0.5))

			act := m.Forward(ids)
			var worst float64
			for tt := range ids {
				if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > worst {
					worst = d
				}
			}
			if worst <= cpuOracleTol {
				t.Fatalf("perturbed fixture still within tolerance (max|delta|=%.3e) — the oracle is vacuous", worst)
			}
		})
	}
}
