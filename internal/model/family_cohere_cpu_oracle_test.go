package model

// family_cohere_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU
// numeric oracle for the Cohere (Command-R / Command-R+) family. Companion to
// family_cpu_oracle_test.go, which states the doctrine, and to
// family_gptneox_cpu_oracle_test.go, the other ParallelResidual reference.
//
// Independence discipline (family_cpu_oracle_test.go:13-24): the reference below is a
// plain scalar transcription of HuggingFace transformers/models/cohere/
// {modeling,configuration}_cohere.py. It reuses NONE of the production machinery —
// tensors come straight out of the fixture's own HF-order bytes, every matmul / norm /
// softmax / SiLU is a naive in-order scalar loop, the rotary table is rebuilt from theta
// here, and the block dataflow is hardcoded to Cohere's published topology instead of
// being routed through cfg.BlockTopology / normCfg / parallelMLPNorms / applyQKNormCfg /
// applyRopeRow / logitScaleInPlace.
//
// What Cohere actually does, transcribed from the HF source (not from the family's
// reputation). Every claim below was read off the upstream file, not assumed:
//
//   - PARALLEL RESIDUAL with ONE norm (CohereDecoderLayer.forward):
//     residual = x; xn = input_layernorm(x); x = residual + self_attn(xn) + mlp(xn).
//     Both branches read the SAME normed tensor from the SAME norm — unlike GPT-NeoX,
//     which has two distinct norms. Cohere carries NO post_attention_layernorm at all,
//     so this fixture also exercises weights.go parallelMLPNorms' shared-norm fallback.
//   - NORM: CohereLayerNorm — mean-subtracting LayerNorm with a learned WEIGHT and NO
//     BIAS ("self.weight = nn.Parameter(torch.ones(hidden_size))"; there is no bias
//     Parameter). So it is neither Llama's RMSNorm (no mean subtraction) nor GPT-NeoX's
//     nn.LayerNorm (which does carry a bias). This is the axis config.go:624 turns on
//     (cfg.LayerNorm), implemented by arch.go layernorm() with a nil bias.
//   - QK-NORM (optional, config.use_qk_norm, default False; ON for Command-R+): q and k
//     are viewed as (..., num_heads, head_dim) and normed by a CohereLayerNorm whose
//     hidden_size is the TUPLE (num_heads, head_dim). The upstream docstring is explicit:
//     "The hidden size can be a tuple or an int. The tuple is used for QKNorm to
//     normalize across head_dim". Two facts follow and BOTH matter: (a) the reduction is
//     PER HEAD over head_dim — not one reduction over the whole num_heads*head_dim
//     projection (that is OLMo2's shape, a different family), and (b) it is a
//     MEAN-SUBTRACTING LayerNorm, not an RMSNorm. The weight is per-head: row h of the
//     (num_heads, head_dim) parameter scales head h.
//   - ROTARY: Cohere's is the INTERLEAVED (GPT-J style) convention, NOT Llama's
//     rotate_half. CohereRotaryEmbedding.forward builds
//     emb = torch.repeat_interleave(freqs, 2, dim=-1)  ("diff from Llama: we
//     interleave() instead of cat()") and Cohere's own rotate_half is
//     x1 = x[..., ::2]; x2 = x[..., 1::2]; stack([-x2, x1]).flatten(-2)  ("different
//     from e.g. Llama"), so element 2j pairs with element 2j+1 under angle
//     pos*theta^(-2j/head_dim) — NOT element j with element j+head_dim/2.
//   - ATTENTION: causal GQA, scores scaled by head_dim**-0.5 (CohereAttention.scaling),
//     softmax in fp32, no ALiBi, no sinks, no softcap. attention_bias defaults False, so
//     no projection in the fixture is biased.
//   - MLP: the standard SwiGLU down(silu(gate(x)) * up(x)) — hidden_act defaults "silu".
//   - HEAD: tie_word_embeddings defaults True, so the logits are embed_tokens @ x_final
//     with NO separate lm_head tensor.
//   - LOGIT SCALE: CohereForCausalLM multiplies the final logits by config.logit_scale
//     (default 0.0625) — "main diff from Llama". This is the axis arch.go
//     logitScaleInPlace implements and config.go:621 defaults for the family.
//
// The fixture is built with synthBuildRaw on Cohere's REAL checkpoint names (the
// HF-standard Llama names minus post_attention_layernorm, plus the optional per-head
// q_norm/k_norm) and loaded through newHFCheckpointModel — the constructor the f32
// HF-source loaders funnel through: LoadSafetensors and LoadSafetensorsDir
// (safetensors.go) and Load (weights.go), the three call sites
// cohere_loader_routing_test.go pins from the PUBLIC entry points, since this file
// constructs its model directly and so witnesses the constructor but not the routing.
// The GPTQ loader is NOT one of them: gptq.go still calls newModel, matching the KNOWN
// RESIDUAL paragraph in cohere_rotary.go — a quantized checkpoint keeps q_proj/k_proj in
// its own decoded store rather than in the f32 blob this pass rewrites, so it stays
// mis-rotated and carries no witness. What the oracle below covers is therefore an
// UNQUANTIZED Cohere download, seen exactly as HuggingFace presents it. Every norm weight
// gets a distinct NON-UNIT gain so norm routing is numerically live rather than masked by
// 1.0. nH*hd (32) deliberately differs from HiddenSize (24) so a projection-width /
// hidden-width conflation cannot cancel, nKV != nH so a GQA grouping bug cannot cancel,
// and head_dim 8 makes the interleaved pairing (0,1),(2,3),(4,5),(6,7) distinguishable
// from the rotate_half pairing (0,4),(1,5),(2,6),(3,7) — at head_dim 2 the two
// conventions coincide and the rotary axis would be unobservable.

import (
	"encoding/binary"
	"math"
	"testing"
)

// cohereOracleLayerNorm is CohereLayerNorm: (x-mean)/sqrt(var+eps)*w, with the biased
// (1/N) variance torch uses and NO bias term (Cohere's norm has no bias Parameter).
func cohereOracleLayerNorm(x, w []float32, eps float32) []float32 {
	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= float32(len(x))
	var ss float32
	for _, v := range x {
		d := v - mean
		ss += d * d
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = (v - mean) * inv * w[i]
	}
	return out
}

// cohereOracleRopeInterleaved rotates ONE head vector (length hd) in place at position
// pos with Cohere's INTERLEAVED convention: freqs[j] = pos*theta^(-2j/hd) is
// repeat_interleave'd across the pair (2j, 2j+1), and Cohere's rotate_half pairs the
// EVEN element with the ODD one that follows it:
//
//	out[2j]   = x[2j]*cos_j   - x[2j+1]*sin_j
//	out[2j+1] = x[2j+1]*cos_j + x[2j]*sin_j
//
// This is NOT applyRopeRow (which pairs j with j+hd/2); reconciling the two conventions
// is exactly what the load-time Cohere rotary re-layout exists to do.
func cohereOracleRopeInterleaved(hv []float32, pos, hd int, theta float64) {
	for j := 0; j < hd/2; j++ {
		angle := float64(pos) / math.Pow(theta, float64(2*j)/float64(hd))
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		a, b := hv[2*j], hv[2*j+1]
		hv[2*j] = a*c - b*s
		hv[2*j+1] = b*c + a*s
	}
}

// cohereOracleQKNorm applies CohereLayerNorm(hidden_size=(nHeads, hd)) to a packed
// projection: ONE mean/variance reduction PER HEAD over head_dim, scaled by that head's
// own row of the (nHeads, hd) weight. Deliberately not applyQKNormCfg.
func cohereOracleQKNorm(hv, w []float32, nHeads, hd int, eps float32) {
	for h := 0; h < nHeads; h++ {
		head := hv[h*hd : (h+1)*hd]
		copy(head, cohereOracleLayerNorm(head, w[h*hd:(h+1)*hd], eps))
	}
}

// cohereOracleCfg is the tiny Cohere fixture config. LogitScale / LayerNorm /
// BlockTopology are left ZERO so the test proves deriveConfigAxes lands them for the
// family rather than the fixture asserting them by hand.
func cohereOracleCfg(qkNorm bool) Config {
	return Config{
		HiddenSize:        24,
		NumLayers:         3,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  40,
		VocabSize:         53,
		ModelType:         "cohere",
		Architectures:     []string{"CohereForCausalLM"},
		HiddenAct:         "silu",
		RMSNormEps:        1e-5, // HF layer_norm_eps
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		QKNorm:            qkNorm, // HF use_qk_norm (Command-R+ sets it)
	}
}

// cohereOracleTensors is Cohere's REAL tensor roster at the fixture geometry: no
// post_attention_layernorm (Cohere has none), no lm_head (tied), no projection biases
// (attention_bias False), and per-head (nHeads, head_dim) q_norm/k_norm only on the
// use_qk_norm lineage.
func cohereOracleTensors(cfg Config) []synthTensor {
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	type ts = synthTensor
	out := []ts{{"model.embed_tokens.weight", []int{V, H}}}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		out = append(out,
			ts{p + "input_layernorm.weight", []int{H}}, // the ONE norm; no bias, no MLP norm
			ts{p + "self_attn.q_proj.weight", []int{nH * hd, H}},
			ts{p + "self_attn.k_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.v_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.o_proj.weight", []int{H, nH * hd}},
		)
		if cfg.QKNorm {
			out = append(out,
				ts{p + "self_attn.q_norm.weight", []int{nH, hd}},
				ts{p + "self_attn.k_norm.weight", []int{nKV, hd}},
			)
		}
		out = append(out,
			ts{p + "mlp.gate_proj.weight", []int{I, H}},
			ts{p + "mlp.up_proj.weight", []int{I, H}},
			ts{p + "mlp.down_proj.weight", []int{H, I}},
		)
	}
	return append(out, ts{"model.norm.weight", []int{H}})
}

// newCohereOracleModel builds the fixture and loads it through newHFCheckpointModel, the
// HF-source construction seam. It also returns a PRISTINE copy of the checkpoint bytes in
// HF order, which the reference reads — so the reference sees the download as HuggingFace
// wrote it and stays independent of any layout normalization the loader performs.
func newCohereOracleModel(t *testing.T, qkNorm bool) (*Model, []byte) {
	t.Helper()
	cfg := cohereOracleCfg(qkNorm)
	if err := cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	man, raw := synthBuildRaw(cohereOracleTensors(cfg), func(name string, next func() float32) float32 {
		if isCPUOracleNormWeight(name) {
			return 1 + 0.25*next() // distinct NON-UNIT gains, well-conditioned
		}
		return synthMatmulFill(name, next)
	})
	hf := append([]byte(nil), raw...)
	m, err := newHFCheckpointModel(cfg, man, raw)
	if err != nil {
		t.Fatalf("newHFCheckpointModel(cohere fixture): %v", err)
	}
	return m, hf
}

// cohereReference runs the independent Cohere forward: per-position logits for ids.
// Every step is the HF CohereDecoderLayer dataflow, hardcoded — NOT routed through
// cfg.BlockTopology, normCfg, parallelMLPNorms, applyQKNormCfg, applyRopeRow or
// logitScaleInPlace.
func cohereReference(t *testing.T, m *Model, hf []byte, ids []int) [][]float32 {
	t.Helper()
	cfg := m.Cfg
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	grp := nH / nKV
	eps := float32(cfg.RMSNormEps)
	theta := cfg.RopeTheta
	seq := len(ids)
	qkNorm := cfg.QKNorm

	// tensor reads the PRISTINE HF-order bytes, not m.raw.
	tensor := func(name string) []float32 {
		t.Helper()
		meta, ok := m.manifest[name]
		if !ok {
			t.Fatalf("fixture tensor %q missing from manifest", name)
		}
		n := meta.Nbytes / 4
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(hf[meta.Offset+i*4:]))
		}
		return out
	}

	embed := tensor("model.embed_tokens.weight")
	x := make([][]float32, seq)
	for tt, id := range ids {
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		inNorm := tensor(p + "input_layernorm.weight")
		wq := tensor(p + "self_attn.q_proj.weight")
		wk := tensor(p + "self_attn.k_proj.weight")
		wv := tensor(p + "self_attn.v_proj.weight")
		wo := tensor(p + "self_attn.o_proj.weight")
		wg := tensor(p + "mlp.gate_proj.weight")
		wu := tensor(p + "mlp.up_proj.weight")
		wd := tensor(p + "mlp.down_proj.weight")
		var qn, kn []float32
		if qkNorm {
			qn = tensor(p + "self_attn.q_norm.weight")
			kn = tensor(p + "self_attn.k_norm.weight")
		}

		// ONE norm feeds BOTH branches (CohereDecoderLayer.forward).
		xn := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			xn[tt] = cohereOracleLayerNorm(x[tt], inNorm, eps)
		}

		// --- attention branch on xn ---
		q := make([][]float32, seq)
		k := make([][]float32, seq)
		v := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			q[tt] = cpuOracleMatVec(wq, xn[tt], nH*hd, H)
			k[tt] = cpuOracleMatVec(wk, xn[tt], nKV*hd, H)
			v[tt] = cpuOracleMatVec(wv, xn[tt], nKV*hd, H)
			if qkNorm {
				// per-head CohereLayerNorm over head_dim, BEFORE rotary.
				cohereOracleQKNorm(q[tt], qn, nH, hd, eps)
				cohereOracleQKNorm(k[tt], kn, nKV, hd, eps)
			}
			for h := 0; h < nH; h++ {
				cohereOracleRopeInterleaved(q[tt][h*hd:(h+1)*hd], tt, hd, theta)
			}
			for h := 0; h < nKV; h++ {
				cohereOracleRopeInterleaved(k[tt][h*hd:(h+1)*hd], tt, hd, theta)
			}
		}
		scale := float32(1.0 / math.Sqrt(float64(hd))) // CohereAttention.scaling
		attnOut := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			concat := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				kvh := h / grp
				qh := q[tt][h*hd : (h+1)*hd]
				scores := make([]float32, tt+1)
				for j := 0; j <= tt; j++ {
					kh := k[j][kvh*hd : (kvh+1)*hd]
					var s float32
					for d := 0; d < hd; d++ {
						s += qh[d] * kh[d]
					}
					scores[j] = s * scale
				}
				cpuOracleSoftmax(scores)
				o := concat[h*hd : (h+1)*hd]
				for j := 0; j <= tt; j++ {
					vh := v[j][kvh*hd : (kvh+1)*hd]
					for d := 0; d < hd; d++ {
						o[d] += scores[j] * vh[d]
					}
				}
			}
			attnOut[tt] = cpuOracleMatVec(wo, concat, H, nH*hd) // no o_proj bias
		}

		// --- MLP branch, on the SAME xn (not on x+attnOut) ---
		mlpOut := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			gate := cpuOracleMatVec(wg, xn[tt], I, H)
			up := cpuOracleMatVec(wu, xn[tt], I, H)
			for i := 0; i < I; i++ {
				gate[i] = cpuOracleSilu(gate[i]) * up[i]
			}
			mlpOut[tt] = cpuOracleMatVec(wd, gate, H, I)
		}

		// residual + attention + mlp, ONE add (CohereDecoderLayer.forward)
		for tt := 0; tt < seq; tt++ {
			for i := 0; i < H; i++ {
				x[tt][i] += attnOut[tt][i] + mlpOut[tt][i]
			}
		}
	}

	norm := tensor("model.norm.weight")
	logitScale := float32(cfg.LogitScale)
	logits := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		xf := cohereOracleLayerNorm(x[tt], norm, eps)
		row := cpuOracleMatVec(embed, xf, V, H) // tied head
		for i := range row {
			row[i] *= logitScale // CohereForCausalLM: logits = logits * logit_scale
		}
		logits[tt] = row
	}
	return logits
}

var cohereOracleIDs = []int{3, 17, 5, 23, 41, 2, 19}

// runCohereOracle compares the production cacheless Forward AND the cached Prefill/Step
// decode path against the independent reference at every position.
func runCohereOracle(t *testing.T, qkNorm bool) {
	t.Helper()
	m, hf := newCohereOracleModel(t, qkNorm)

	// The derivation must land the published Cohere axes — the reference hardcodes them.
	if m.Cfg.BlockTopology != ParallelResidual {
		t.Fatalf("cohere derived topology = %v, want ParallelResidual", m.Cfg.BlockTopology)
	}
	if !m.Cfg.LayerNorm {
		t.Fatal("cohere derived LayerNorm = false, want true (CohereLayerNorm is mean-subtracting, not RMSNorm)")
	}
	if m.Cfg.LogitScale != 0.0625 {
		t.Fatalf("cohere derived LogitScale = %v, want 0.0625", m.Cfg.LogitScale)
	}
	if m.Cfg.DenseMLP {
		t.Fatal("cohere derived DenseMLP = true, want false (CohereMLP is SwiGLU gate/up/down)")
	}
	// Cohere carries no post_attention_layernorm: the MLP branch must fall back to the
	// SHARED attention norm (weights.go parallelMLPNorms). If the fixture ever grew one,
	// that fallback would stop being exercised.
	if m.has(layerName(0, "post_attention_layernorm.weight")) {
		t.Fatal("fixture grew a post_attention_layernorm — Cohere has none, and the shared-norm fallback would no longer be under test")
	}
	if qkNorm && !m.has(layerName(0, "self_attn.q_norm.weight")) {
		t.Fatal("use_qk_norm lineage lost its q_norm tensor")
	}

	ids := cohereOracleIDs
	ref := cohereReference(t, m, hf, ids)

	act := m.Forward(ids)
	for tt := range ids {
		if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > cpuOracleTol {
			t.Errorf("Forward logits pos %d: max|delta| vs reference = %.3e (tol %.0e)", tt, d, cpuOracleTol)
		}
	}

	// Cached decode path: Prefill then Step must match the reference at the same
	// positions (the reference is cacheless, so Step(id) at position len(ids) is compared
	// against a reference run over the extended prompt).
	s := m.NewSession()
	pf := s.Prefill(ids)
	if d := cpuOracleMaxAbsDiff(pf, ref[len(ids)-1]); d > cpuOracleTol {
		t.Errorf("Prefill last logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
	next := 11
	st := s.Step(next)
	extRef := cohereReference(t, m, hf, append(append([]int(nil), ids...), next))
	if d := cpuOracleMaxAbsDiff(st, extRef[len(ids)]); d > cpuOracleTol {
		t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
}

// TestCohereCPUNumericOracle is the Cohere×cpu M4 witness for the Command-R lineage
// (use_qk_norm off): the production forward must reproduce the independent HF-semantics
// reference on every position within cpuOracleTol. It covers the parallel residual with
// ONE shared norm, the bias-free mean-subtracting CohereLayerNorm, the tied head, the
// SwiGLU MLP, causal GQA at head_dim**-0.5, the 0.0625 logit scale, and Cohere's
// INTERLEAVED rotary.
//
// THIS TEST CAUGHT the rotary-convention bug and now holds it fixed. fak has one rotary
// kernel and it is Llama's rotate_half; an HF Cohere checkpoint's q/k rows are in the
// interleaved order, so every position past 0 was rotated with the wrong component pairs
// (max|delta| 1.2e-3 .. 2.3e-3 on this fixture, growing with position, and exact at
// position 0 because a length-1 causal softmax is 1.0 no matter what q and k are).
// cohere_rotary.go now re-lays-out q/k at load on the HF-source seam. Overlaying that
// file with a pass-through stub reproduces the original deltas exactly.
//
// If this test reds, the honesty fence demotes the cell back to M3 (drop the covmatrix
// OracleInCI bit with it).
func TestCohereCPUNumericOracle(t *testing.T) {
	runCohereOracle(t, false)
}

// TestCohereCPUNumericOracleQKNorm is the same witness at the Command-R+ geometry:
// use_qk_norm on, so each layer carries a CohereLayerNorm of shape (num_heads, head_dim)
// on q and (num_key_value_heads, head_dim) on k.
//
// THIS TEST CAUGHT the qk-norm bug. arch.go applyQKNormCfg read a full-projection-width
// qk-norm weight as OLMo2's — ONE RMSNorm reduction over the whole num_heads*head_dim
// vector against one shared parameter — but Cohere's is a tuple-shaped CohereLayerNorm:
// one MEAN-SUBTRACTING reduction PER HEAD over head_dim against that head's own weight
// row. Two divergences at once (wrong reduction extent, wrong norm function), and no
// length check can see either, because Cohere's (num_heads, head_dim) parameter has
// exactly the width OLMo2's flat form expects. Residual after the rotary fix was
// 8.2e-3 .. 2.5e-2; the cfg.QKNormPerHeadWeight branch closes it, and reverting arch.go
// to its pre-fix revision under -overlay reproduces those deltas exactly.
func TestCohereCPUNumericOracleQKNorm(t *testing.T) {
	m, _ := newCohereOracleModel(t, true)
	if !m.Cfg.QKNorm {
		t.Fatal("cohere use_qk_norm lineage derived QKNorm = false")
	}
	runCohereOracle(t, true)
}

// TestCohereCPUNumericOracleIsSensitive proves the comparison is non-vacuous: perturbing
// ONE raw fixture element must move the compared logits far beyond the tolerance. The
// per-head qk-norm weights and the tied embedding are included because a reference that
// ignored the qk-norm, or that used a separate untied head, would stay green without them.
func TestCohereCPUNumericOracleIsSensitive(t *testing.T) {
	for _, tc := range []struct {
		name   string
		tensor string
	}{
		{"q_proj_weight", "model.layers.0.self_attn.q_proj.weight"},
		{"input_layernorm_weight", "model.layers.0.input_layernorm.weight"},
		{"q_norm_weight", "model.layers.0.self_attn.q_norm.weight"},
		{"k_norm_weight", "model.layers.0.self_attn.k_norm.weight"},
		{"down_proj_weight", "model.layers.0.mlp.down_proj.weight"},
		{"final_norm_weight", "model.norm.weight"},
		{"embed_tokens_weight", "model.embed_tokens.weight"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, hf := newCohereOracleModel(t, true)
			ids := cohereOracleIDs
			ref := cohereReference(t, m, hf, ids)

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
