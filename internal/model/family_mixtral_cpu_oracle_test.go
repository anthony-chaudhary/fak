package model

// family_mixtral_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU
// numeric oracle for the Mixtral (Mixtral-8x7B / 8x22B) family. Companion to
// family_cpu_oracle_test.go, which states the doctrine, and the first oracle in the
// family set whose FFN sub-layer is a routed Mixture-of-Experts rather than a single
// dense MLP.
//
// Independence discipline (family_cpu_oracle_test.go:13-24): the reference below is a
// plain scalar transcription of HuggingFace transformers/models/mixtral/
// {modeling,configuration}_mixtral.py. It reuses NONE of the production machinery —
// tensors come straight out of the manifest bytes via cpuOracleTensor under their
// HF-CHECKPOINT names (block_sparse_moe.gate / experts.N.w1|w2|w3), never the canonical
// mlp.gate / mlp.experts.N.{gate,up,down}_proj names materializeMixtralBlockSparseTensors
// aliases them to; every matmul / norm / softmax / SiLU is a naive in-order scalar loop;
// the router is re-derived here rather than routed through moe.go's route(); and the block
// dataflow is hardcoded to Mixtral's published topology instead of being routed through
// cfg.BlockTopology / ffnFor / ffnForLayer / normCfg / applyRopeRow.
//
// What Mixtral actually does, transcribed from the HF source (not from the family's
// reputation). Every claim below was read off the upstream file:
//
//   - BLOCK: the verbatim Llama PreNorm decoder layer (MixtralDecoderLayer.forward):
//     x += self_attn(input_layernorm(x)); x += block_sparse_moe(post_attention_layernorm(x)).
//     Two distinct RMSNorms, serial residuals — NOT parallel.
//   - NORM: MixtralRMSNorm — x/sqrt(mean(x^2)+eps)*w. No mean subtraction, no bias,
//     no 1+w gain.
//   - ATTENTION: causal GQA, rotate_half (non-interleaved) RoPE on q and k, scores
//     scaled by head_dim**-0.5, softmax in fp32. MixtralAttention builds every
//     projection with bias=False, so NOTHING in the attention block is biased, and
//     sliding_window is None on both published checkpoints (full causal attention).
//   - ROUTER (MixtralSparseMoeBlock.forward):
//     1. router_logits = self.gate(x)                     # nn.Linear(hidden, n_experts, bias=False)
//     2. routing_weights = softmax(router_logits, dim=1)  # over ALL num_local_experts
//     3. routing_weights, selected = topk(routing_weights, top_k)   # AFTER the softmax
//     4. routing_weights /= routing_weights.sum(dim=-1, keepdim=True)  # RENORMALIZE
//     Step 4 is the axis that actually protects Mixtral, and there is a MEASURED
//     result behind that claim which contradicts the usual folklore. The commonly
//     cited "silent bug" is taking the top-k BEFORE the softmax and softmaxing only
//     the survivors (the GPT-OSS router order, routeTopKSoftmax). For Mixtral that
//     is not a bug at all — it is an exact identity. softmax is strictly monotonic,
//     so the selection is unchanged, and the shared denominator Z cancels inside the
//     renormalization:
//     (e^za/Z) / ((e^za + e^zb)/Z)  ==  e^za / (e^za + e^zb).
//     TestMixtralCPUNumericOracleAxesAreLive measures the two orders on this fixture
//     at max|delta| 1.9e-7 — float noise. The order only becomes load-bearing when
//     the renormalization is OFF, which is the regime that axis is exercised in.
//     So the router's genuinely divergent modes are: skipping step 4, softmaxing over
//     a subset in step 2, letting unselected experts contribute, and getting the
//     selection itself wrong. Each is shape-clean, and each is pinned below.
//     One corollary the sensitivity test has to respect: because the renormalization
//     cancels the shared denominator, a router weight is observable ONLY if it flips
//     the selection or moves a SELECTED expert's logit relative to the other selected
//     ones.
//   - EXPERT (MixtralBlockSparseTop2MLP.forward): w2(act_fn(w1(x)) * w3(x)) with
//     act_fn = ACT2FN["silu"]. The upstream names are w1/w2/w3, NOT gate/up/down:
//     w1 is the GATE (the silu'd branch), w3 is the UP (the multiplied branch), w2 is
//     the DOWN projection. w1 and w3 have the SAME shape [ffn_dim, hidden], so swapping
//     them is shape-clean and silently wrong; materialize.go:126-128 is the mapping
//     under test and the reference re-derives it from the source names.
//   - WEIGHTING: `expert_layer(current_state) * routing_weights[top_x, idx, None]` —
//     the gate weight scales the expert's OUTPUT, after the down projection. Scaling
//     the INPUT instead is NOT equivalent (silu is not homogeneous and the gate·up
//     product is quadratic), and again shape-clean.
//   - ONLY THE SELECTED EXPERTS CONTRIBUTE: final_hidden_states is index_add_'d from
//     the top_k experts only; the other num_local_experts-top_k contribute nothing.
//   - HEAD: tie_word_embeddings is False on both published Mixtral checkpoints, so
//     lm_head is a separate nn.Linear(hidden, vocab, bias=False).
//
// The fixture is built with synthBuildRaw on Mixtral's REAL checkpoint names and loaded
// through newHFCheckpointModel — the HF-source construction seam every f32 safetensors
// loader funnels through — so the production side sees the download exactly as a real
// Mixtral repo presents it and must go through materializeMixtralBlockSparseTensors'
// w1/w2/w3 -> gate/up/down aliasing. Every norm weight gets a distinct NON-UNIT gain so
// norm routing is numerically live rather than masked by 1.0. Geometry choices that are
// deliberate, not incidental:
//
//   - nH*hd (32) != HiddenSize (24), so a projection-width / hidden-width conflation
//     cannot cancel; nKV (2) != nH (4), so a GQA grouping bug cannot cancel.
//   - NumExperts is 5, an ODD non-power-of-two, so an expert-index bug cannot alias
//     into a neat power-of-two grouping, and E > 2*top_k so "selected" is a strict
//     subset twice over.
//   - The router weight is filled at a WIDER scale than the other matmuls so the
//     per-expert logits genuinely separate: a router whose logits all collapse to
//     within float noise would make the selection arbitrary and the top-k/renorm
//     axes unobservable.
//   - NormTopKProb is left FALSE in the fixture Config so deriveConfigAxes has to
//     derive it for the family (config.go:605); the reference hardcodes HF's
//     unconditional renormalization.

import (
	"encoding/binary"
	"math"
	"sort"
	"strconv"
	"testing"
)

// mixtralOracleLayerSrc is the SOURCE (checkpoint) prefix for layer l. Built with
// strconv here rather than through layerPrefix so the reference owns its own names.
func mixtralOracleLayerSrc(l int) string { return "model.layers." + strconv.Itoa(l) + "." }

// mixtralOracleRouterSrc is HF's per-layer router: block_sparse_moe.gate.weight, an
// [num_local_experts, hidden] nn.Linear with bias=False.
func mixtralOracleRouterSrc(l int) string {
	return mixtralOracleLayerSrc(l) + "block_sparse_moe.gate.weight"
}

// mixtralOracleExpertSrc names one expert projection under HF's UPSTREAM spelling:
// w1 (gate), w2 (down), w3 (up). The production side reads the canonical
// mlp.experts.N.{gate,up,down}_proj aliases; the reference never does.
func mixtralOracleExpertSrc(l, e int, w string) string {
	return mixtralOracleLayerSrc(l) + "block_sparse_moe.experts." + strconv.Itoa(e) + "." + w + ".weight"
}

// mixtralOracleSoftmaxOver is F.softmax over the WHOLE vector, allocating and
// max-subtracted, in f32 — HF computes the router softmax in float32 explicitly
// (`dtype=torch.float`). Deliberately not softmaxOf (production).
func mixtralOracleSoftmaxOver(z []float32) []float32 {
	out := append([]float32(nil), z...)
	cpuOracleSoftmax(out)
	return out
}

// mixtralOraclePick is one (expert, gate weight) selection from the reference router.
type mixtralOraclePick struct {
	expert int
	weight float32
}

// mixtralOracleSpec is every axis the reference consults. The HF spec is built by
// mixtralOracleSpecFor from the DERIVED Config; the deliberate-wrong knobs at the bottom
// exist only so TestMixtralCPUNumericOracleAxesAreLive can prove each semantic is
// observable in this fixture rather than inert.
type mixtralOracleSpec struct {
	layers  int
	heads   int
	kvHeads int
	headDim int
	hidden  int
	inter   int
	vocab   int
	experts int
	topK    int
	theta   float64
	eps     float32

	// normTopK: HF renormalizes the selected top-k routing weights to sum to 1.
	normTopK bool
	// tiedHead: published Mixtral is UNTIED (a separate lm_head.weight).
	tiedHead bool

	// --- deliberate-wrong knobs; all false/zero for the HF transcription ---

	// topKBeforeSoftmax selects on the RAW logits and softmaxes only the survivors
	// (the GPT-OSS router order). Same selection, different gate weights.
	topKBeforeSoftmax bool
	// weightExpertInput scales the expert INPUT by the gate weight instead of its output.
	weightExpertInput bool
	// allExperts lets every expert contribute (a dense MoE): top_k is ignored.
	allExperts bool
	// swapW1W3 reads w3 as the silu'd gate and w1 as the multiplied up branch.
	swapW1W3 bool
	// kvHeadModulo groups query head h onto kv head h%nKV instead of h/(nH/nKV).
	kvHeadModulo bool
}

// mixtralOracleSpecFor lowers the DERIVED production Config to the reference spec. Only
// the axes are read across; the dataflow stays hardcoded to HF's Mixtral topology.
func mixtralOracleSpecFor(cfg Config) mixtralOracleSpec {
	return mixtralOracleSpec{
		layers:   cfg.NumLayers,
		heads:    cfg.NumHeads,
		kvHeads:  cfg.NumKVHeads,
		headDim:  cfg.HeadDim,
		hidden:   cfg.HiddenSize,
		inter:    cfg.IntermediateSize,
		vocab:    cfg.VocabSize,
		experts:  cfg.NumExperts,
		topK:     cfg.NumExpertsPerTok,
		theta:    cfg.RopeTheta,
		eps:      float32(cfg.RMSNormEps),
		normTopK: cfg.NormTopKProb,
		tiedHead: cfg.TieWordEmbeddings,
	}
}

// mixtralOracleRoute is MixtralSparseMoeBlock's router, transcribed. HF order:
// softmax over ALL experts -> torch.topk -> renormalize the k survivors. torch.topk's
// tie-break is stable (largest first; equal values keep the lower index), which
// sort.SliceStable over an ascending index list reproduces.
func mixtralOracleRoute(logits []float32, spec mixtralOracleSpec) []mixtralOraclePick {
	E := len(logits)
	k := spec.topK
	if spec.allExperts {
		k = E
	}
	if k > E {
		k = E
	}
	if k < 1 {
		k = 1
	}

	order := func(score []float32) []int {
		idx := make([]int, E)
		for e := range idx {
			idx[e] = e
		}
		sort.SliceStable(idx, func(a, b int) bool { return score[idx[a]] > score[idx[b]] })
		return idx
	}

	if spec.topKBeforeSoftmax {
		// WRONG-BY-CONSTRUCTION variant: select on the raw logits, then softmax only
		// the survivors. Identical selection, different weights, and already normalized
		// (so the renormalization step is silently absorbed).
		idx := order(logits)
		top := make([]float32, k)
		for i := 0; i < k; i++ {
			top[i] = logits[idx[i]]
		}
		probs := mixtralOracleSoftmaxOver(top)
		picks := make([]mixtralOraclePick, k)
		for i := 0; i < k; i++ {
			picks[i] = mixtralOraclePick{expert: idx[i], weight: probs[i]}
		}
		return picks
	}

	probs := mixtralOracleSoftmaxOver(logits) // step 2: over ALL experts
	idx := order(probs)                       // step 3: topk AFTER the softmax
	picks := make([]mixtralOraclePick, k)
	var sum float32
	for i := 0; i < k; i++ {
		e := idx[i]
		picks[i] = mixtralOraclePick{expert: e, weight: probs[e]}
		sum += probs[e]
	}
	if spec.normTopK && sum != 0 { // step 4: renormalize the survivors
		for i := range picks {
			picks[i].weight /= sum
		}
	}
	return picks
}

// mixtralOracleCfg is the tiny Mixtral fixture config. NormTopKProb and BlockTopology
// are left ZERO so the test proves deriveConfigAxes lands them for the family rather
// than the fixture asserting them by hand. TieWordEmbeddings stays false: both published
// Mixtral checkpoints carry a separate lm_head.
func mixtralOracleCfg(topK int) Config {
	return Config{
		HiddenSize:       24,
		NumLayers:        3,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 40,
		VocabSize:        53,
		ModelType:        "mixtral",
		Architectures:    []string{"MixtralForCausalLM"},
		HiddenAct:        "silu",
		RMSNormEps:       1e-5,
		RopeTheta:        1000000, // Mixtral's published rope_theta
		NumExperts:       5,
		NumExpertsPerTok: topK,
	}
}

// mixtralOracleTensors is Mixtral's REAL tensor roster at the fixture geometry: the
// Llama attention/norm set with bias-free projections, the block_sparse_moe router and
// w1/w2/w3 expert triples, and an UNTIED lm_head.
func mixtralOracleTensors(cfg Config) []synthTensor {
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V, E := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize, cfg.NumExperts
	type ts = synthTensor
	out := []ts{{"model.embed_tokens.weight", []int{V, H}}}
	for l := 0; l < cfg.NumLayers; l++ {
		p := mixtralOracleLayerSrc(l)
		out = append(out,
			ts{p + "input_layernorm.weight", []int{H}},
			ts{p + "self_attn.q_proj.weight", []int{nH * hd, H}},
			ts{p + "self_attn.k_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.v_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.o_proj.weight", []int{H, nH * hd}},
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{mixtralOracleRouterSrc(l), []int{E, H}},
		)
		for e := 0; e < E; e++ {
			out = append(out,
				ts{mixtralOracleExpertSrc(l, e, "w1"), []int{I, H}}, // gate
				ts{mixtralOracleExpertSrc(l, e, "w2"), []int{H, I}}, // down
				ts{mixtralOracleExpertSrc(l, e, "w3"), []int{I, H}}, // up
			)
		}
	}
	return append(out,
		ts{"model.norm.weight", []int{H}},
		ts{"lm_head.weight", []int{V, H}}, // untied
	)
}

// isMixtralOracleRouterTensor reports the per-layer router weight, which is filled at a
// wider scale than the other matmuls (see the header: a router whose logits collapse
// makes the top-k and renormalization axes unobservable).
func isMixtralOracleRouterTensor(name string) bool {
	const suffix = "block_sparse_moe.gate.weight"
	return len(name) >= len(suffix) && name[len(name)-len(suffix):] == suffix
}

// newMixtralOracleModel builds the fixture on Mixtral's REAL checkpoint roster and loads
// it through newHFCheckpointModel, so the production side exercises
// materializeMixtralBlockSparseTensors' w1/w2/w3 -> gate/up/down aliasing.
func newMixtralOracleModel(t *testing.T, topK int) *Model {
	t.Helper()
	cfg := mixtralOracleCfg(topK)
	if err := cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	man, raw := synthBuildRaw(mixtralOracleTensors(cfg), func(name string, next func() float32) float32 {
		switch {
		case isCPUOracleNormWeight(name):
			return 1 + 0.25*next() // distinct NON-UNIT RMSNorm gains
		case isMixtralOracleRouterTensor(name):
			return next() * 0.6 // wide enough that the per-expert logits separate
		}
		return synthMatmulFill(name, next)
	})
	m, err := newHFCheckpointModel(cfg, man, raw)
	if err != nil {
		t.Fatalf("newHFCheckpointModel(mixtral fixture): %v", err)
	}
	return m
}

// mixtralReference runs the independent Mixtral forward: per-position logits for ids.
// Every step is the HF MixtralDecoderLayer / MixtralSparseMoeBlock /
// MixtralBlockSparseTop2MLP dataflow, hardcoded.
func mixtralReference(t *testing.T, m *Model, ids []int, spec mixtralOracleSpec) [][]float32 {
	t.Helper()
	H, I, V := spec.hidden, spec.inter, spec.vocab
	nH, nKV, hd := spec.heads, spec.kvHeads, spec.headDim
	grp := nH / nKV
	eps, theta := spec.eps, spec.theta
	seq := len(ids)

	embed := cpuOracleTensor(t, m, "model.embed_tokens.weight")
	x := make([][]float32, seq)
	for tt, id := range ids {
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}

	for l := 0; l < spec.layers; l++ {
		p := mixtralOracleLayerSrc(l)
		inNorm := cpuOracleTensor(t, m, p+"input_layernorm.weight")
		wq := cpuOracleTensor(t, m, p+"self_attn.q_proj.weight")
		wk := cpuOracleTensor(t, m, p+"self_attn.k_proj.weight")
		wv := cpuOracleTensor(t, m, p+"self_attn.v_proj.weight")
		wo := cpuOracleTensor(t, m, p+"self_attn.o_proj.weight")
		postNorm := cpuOracleTensor(t, m, p+"post_attention_layernorm.weight")
		router := cpuOracleTensor(t, m, mixtralOracleRouterSrc(l))

		// --- attention sub-layer, Pre-Norm: x += self_attn(rmsnorm(x)) ---
		q := make([][]float32, seq)
		k := make([][]float32, seq)
		v := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			xn := cpuOracleRMSNorm(x[tt], inNorm, eps)
			q[tt] = cpuOracleMatVec(wq, xn, nH*hd, H) // bias=False everywhere
			k[tt] = cpuOracleMatVec(wk, xn, nKV*hd, H)
			v[tt] = cpuOracleMatVec(wv, xn, nKV*hd, H)
			for h := 0; h < nH; h++ {
				cpuOracleRope(q[tt][h*hd:(h+1)*hd], tt, hd, theta)
			}
			for h := 0; h < nKV; h++ {
				cpuOracleRope(k[tt][h*hd:(h+1)*hd], tt, hd, theta)
			}
		}
		scale := float32(1.0 / math.Sqrt(float64(hd))) // head_dim**-0.5
		for tt := 0; tt < seq; tt++ {
			concat := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				kvh := h / grp
				if spec.kvHeadModulo {
					kvh = h % nKV
				}
				qh := q[tt][h*hd : (h+1)*hd]
				scores := make([]float32, tt+1) // causal, no sliding window
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
			attnOut := cpuOracleMatVec(wo, concat, H, nH*hd)
			for i := 0; i < H; i++ {
				x[tt][i] += attnOut[i]
			}
		}

		// --- MoE sub-layer, Pre-Norm: x += block_sparse_moe(rmsnorm(x)) ---
		for tt := 0; tt < seq; tt++ {
			xn := cpuOracleRMSNorm(x[tt], postNorm, eps)
			picks := mixtralOracleRoute(cpuOracleMatVec(router, xn, spec.experts, H), spec)

			delta := make([]float32, H)
			for _, pk := range picks {
				in := xn
				outScale := pk.weight
				if spec.weightExpertInput {
					// WRONG-BY-CONSTRUCTION variant: scale the expert's INPUT.
					in = make([]float32, H)
					for i := 0; i < H; i++ {
						in[i] = xn[i] * pk.weight
					}
					outScale = 1
				}
				w1 := cpuOracleTensor(t, m, mixtralOracleExpertSrc(l, pk.expert, "w1"))
				w2 := cpuOracleTensor(t, m, mixtralOracleExpertSrc(l, pk.expert, "w2"))
				w3 := cpuOracleTensor(t, m, mixtralOracleExpertSrc(l, pk.expert, "w3"))
				gateW, upW := w1, w3
				if spec.swapW1W3 {
					gateW, upW = w3, w1
				}
				// MixtralBlockSparseTop2MLP: w2(silu(w1(x)) * w3(x)).
				g := cpuOracleMatVec(gateW, in, I, H)
				u := cpuOracleMatVec(upW, in, I, H)
				for i := 0; i < I; i++ {
					g[i] = cpuOracleSilu(g[i]) * u[i]
				}
				out := cpuOracleMatVec(w2, g, H, I)
				for i := 0; i < H; i++ {
					delta[i] += outScale * out[i] // gate weight scales the OUTPUT
				}
			}
			for i := 0; i < H; i++ {
				x[tt][i] += delta[i]
			}
		}
	}

	norm := cpuOracleTensor(t, m, "model.norm.weight")
	head := cpuOracleTensor(t, m, "lm_head.weight") // untied
	if spec.tiedHead {
		head = embed
	}
	logits := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		xf := cpuOracleRMSNorm(x[tt], norm, eps)
		logits[tt] = cpuOracleMatVec(head, xf, V, H)
	}
	return logits
}

var mixtralOracleIDs = []int{3, 17, 5, 23, 41, 2, 19}

// mixtralOracleAssertDerivedAxes checks that config.go lowered the published Mixtral
// config to the axes the reference hardcodes, and that the loader actually performed the
// block_sparse_moe aliasing. This is the structural half of the witness — it names the
// broken axis directly instead of leaving only a logit delta — but it is NOT the gate:
// the numeric comparison stands on its own.
func mixtralOracleAssertDerivedAxes(t *testing.T, m *Model, topK int) {
	t.Helper()
	cfg := m.Cfg
	if cfg.BlockTopology != PreNorm {
		t.Fatalf("mixtral derived topology = %v, want PreNorm", cfg.BlockTopology)
	}
	if !cfg.IsMoE() || cfg.NumExperts != 5 || cfg.NumExpertsPerTok != topK {
		t.Fatalf("mixtral MoE axes = experts:%d topk:%d IsMoE:%v, want 5/%d/true",
			cfg.NumExperts, cfg.NumExpertsPerTok, cfg.IsMoE(), topK)
	}
	if !cfg.NormTopKProb {
		t.Fatal("mixtral derived NormTopKProb = false; MixtralSparseMoeBlock renormalizes the selected top-k weights (the fixture leaves the field zero on purpose, so this is the derivation under test)")
	}
	if cfg.DenseMLP || cfg.ActGeluTanh || cfg.ActGeluErf {
		t.Fatalf("mixtral activation axes = dense:%v tanh:%v erf:%v, want SwiGLU/SiLU",
			cfg.DenseMLP, cfg.ActGeluTanh, cfg.ActGeluErf)
	}
	if cfg.LayerNorm || cfg.NormGain1p || cfg.QKNorm || cfg.AttentionBias {
		t.Fatalf("mixtral norm/bias axes = layernorm:%v gain1p:%v qknorm:%v attnbias:%v, want all false",
			cfg.LayerNorm, cfg.NormGain1p, cfg.QKNorm, cfg.AttentionBias)
	}
	if cfg.windowForLayer(0) != -1 {
		t.Fatalf("mixtral derived window = %d, want -1 (sliding_window is null on both published checkpoints)", cfg.windowForLayer(0))
	}
	if _, ok := m.ffnForLayer(0).(moeFFN); !ok {
		t.Fatalf("mixtral layer 0 FFN = %T, want moeFFN", m.ffnForLayer(0))
	}
	if !m.has("lm_head.weight") {
		t.Fatal("mixtral fixture lost its untied lm_head")
	}

	// The w1/w2/w3 -> gate/up/down aliasing must have happened, and must have happened
	// in the RIGHT direction: w1 is the gate, w3 the up, w2 the down. All three are
	// zero-copy manifest aliases, so identical offsets pin the mapping exactly. A
	// w1<->w3 swap here is shape-clean and would otherwise only show as a logit delta.
	for _, tc := range []struct{ canonical, source string }{
		{"model.layers.0.mlp.gate.weight", mixtralOracleRouterSrc(0)},
		{"model.layers.0.mlp.experts.0.gate_proj.weight", mixtralOracleExpertSrc(0, 0, "w1")},
		{"model.layers.0.mlp.experts.0.down_proj.weight", mixtralOracleExpertSrc(0, 0, "w2")},
		{"model.layers.0.mlp.experts.0.up_proj.weight", mixtralOracleExpertSrc(0, 0, "w3")},
	} {
		got, ok := m.manifest[tc.canonical]
		if !ok {
			t.Fatalf("block_sparse_moe alias missing: %s was never materialized", tc.canonical)
		}
		want, ok := m.manifest[tc.source]
		if !ok {
			t.Fatalf("fixture source tensor %q missing", tc.source)
		}
		if got.Offset != want.Offset {
			t.Fatalf("%s aliases offset %d, want %s at offset %d", tc.canonical, got.Offset, tc.source, want.Offset)
		}
	}
}

// mixtralOracleRouteIsLive fails if the fixture's routing is degenerate — if the same
// expert set fires for every token, or if fewer than top_k+1 distinct experts are ever
// selected, then the top-k / selection axes are not actually under test even though the
// numeric comparison would still be green.
func mixtralOracleRouteIsLive(t *testing.T, m *Model, ids []int, spec mixtralOracleSpec) {
	t.Helper()
	seen := map[int]bool{}
	sets := map[string]bool{}
	// Re-run the reference's router on the reference's own hidden states would need the
	// whole forward; the layer-0 router over the embedding-normed input is enough to show
	// the selection varies, and is what the first MoE sub-layer routes closest to.
	inNorm := cpuOracleTensor(t, m, mixtralOracleLayerSrc(0)+"post_attention_layernorm.weight")
	router := cpuOracleTensor(t, m, mixtralOracleRouterSrc(0))
	embed := cpuOracleTensor(t, m, "model.embed_tokens.weight")
	H := spec.hidden
	for _, id := range ids {
		xn := cpuOracleRMSNorm(embed[id*H:(id+1)*H], inNorm, spec.eps)
		picks := mixtralOracleRoute(cpuOracleMatVec(router, xn, spec.experts, H), spec)
		key := ""
		for _, pk := range picks {
			seen[pk.expert] = true
			key += strconv.Itoa(pk.expert) + ","
		}
		sets[key] = true
	}
	if len(seen) <= spec.topK {
		t.Fatalf("fixture routing is degenerate: only %d distinct experts ever selected (top_k=%d, experts=%d) — the selection axis is not under test",
			len(seen), spec.topK, spec.experts)
	}
	if len(sets) < 2 {
		t.Fatalf("fixture routing is degenerate: every token selected the same expert set %v — the router is not under test", sets)
	}
}

// runMixtralOracle compares the production cacheless Forward AND the cached Prefill/Step
// decode path against the independent reference at every position.
func runMixtralOracle(t *testing.T, topK int) {
	t.Helper()
	m := newMixtralOracleModel(t, topK)
	mixtralOracleAssertDerivedAxes(t, m, topK)
	spec := mixtralOracleSpecFor(m.Cfg)
	mixtralOracleRouteIsLive(t, m, mixtralOracleIDs, spec)

	ids := mixtralOracleIDs
	ref := mixtralReference(t, m, ids, spec)

	act := m.Forward(ids)
	for tt := range ids {
		if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > cpuOracleTol {
			t.Errorf("Forward logits pos %d: max|delta| vs reference = %.3e (tol %.0e)", tt, d, cpuOracleTol)
		}
	}

	// Cached decode path: Prefill then Step must match the reference at the same
	// positions (the reference is cacheless, so Step(id) at position len(ids) is compared
	// against a reference run over the extended prompt). This half is not redundant for
	// an MoE family: the decode path routes through moeFFN's residentKernel batched-expert
	// branch, which the cacheless prefill does not take.
	s := m.NewSession()
	pf := s.Prefill(ids)
	if d := cpuOracleMaxAbsDiff(pf, ref[len(ids)-1]); d > cpuOracleTol {
		t.Errorf("Prefill last logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
	next := 11
	st := s.Step(next)
	extRef := mixtralReference(t, m, append(append([]int(nil), ids...), next), spec)
	if d := cpuOracleMaxAbsDiff(st, extRef[len(ids)]); d > cpuOracleTol {
		t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
}

// TestMixtralCPUNumericOracle is the Mixtral×cpu M4 witness: the production forward must
// reproduce the independent HF-semantics reference on every position within cpuOracleTol.
// It covers the PreNorm block with two distinct RMSNorms, bias-free causal GQA at
// head_dim**-0.5 with rotate_half RoPE, the untied lm_head, the block_sparse_moe
// w1/w2/w3 aliasing, and — the family-defining part — the full router: softmax over ALL
// experts, top-k AFTER the softmax, renormalization of the survivors, only the selected
// experts contributing, and the gate weight scaling the expert OUTPUT.
//
// top_k=2 is the published Mixtral geometry; top_k=3 is run as well so the renormalization
// divisor is exercised over more than a pair (with k=2 a sum-of-two renormalization can be
// mistaken for a pairwise ratio).
//
// If this test reds, the honesty fence demotes the cell back to M3 (drop the covmatrix
// OracleInCI bit with it).
func TestMixtralCPUNumericOracle(t *testing.T) {
	for _, topK := range []int{2, 3} {
		t.Run("top_k="+strconv.Itoa(topK), func(t *testing.T) { runMixtralOracle(t, topK) })
	}
}

// mixtralOracleAssertAxisLive proves one Mixtral semantic is not numerically inert in this
// fixture: the reference under the HF spec must differ from the reference under a
// deliberately WRONG spec by more than the tolerance. Without this, "production matches the
// reference" could be green simply because the axis does nothing at this scale — a vacuous
// witness. This compares reference-to-reference, so it cannot be satisfied by anything
// production does.
func mixtralOracleAssertAxisLive(t *testing.T, m *Model, base [][]float32, wrong mixtralOracleSpec, axis string) {
	t.Helper()
	alt := mixtralReference(t, m, mixtralOracleIDs, wrong)
	var worst float64
	for tt := range mixtralOracleIDs {
		if d := cpuOracleMaxAbsDiff(base[tt], alt[tt]); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Errorf("axis %q is inert in this fixture (max|delta| = %.3e <= tol %.0e): the oracle would stay green with that axis wrong",
			axis, worst, cpuOracleTol)
	}
}

// TestMixtralCPUNumericOracleAxesAreLive is the anti-vacuity gate for every axis a size or
// scale choice could accidentally switch off. Each is mutated to a wrong-but-plausible
// value and the reference must move. The router entries are the point of the file: all
// four of them keep every tensor shape identical, so nothing but a numeric oracle can see
// them.
func TestMixtralCPUNumericOracleAxesAreLive(t *testing.T) {
	m := newMixtralOracleModel(t, 2)
	spec := mixtralOracleSpecFor(m.Cfg)
	base := mixtralReference(t, m, mixtralOracleIDs, spec)

	// --- the router divergences named in the header ---

	noRenorm := spec
	noRenorm.normTopK = false
	mixtralOracleAssertAxisLive(t, m, base, noRenorm, "norm_topk_prob (renormalize the selected router weights)")

	// Selection ORDER, exercised in the only regime where it is observable. With the
	// renormalization ON (Mixtral's setting) softmax-over-all -> top-k -> renorm is an
	// EXACT identity with top-k -> softmax-over-the-survivors: softmax is monotonic so the
	// selection is unchanged, and the shared denominator cancels in the renormalization.
	// Asserting the order were live against the renormalized base would therefore be a
	// false claim, and the measurement says so — 1.9e-7, float noise. Turning the
	// renormalization off separates them, so the order is pinned there instead. This is
	// deliberately NOT a skip or a widened tolerance: the axis is fully asserted, just in
	// the regime where the arithmetic makes it distinguishable.
	rawWeights := spec
	rawWeights.normTopK = false
	rawBase := mixtralReference(t, m, mixtralOracleIDs, rawWeights)
	preSoftmaxTopK := rawWeights
	preSoftmaxTopK.topKBeforeSoftmax = true
	mixtralOracleAssertAxisLive(t, m, rawBase, preSoftmaxTopK, "top-k AFTER the full-width softmax (un-renormalized regime)")

	inputWeighted := spec
	inputWeighted.weightExpertInput = true
	mixtralOracleAssertAxisLive(t, m, base, inputWeighted, "the gate weight scales the expert OUTPUT, not its input")

	dense := spec
	dense.allExperts = true
	mixtralOracleAssertAxisLive(t, m, base, dense, "only the top-k experts contribute")

	// --- expert weight layout ---

	swapped := spec
	swapped.swapW1W3 = true
	mixtralOracleAssertAxisLive(t, m, base, swapped, "w1 is the gate and w3 the up projection (shape-identical, so a swap is silent)")

	// --- the remaining derived axes ---

	topK1 := spec
	topK1.topK = 1
	mixtralOracleAssertAxisLive(t, m, base, topK1, "num_experts_per_tok")

	fewerExperts := spec
	fewerExperts.experts = m.Cfg.NumExperts - 1
	mixtralOracleAssertAxisLive(t, m, base, fewerExperts, "num_local_experts (router width)")

	llamaTheta := spec
	llamaTheta.theta = 10000
	mixtralOracleAssertAxisLive(t, m, base, llamaTheta, "rope_theta")

	moduloKV := spec
	moduloKV.kvHeadModulo = true
	mixtralOracleAssertAxisLive(t, m, base, moduloKV, "GQA grouping (num_key_value_heads)")

	tied := spec
	tied.tiedHead = true
	mixtralOracleAssertAxisLive(t, m, base, tied, "tie_word_embeddings (Mixtral carries a separate lm_head)")

	fullHead := spec
	fullHead.headDim = m.Cfg.HiddenSize / m.Cfg.NumHeads
	mixtralOracleAssertAxisLive(t, m, base, fullHead, "head_dim (nH*head_dim != hidden_size in this fixture)")
}

// TestMixtralCPUNumericOracleIsSensitive proves the comparison is non-vacuous on the
// WEIGHT axis: perturbing ONE raw fixture element must move the compared logits far beyond
// the tolerance. Every distinct weight ROLE is listed, because a reference that silently
// ignored one of them would stay green without it: both norms and the final norm, the
// untied head, the embedding, o_proj, all three expert projections under their SOURCE
// names, and the router.
//
// Two per-case knobs, both properties of the model rather than weakened assertions:
//
//   - elem picks WHICH element is perturbed. model.embed_tokens.weight needs it: element 0
//     is row 0, i.e. token id 0, which mixtralOracleIDs never contains, so perturbing it
//     is a genuine no-op (measured 1.8e-7) — and on an UNTIED family the embedding row is
//     not reachable through the head either. The embedding case therefore perturbs the row
//     of a token that is actually in the prompt.
//   - delta is larger for the router. Because the selected weights are renormalized by
//     their own sum, the shared softmax denominator cancels, so a router-weight change is
//     observable only if it flips the selection or moves a SELECTED expert's logit relative
//     to the other selected ones. A perturbation of one element of expert 0's row must
//     therefore be big enough to pull expert 0 across the top-k boundary for some token.
//
// Both are per-case rather than global so they stay visible instead of being smuggled into
// a shared constant.
func TestMixtralCPUNumericOracleIsSensitive(t *testing.T) {
	const H = 24 // mixtralOracleCfg HiddenSize, so elem can name an embedding ROW
	for _, tc := range []struct {
		name   string
		tensor string
		elem   int
		delta  float32
	}{
		{"embed_tokens", "model.embed_tokens.weight", mixtralOracleIDs[0] * H, 0.5},
		{"input_layernorm", "model.layers.0.input_layernorm.weight", 0, 0.5},
		{"post_attention_layernorm", "model.layers.0.post_attention_layernorm.weight", 0, 0.5},
		{"q_proj", "model.layers.0.self_attn.q_proj.weight", 0, 0.5},
		{"o_proj", "model.layers.0.self_attn.o_proj.weight", 0, 0.5},
		{"expert_w1_gate", mixtralOracleExpertSrc(0, 0, "w1"), 0, 0.5},
		{"expert_w2_down", mixtralOracleExpertSrc(0, 0, "w2"), 0, 0.5},
		{"expert_w3_up", mixtralOracleExpertSrc(0, 0, "w3"), 0, 0.5},
		{"router_gate", mixtralOracleRouterSrc(0), 0, 4},
		{"final_norm", "model.norm.weight", 0, 0.5},
		{"lm_head", "lm_head.weight", 0, 0.5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newMixtralOracleModel(t, 2)
			ids := mixtralOracleIDs
			ref := mixtralReference(t, m, ids, mixtralOracleSpecFor(m.Cfg))

			meta, ok := m.manifest[tc.tensor]
			if !ok {
				t.Fatalf("fixture tensor %q missing", tc.tensor)
			}
			at := meta.Offset + tc.elem*4
			orig := math.Float32frombits(binary.LittleEndian.Uint32(m.raw[at:]))
			binary.LittleEndian.PutUint32(m.raw[at:], math.Float32bits(orig+tc.delta))

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
