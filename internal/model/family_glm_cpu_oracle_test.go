package model

// family_glm_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU numeric
// oracle for GLM-5.2's Dynamic-Sparse-Attention family (model_type glm_moe_dsa;
// covmatrix family "GLM-5.2-DSA", Topology SparseAttn).
//
// Why this family needs its own oracle. GLM-5.2-DSA does NOT resolve through
// tensor_resolver.go's resolveSpecFor token switch — that is why its covmatrix row
// carries an empty ResolverToken. It is selected by Config.usesMLAMoELayout()
// (config.go: isGLM -> isGLMMoeDsa -> usesMLAMoELayout), which routes forwardHiddenRows
// to layerGLMDsa: an entirely different block body from the dense m.layer(). Inside one
// layer it runs a q_a/q_b LoRA query projection, a kv_a/kv_b latent KV projection (MLA),
// TWO different rotary conventions, a learned "lightning indexer" that selects a top-k
// subset of keys per query, and an IndexShare rule that lets a layer reuse the previous
// full-indexer layer's selection. It also has its own KV cache (glmDsaKVCache) and its
// own inv_freq denominator rule (invFreqDenom -> qk_rope_head_dim, NOT head_dim). None
// of that is witnessed by the Llama/Qwen reference in family_cpu_oracle_test.go.
//
// Independence discipline (same contract as the other family oracles): the reference
// below is a plain scalar transcription of the family's published dataflow —
// DeepSeek-V3-style MLA attention plus the DSA lightning indexer. Tensors are decoded
// straight from the manifest bytes; the matmuls, norms, softmax, BOTH rotary
// conventions, the index score and the top-k selection are naive in-order scalar loops
// written here. It calls NONE of rmsnorm / layernorm / dot / softmaxInPlace /
// applyRopeRow / glmDsaApplyInterleavedRoPE / glmDsaApplyIndexerRoPE / dsaIndexScores /
// dsaTopKIndices / residentMatMulBatch, and it hardcodes the PreNorm block placement
// rather than routing through cfg.BlockTopology. The family axes it needs (the rotary
// denominator, the two rotary conventions, the softmax scale, the IndexShare pattern,
// index_topk) live in glmOracleSpec as literals written independently of glmOracleCfg,
// so a config-derivation bug diverges the two sides instead of cancelling.
//
// Two things make this oracle non-vacuous for the SparseAttn topology specifically:
//
//  1. The fixture prompt (9 positions) is longer than index_topk (3), so every query
//     from position 3 on genuinely PRUNES keys. runGLMDsaOracle ASSERTS the pruning
//     happened (glmOracleTrace.dropped) instead of assuming it. A prompt shorter than
//     index_topk would run the dense causal path under a sparse name — vacuous for the
//     very topology the covmatrix cell claims.
//  2. The discrete selection itself is compared: the reference's independently computed
//     top-k set is checked against production's glmDsaTopKIndicesNormed on every
//     full-indexer layer. A logits-only comparison can hide a selection bug behind
//     softmax mass. The smallest score margin at the top-k boundary is asserted
//     non-zero and logged, so a near-tie — which would make the discrete decision
//     precision-fragile rather than semantically pinned — cannot pass silently.
//
// What this file does NOT witness, stated plainly:
//
//   - The ROUTED-MoE FFN (glmMoeFFN / glmRoute: sigmoid scoring, e_score_correction_bias,
//     n_group/topk_group group routing, shared experts, routed_scaling_factor). The
//     fixture carries the dense SwiGLU MLP that NewSyntheticGLMDsa builds, so the MoE
//     router is out of scope here — the same kind of bounded claim covmatrix already
//     records for Falcon (7B multi_query only).
//   - The two indexer SCALE constants, 1/sqrt(index_head_dim) and 1/sqrt(index_n_heads),
//     are provably UNOBSERVABLE end to end: relu(c*d) == c*relu(d) for c>0, so both
//     multiply every score by the same positive constant, and the scores are consumed
//     only by a ranking (dsaTopKIndices). No end-to-end witness can see them, so
//     TestGLMDsaCPUNumericOracleAxesAreLive deliberately does not claim them.
//   - The indexer's k_norm is an nn.LayerNorm and production applies eps 1e-6
//     (glmDsaInnerNormEps) where torch's nn.LayerNorm default is 1e-5. Nothing in this
//     tree pins which constant GLM-5.2 publishes, so the reference transcribes
//     production's 1e-6 and this stays an explicitly named unverified constant rather
//     than a silent assumption. It can only change an outcome at a near-tie in the
//     ranking, and the margin assertion shows this fixture has none.
//   - The MTP / self-speculation head. RetainMTP is off by default, so mtp tensors are
//     dropped at load; the fixture emits none and the builder asserts the manifest is
//     free of them, so MTP cannot silently change the path under test.

import (
	"encoding/binary"
	"math"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// glmOracleIDs is the fixture prompt. len(ids)=9 is deliberately > index_topk=3 so the
// DSA indexer must actually discard keys from position 3 on.
var glmOracleIDs = []int{3, 17, 5, 23, 31, 2, 19, 11, 27}

// glmOracleSpec is the family semantics the reference hardcodes. Every field is the
// published GLM-5.2-DSA / DeepSeek-MLA rule written out HERE as a literal, never read
// out of Config — that is what lets the numeric comparison witness config.go's
// derivation as well as the kernel.
type glmOracleSpec struct {
	layers   int
	hidden   int
	inter    int
	vocab    int
	heads    int
	qLora    int
	kvLora   int
	qkNope   int
	qkRope   int
	vHead    int
	idxHeads int
	idxDim   int
	// topK is index_topk: the number of key positions the lightning indexer keeps.
	topK int
	// indexerTypes is the per-layer IndexShare pattern; a "shared" layer reuses the
	// previous "full" layer's selection verbatim. Layer 0 must be "full".
	indexerTypes []string
	// ropeDenom is the inv_freq denominator for BOTH rotary conventions in this family:
	// inv[j] = 1/theta^(2j/qk_rope_head_dim). NOT head_dim — the GGUF sets head_dim from
	// the larger MLA latent width, so denominating by it silently detunes every frequency.
	ropeDenom int
	theta     float64
	// outerEps is rms_norm_eps (input_layernorm / post_attention_layernorm / model.norm).
	outerEps float32
	// innerEps is the eps of q_a_layernorm / kv_a_layernorm (DeepSeekV3RMSNorm's default
	// 1e-6, NOT rms_norm_eps) and of the indexer's k_norm (see the header residual).
	innerEps float32
	// attnScaleDenom is the MLA softmax scale denominator: qk_nope_head_dim +
	// qk_rope_head_dim, i.e. scale = (qk_nope+qk_rope)**-0.5.
	attnScaleDenom int
	// idxScaleDenom / weightDenom are the two indexer scales. Both are rank-inert (see
	// the header); they are carried so the reference is a complete transcription.
	idxScaleDenom int
	weightDenom   int
	// mlaRopeInterleaved selects DeepSeek's apply_rotary_pos_emb for the MLA q_pe/k_pe
	// halves: de-interleave [x0,x1,x2,...] into [x0,x2,...|x1,x3,...] and then rotate_half.
	mlaRopeInterleaved bool
	// indexerRopeInterleaved selects the SAME convention for the indexer's rotary prefix.
	// The family uses plain rotate_half there — the two conventions coexist in one layer.
	indexerRopeInterleaved bool
	// attentionBias is HF attention_bias: q_a_proj / kv_a_proj_with_mqa / o_proj carry a
	// learned bias (DeepseekV3Attention wires exactly those three).
	attentionBias bool
}

func glmOracleSpecFor(attentionBias bool) glmOracleSpec {
	return glmOracleSpec{
		layers:                 3,
		hidden:                 32,
		inter:                  40,
		vocab:                  37,
		heads:                  2,
		qLora:                  12,
		kvLora:                 8,
		qkNope:                 6,
		qkRope:                 4,
		vHead:                  14,
		idxHeads:               3,
		idxDim:                 8,
		topK:                   3,
		indexerTypes:           []string{"full", "shared", "full"},
		ropeDenom:              4, // qk_rope_head_dim (head_dim is 12 — deliberately different)
		theta:                  10000,
		outerEps:               1e-5,
		innerEps:               1e-6,
		attnScaleDenom:         10, // qk_nope_head_dim + qk_rope_head_dim
		idxScaleDenom:          8,  // index_head_dim
		weightDenom:            3,  // index_n_heads
		mlaRopeInterleaved:     true,
		indexerRopeInterleaved: false,
		attentionBias:          attentionBias,
	}
}

func (s glmOracleSpec) indexerKind(l int) string {
	if l < 0 || l >= len(s.indexerTypes) {
		return "full"
	}
	return s.indexerTypes[l]
}

// glmOracleCfg is the tiny GLM-5.2-DSA fixture config — what config.json would say.
// Every width is deliberately distinct: head_dim 12 != qk_head 10 != v_head 14 !=
// index_head_dim 8 != q_lora 12/kv_lora 8, and hidden 32 shares no value with any head
// width, so a boundary/stride/denominator conflation cannot cancel.
func glmOracleCfg(attentionBias bool) Config {
	_ = attentionBias // the bias lineage changes the tensor roster, not the config axes
	return Config{
		HiddenSize:        32,
		NumLayers:         3,
		NumHeads:          2,
		NumKVHeads:        2,
		HeadDim:           12,
		IntermediateSize:  40,
		VocabSize:         37,
		ModelType:         "glm_moe_dsa",
		Architectures:     []string{"Glm5MoeDsaForCausalLM"},
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		QLoraRank:         12,
		KVLoraRank:        8,
		QKNopeHeadDim:     6,
		QKRopeHeadDim:     4,
		VHeadDim:          14,
		IndexNHeads:       3,
		IndexHeadDim:      8,
		IndexTopK:         3,
		IndexerTypes:      []string{"full", "shared", "full"},
		TieWordEmbeddings: false, // untied lm_head so the head is a distinct weight role
	}
}

// glmOracleFill gives every norm weight a distinct NON-UNIT gain and every bias a
// distinct NON-ZERO offset, so norm-gain application and bias routing are numerically
// live rather than masked by 1.0 / 0.0.
func glmOracleFill(name string, next func() float32) float32 {
	switch {
	case strings.HasSuffix(name, ".bias"):
		return 0.1 + 0.25*next()
	case isCPUOracleNormWeight(name):
		return 1 + 0.25*next()
	}
	return synthMatmulFill(name, next)
}

// newGLMOracleModel builds the fixture on GLM-5.2-DSA's REAL tensor roster and loads it
// through newModel — the single construction point every loader funnels through.
func newGLMOracleModel(t *testing.T, attentionBias bool) *Model {
	t.Helper()
	cfg := glmOracleCfg(attentionBias)
	if err := cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	nH := cfg.NumHeads
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	qLora, kvLora := cfg.QLoraRank, cfg.KVLoraRank
	qkNope, qkRope, vHead := cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.VHeadDim
	idxH, idxD := cfg.IndexNHeads, cfg.IndexHeadDim

	type ts = synthTensor
	tensors := []ts{{"model.embed_tokens.weight", []int{V, H}}}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		tensors = append(tensors,
			ts{p + "input_layernorm.weight", []int{H}},
			// MLA: a LoRA query projection through a normed q_lora_rank latent...
			ts{p + "self_attn.q_a_proj.weight", []int{qLora, H}},
			ts{p + "self_attn.q_a_layernorm.weight", []int{qLora}},
			ts{p + "self_attn.q_b_proj.weight", []int{nH * (qkNope + qkRope), qLora}},
			// ...and a joint KV latent whose tail carries the ONE rope key shared by
			// every head (the "with_mqa" suffix).
			ts{p + "self_attn.kv_a_proj_with_mqa.weight", []int{kvLora + qkRope, H}},
			ts{p + "self_attn.kv_a_layernorm.weight", []int{kvLora}},
			ts{p + "self_attn.kv_b_proj.weight", []int{nH * (qkNope + vHead), kvLora}},
			ts{p + "self_attn.o_proj.weight", []int{H, nH * vHead}},
		)
		if attentionBias {
			tensors = append(tensors,
				ts{p + "self_attn.q_a_proj.bias", []int{qLora}},
				ts{p + "self_attn.kv_a_proj_with_mqa.bias", []int{kvLora + qkRope}},
				ts{p + "self_attn.o_proj.bias", []int{H}},
			)
		}
		if glmDsaIndexerKindForFixture(cfg, l) == "full" {
			// The learned lightning indexer exists only on full-indexer layers; a
			// shared layer has no indexer tensors at all.
			tensors = append(tensors,
				ts{p + "self_attn.indexer.wq_b.weight", []int{idxH * idxD, qLora}},
				ts{p + "self_attn.indexer.wk.weight", []int{idxD, H}},
				ts{p + "self_attn.indexer.k_norm.weight", []int{idxD}},
				ts{p + "self_attn.indexer.k_norm.bias", []int{idxD}},
				ts{p + "self_attn.indexer.weights_proj.weight", []int{idxH, H}},
			)
		}
		tensors = append(tensors,
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{p + "mlp.gate_proj.weight", []int{I, H}},
			ts{p + "mlp.up_proj.weight", []int{I, H}},
			ts{p + "mlp.down_proj.weight", []int{H, I}},
		)
	}
	tensors = append(tensors,
		ts{"model.norm.weight", []int{H}},
		ts{"lm_head.weight", []int{V, H}},
	)

	man, raw := synthBuildRaw(tensors, glmOracleFill)
	m, err := newModel(cfg, man, raw)
	if err != nil {
		t.Fatalf("newModel(glm_moe_dsa fixture): %v", err)
	}
	// MTP / self-speculation scaffolding must not be in the path under test.
	for name := range m.manifest {
		if strings.Contains(name, "mtp") || strings.Contains(name, "nextn") {
			t.Fatalf("fixture manifest carries an MTP tensor %q — it would change the path under test", name)
		}
	}
	return m
}

// glmDsaIndexerKindForFixture reads the fixture's own IndexerTypes list to decide which
// layers get indexer tensors. Written out here rather than calling glmDsaIndexerKind so
// the ROSTER the reference reads is not defined by the production classifier it tests.
func glmDsaIndexerKindForFixture(cfg Config, l int) string {
	if l < 0 || l >= len(cfg.IndexerTypes) {
		return "full"
	}
	return cfg.IndexerTypes[l]
}

// ---- independent scalar primitives ----------------------------------------
//
// Written out here rather than reused from production. cpuOracleMatVec /
// cpuOracleRMSNorm / cpuOracleSoftmax / cpuOracleSilu (family_cpu_oracle_test.go) are
// the shared ORACLE-side transcriptions and are reused; nothing below routes through a
// production kernel.

// glmOracleLayerNorm is the plain HF nn.LayerNorm: subtract the mean, divide by the
// BIASED std, scale by weight, add bias. The indexer's k_norm is a LayerNorm, not an
// RMSNorm — a conflation here shifts every index score.
func glmOracleLayerNorm(x, w, b []float32, eps float32) []float32 {
	n := float32(len(x))
	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= n
	var ss float32
	for _, v := range x {
		d := v - mean
		ss += d * d
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/n+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = (v-mean)*inv*w[i] + b[i]
	}
	return out
}

// glmOracleInvFreq is inv_freq[j] = 1/theta^(2j/denom) for j in [0, half).
func glmOracleInvFreq(j, denom int, theta float64) float64 {
	return 1.0 / math.Pow(theta, float64(2*j)/float64(denom))
}

// glmOracleInterleavedRope is DeepSeek/GLM apply_rotary_pos_emb for the MLA rope halves:
// the checkpoint stores the rope slice INTERLEAVED ([x0,x1,x2,x3,...]), HF views it as
// (-1, 2) and transposes to [x0,x2,...|x1,x3,...], then applies rotate_half. Returns a
// new slice; the input is left untouched.
func glmOracleInterleavedRope(x []float32, pos, denom int, theta float64) []float32 {
	half := len(x) / 2
	out := make([]float32, len(x))
	for j := 0; j < half; j++ {
		angle := float64(pos) * glmOracleInvFreq(j, denom, theta)
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		a, b := x[2*j], x[2*j+1]
		out[j] = a*c - b*s
		out[j+half] = b*c + a*s
	}
	return out
}

// glmOracleHalfRope is plain rotate_half over a contiguous slice: pair j with j+half.
// This is the convention the DSA indexer uses on the LEADING qk_rope_head_dim of each
// index head — the other of the two conventions that coexist inside one GLM layer.
func glmOracleHalfRope(x []float32, pos, denom int, theta float64) {
	half := len(x) / 2
	for j := 0; j < half; j++ {
		angle := float64(pos) * glmOracleInvFreq(j, denom, theta)
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		a, b := x[j], x[j+half]
		x[j] = a*c - b*s
		x[j+half] = b*c + a*s
	}
}

// glmOracleRopeInto applies the selected convention to a rope slice, in place.
func glmOracleRopeInto(x []float32, pos, denom int, theta float64, interleaved bool) []float32 {
	if interleaved {
		return glmOracleInterleavedRope(x, pos, denom, theta)
	}
	out := append([]float32(nil), x...)
	glmOracleHalfRope(out, pos, denom, theta)
	return out
}

func glmOracleDot32(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// glmOracleDot64 accumulates in f64 from f32 inputs, matching the precision the DSA
// index score is computed at (the selection is a DISCRETE decision, so the reference
// must not introduce a precision gap that could flip a near-tie).
func glmOracleDot64(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func glmOracleAddBias(y, b []float32) {
	if b == nil {
		return
	}
	for i := range y {
		y[i] += b[i]
	}
}

// glmOracleTopK selects the top-k highest scores (ties broken by LOWER key position),
// returns the selected positions sorted ASCENDING (the attention output is a softmax
// over a SET, so order is not semantic), and the score margin between the last kept and
// first dropped candidate (+Inf when nothing was dropped).
func glmOracleTopK(scores []float64, k int) ([]int, float64) {
	idx := make([]int, len(scores))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		if scores[idx[a]] == scores[idx[b]] {
			return idx[a] < idx[b]
		}
		return scores[idx[a]] > scores[idx[b]]
	})
	n := k
	if n > len(idx) {
		n = len(idx)
	}
	margin := math.Inf(1)
	if n < len(idx) {
		margin = scores[idx[n-1]] - scores[idx[n]]
	}
	sel := append([]int(nil), idx[:n]...)
	sort.Ints(sel)
	return sel, margin
}

// ---- the reference forward ------------------------------------------------

// glmOracleTrace is the reference's output plus the sparsity evidence the caller needs
// to prove the sparse path actually ran.
type glmOracleTrace struct {
	logits [][]float32
	// sel[layer][pos] is the key set position pos attended at that layer, ascending.
	sel [][][]int
	// xn[layer][pos] is the layer's normalized input — the exact vector production's
	// indexer consumes, so the discrete selection can be compared directly.
	xn [][][]float32
	// dropped is the total number of (layer,pos) causal keys pruned by the indexer;
	// maxDropped is the largest per-query prune. Both are 0 when the prompt is shorter
	// than index_topk, i.e. when this oracle would be vacuous for SparseAttn.
	dropped    int
	maxDropped int
	// margin is the smallest top-k boundary gap seen; maxScore the largest |score|.
	margin   float64
	maxScore float64
}

// glmOracleReference runs the independent GLM-5.2-DSA forward: per-position logits for
// ids. Every step is the family's published dataflow, hardcoded — MLA q_a/q_b +
// kv_a/kv_b with the interleaved rope on the rope halves, the lightning indexer with
// rotate_half on its rope prefix, top-k under the causal mask, IndexShare reuse, sparse
// softmax, and the PreNorm residual placement.
func glmOracleReference(t *testing.T, m *Model, ids []int, sp glmOracleSpec) glmOracleTrace {
	t.Helper()
	tensor := func(name string) []float32 { return cpuOracleTensor(t, m, name) }

	seq := len(ids)
	H, I, V := sp.hidden, sp.inter, sp.vocab
	nH := sp.heads
	qkNope, qkRope, vHead := sp.qkNope, sp.qkRope, sp.vHead
	qkHead := qkNope + qkRope
	qLora, kvLora := sp.qLora, sp.kvLora
	idxH, idxD := sp.idxHeads, sp.idxDim

	attnScale := float32(1.0 / math.Sqrt(float64(sp.attnScaleDenom)))
	idxScale := 1.0 / math.Sqrt(float64(sp.idxScaleDenom))
	wScale := float32(1.0 / math.Sqrt(float64(sp.weightDenom)))

	embed := tensor("model.embed_tokens.weight")
	x := make([][]float32, seq)
	for p, id := range ids {
		x[p] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}

	tr := glmOracleTrace{
		sel:    make([][][]int, sp.layers),
		xn:     make([][][]float32, sp.layers),
		margin: math.Inf(1),
	}
	var shared [][]int

	for l := 0; l < sp.layers; l++ {
		lp := "model.layers." + strconv.Itoa(l) + "."
		ap := lp + "self_attn."

		// PreNorm: the sub-layer reads rms(x); the residual add happens at the end.
		inW := tensor(lp + "input_layernorm.weight")
		xn := make([][]float32, seq)
		for p := 0; p < seq; p++ {
			xn[p] = cpuOracleRMSNorm(x[p], inW, sp.outerEps)
		}
		tr.xn[l] = xn

		// The q_lora latent is SHARED: both the MLA query and the indexer query read
		// the same normed q_a_proj output. Feeding the indexer a separately normed copy
		// would still be numerically identical here, but reading it once is the family's
		// dataflow and keeps the bias/norm roles bound to a single site.
		wqa := tensor(ap + "q_a_proj.weight")
		qaNorm := tensor(ap + "q_a_layernorm.weight")
		var qaBias []float32
		if sp.attentionBias {
			qaBias = tensor(ap + "q_a_proj.bias")
		}
		qLat := make([][]float32, seq)
		for p := 0; p < seq; p++ {
			r := cpuOracleMatVec(wqa, xn[p], qLora, H)
			glmOracleAddBias(r, qaBias)
			qLat[p] = cpuOracleRMSNorm(r, qaNorm, sp.innerEps)
		}

		// ---- the DSA lightning indexer -------------------------------------
		var sel [][]int
		switch sp.indexerKind(l) {
		case "shared":
			if shared == nil {
				t.Fatalf("layer %d is a shared-indexer layer with no preceding full layer", l)
			}
			sel = make([][]int, seq)
			for p := range shared {
				sel[p] = append([]int(nil), shared[p]...)
			}
		case "full":
			wqb := tensor(ap + "indexer.wq_b.weight")
			wk := tensor(ap + "indexer.wk.weight")
			knW := tensor(ap + "indexer.k_norm.weight")
			knB := tensor(ap + "indexer.k_norm.bias")
			wproj := tensor(ap + "indexer.weights_proj.weight")

			iq := make([][][]float32, seq)
			ik := make([][]float32, seq)
			iw := make([][]float32, seq)
			for p := 0; p < seq; p++ {
				full := cpuOracleMatVec(wqb, qLat[p], idxH*idxD, qLora)
				iq[p] = make([][]float32, idxH)
				for h := 0; h < idxH; h++ {
					head := append([]float32(nil), full[h*idxD:(h+1)*idxD]...)
					// Only the LEADING qk_rope_head_dim of each index head rotates;
					// the tail (idxD-qkRope) stays untouched.
					copy(head[:qkRope], glmOracleRopeInto(head[:qkRope], p, sp.ropeDenom, sp.theta, sp.indexerRopeInterleaved))
					iq[p][h] = head
				}
				k := glmOracleLayerNorm(cpuOracleMatVec(wk, xn[p], idxD, H), knW, knB, sp.innerEps)
				copy(k[:qkRope], glmOracleRopeInto(k[:qkRope], p, sp.ropeDenom, sp.theta, sp.indexerRopeInterleaved))
				ik[p] = k
				w := cpuOracleMatVec(wproj, xn[p], idxH, H)
				for i := range w {
					w[i] *= wScale
				}
				iw[p] = w
			}

			sel = make([][]int, seq)
			for p := 0; p < seq; p++ {
				// Causal mask: only keys at positions <= p are candidates.
				scores := make([]float64, p+1)
				for kp := 0; kp <= p; kp++ {
					var s float64
					for h := 0; h < idxH; h++ {
						d := glmOracleDot64(iq[p][h], ik[kp]) * idxScale
						if d < 0 {
							d = 0 // relu
						}
						s += float64(iw[p][h]) * d
					}
					scores[kp] = s
					if a := math.Abs(s); a > tr.maxScore {
						tr.maxScore = a
					}
				}
				var margin float64
				sel[p], margin = glmOracleTopK(scores, sp.topK)
				if margin < tr.margin {
					tr.margin = margin
				}
			}
			shared = sel
		default:
			t.Fatalf("layer %d has unknown indexer type %q", l, sp.indexerKind(l))
		}
		tr.sel[l] = sel
		for p := 0; p < seq; p++ {
			d := (p + 1) - len(sel[p])
			tr.dropped += d
			if d > tr.maxDropped {
				tr.maxDropped = d
			}
		}

		// ---- MLA attention over the selected keys --------------------------
		wqb2 := tensor(ap + "q_b_proj.weight")
		wkva := tensor(ap + "kv_a_proj_with_mqa.weight")
		kvaNorm := tensor(ap + "kv_a_layernorm.weight")
		wkvb := tensor(ap + "kv_b_proj.weight")
		wo := tensor(ap + "o_proj.weight")
		var kvaBias, oBias []float32
		if sp.attentionBias {
			kvaBias = tensor(ap + "kv_a_proj_with_mqa.bias")
			oBias = tensor(ap + "o_proj.bias")
		}

		q := make([][]float32, seq)  // [seq][nH*qkHead]
		kk := make([][]float32, seq) // [seq][nH*qkHead]
		vv := make([][]float32, seq) // [seq][nH*vHead]
		for p := 0; p < seq; p++ {
			qf := cpuOracleMatVec(wqb2, qLat[p], nH*qkHead, qLora)
			c := cpuOracleMatVec(wkva, xn[p], kvLora+qkRope, H)
			glmOracleAddBias(c, kvaBias)
			lat := cpuOracleRMSNorm(c[:kvLora], kvaNorm, sp.innerEps)
			// ONE rope key, shared by every head (the MQA half of kv_a_proj_with_mqa).
			kRot := glmOracleRopeInto(c[kvLora:], p, sp.ropeDenom, sp.theta, sp.mlaRopeInterleaved)
			kvf := cpuOracleMatVec(wkvb, lat, nH*(qkNope+vHead), kvLora)

			q[p] = make([]float32, nH*qkHead)
			kk[p] = make([]float32, nH*qkHead)
			vv[p] = make([]float32, nH*vHead)
			for h := 0; h < nH; h++ {
				qs := qf[h*qkHead : (h+1)*qkHead]
				copy(q[p][h*qkHead:], qs[:qkNope])
				copy(q[p][h*qkHead+qkNope:], glmOracleRopeInto(qs[qkNope:], p, sp.ropeDenom, sp.theta, sp.mlaRopeInterleaved))
				// kv_b_proj packs [nope | value] per head; the rope half comes from the
				// shared MQA key, not from kv_b.
				ks := kvf[h*(qkNope+vHead) : (h+1)*(qkNope+vHead)]
				copy(kk[p][h*qkHead:], ks[:qkNope])
				copy(kk[p][h*qkHead+qkNope:], kRot)
				copy(vv[p][h*vHead:], ks[qkNope:])
			}
		}

		attn := make([][]float32, seq)
		for p := 0; p < seq; p++ {
			cat := make([]float32, nH*vHead)
			for h := 0; h < nH; h++ {
				sc := make([]float32, len(sel[p]))
				for i, kp := range sel[p] {
					sc[i] = glmOracleDot32(q[p][h*qkHead:(h+1)*qkHead], kk[kp][h*qkHead:(h+1)*qkHead]) * attnScale
				}
				cpuOracleSoftmax(sc)
				for i, kp := range sel[p] {
					w := sc[i]
					for d := 0; d < vHead; d++ {
						cat[h*vHead+d] += w * vv[kp][h*vHead+d]
					}
				}
			}
			o := cpuOracleMatVec(wo, cat, H, nH*vHead)
			glmOracleAddBias(o, oBias)
			attn[p] = o
		}
		for p := 0; p < seq; p++ {
			for i := 0; i < H; i++ {
				x[p][i] += attn[p][i]
			}
		}

		// ---- dense SwiGLU MLP, PreNorm ------------------------------------
		pw := tensor(lp + "post_attention_layernorm.weight")
		wg := tensor(lp + "mlp.gate_proj.weight")
		wu := tensor(lp + "mlp.up_proj.weight")
		wd := tensor(lp + "mlp.down_proj.weight")
		for p := 0; p < seq; p++ {
			h2 := cpuOracleRMSNorm(x[p], pw, sp.outerEps)
			g := cpuOracleMatVec(wg, h2, I, H)
			u := cpuOracleMatVec(wu, h2, I, H)
			act := make([]float32, I)
			for i := 0; i < I; i++ {
				act[i] = cpuOracleSilu(g[i]) * u[i]
			}
			d := cpuOracleMatVec(wd, act, H, I)
			for i := 0; i < H; i++ {
				x[p][i] += d[i]
			}
		}
	}

	nw := tensor("model.norm.weight")
	head := tensor("lm_head.weight")
	tr.logits = make([][]float32, seq)
	for p := 0; p < seq; p++ {
		xf := cpuOracleRMSNorm(x[p], nw, sp.outerEps)
		tr.logits[p] = cpuOracleMatVec(head, xf, V, H)
	}
	return tr
}

// ---- the witness ----------------------------------------------------------

// glmOracleAssertAxes pins the fixture's own geometry (so no axis is unobservable) and
// checks config.go's derivation landed the published values the reference hardcodes.
func glmOracleAssertAxes(t *testing.T, cfg Config, sp glmOracleSpec) {
	t.Helper()
	if !cfg.isGLM() || !cfg.isGLMMoeDsa() || !cfg.usesMLAMoELayout() {
		t.Fatalf("fixture does not classify as GLM-MoE-DSA: isGLM=%v isGLMMoeDsa=%v usesMLAMoELayout=%v",
			cfg.isGLM(), cfg.isGLMMoeDsa(), cfg.usesMLAMoELayout())
	}
	if cfg.BlockTopology != PreNorm {
		t.Fatalf("glm derived topology = %v, want PreNorm (the reference hardcodes x += body(norm(x)))", cfg.BlockTopology)
	}
	for _, c := range []struct {
		name string
		got  int
		want int
	}{
		{"num_hidden_layers", cfg.NumLayers, sp.layers},
		{"hidden_size", cfg.HiddenSize, sp.hidden},
		{"intermediate_size", cfg.IntermediateSize, sp.inter},
		{"vocab_size", cfg.VocabSize, sp.vocab},
		{"num_attention_heads", cfg.NumHeads, sp.heads},
		{"q_lora_rank", cfg.QLoraRank, sp.qLora},
		{"kv_lora_rank", cfg.KVLoraRank, sp.kvLora},
		{"qk_nope_head_dim", cfg.QKNopeHeadDim, sp.qkNope},
		{"qk_rope_head_dim", cfg.QKRopeHeadDim, sp.qkRope},
		{"v_head_dim", cfg.VHeadDim, sp.vHead},
		{"index_n_heads", cfg.IndexNHeads, sp.idxHeads},
		{"index_head_dim", cfg.IndexHeadDim, sp.idxDim},
		{"index_topk", cfg.IndexTopK, sp.topK},
	} {
		if c.got != c.want {
			t.Fatalf("derived %s = %d, want %d (the reference hardcodes the published value)", c.name, c.got, c.want)
		}
	}
	if len(cfg.IndexerTypes) != len(sp.indexerTypes) {
		t.Fatalf("derived indexer_types length = %d, want %d", len(cfg.IndexerTypes), len(sp.indexerTypes))
	}
	for i := range sp.indexerTypes {
		if cfg.IndexerTypes[i] != sp.indexerTypes[i] {
			t.Fatalf("derived indexer_types[%d] = %q, want %q", i, cfg.IndexerTypes[i], sp.indexerTypes[i])
		}
	}
	// Fixture-quality gates: every axis the reference claims must be OBSERVABLE, i.e.
	// the "wrong but plausible" value must differ from the right one in this fixture.
	if cfg.HeadDim == sp.ropeDenom {
		t.Fatalf("fixture head_dim == qk_rope_head_dim (%d): the inv_freq denominator axis is unobservable", cfg.HeadDim)
	}
	if cfg.HeadDim == sp.attnScaleDenom {
		t.Fatalf("fixture head_dim == qk_head_dim (%d): the attention-scale axis is unobservable", cfg.HeadDim)
	}
	if sp.vHead == sp.qkNope || sp.vHead == sp.attnScaleDenom {
		t.Fatalf("fixture v_head_dim (%d) collides with a q/k width: a kv_b_proj split conflation would cancel", sp.vHead)
	}
	if sp.idxDim == sp.attnScaleDenom || sp.idxDim == sp.vHead {
		t.Fatalf("fixture index_head_dim (%d) collides with an attention width: an indexer/attention conflation would cancel", sp.idxDim)
	}
	if sp.indexerKind(0) != "full" {
		t.Fatal("fixture layer 0 must be a full-indexer layer")
	}
	sharedSeen := false
	for l := 0; l < sp.layers; l++ {
		if sp.indexerKind(l) == "shared" {
			sharedSeen = true
		}
	}
	if !sharedSeen {
		t.Fatal("fixture has no shared-indexer layer: the IndexShare axis is unwitnessed")
	}
}

// glmOracleSelectionWitness compares production's DISCRETE top-k selection against the
// reference's, on every full-indexer layer, for the exact normalized input the reference
// fed its own indexer. This is what a logits-only comparison cannot see: a wrong
// selection whose softmax mass happens to land close.
func glmOracleSelectionWitness(t *testing.T, m *Model, sp glmOracleSpec, tr glmOracleTrace, seq int) {
	t.Helper()
	H := sp.hidden
	for l := 0; l < sp.layers; l++ {
		if sp.indexerKind(l) != "full" {
			continue
		}
		flat := make([]float32, 0, seq*H)
		for _, row := range tr.xn[l] {
			flat = append(flat, row...)
		}
		got, ok := glmDsaTopKIndicesNormed(m, l, flat, seq)
		if !ok {
			t.Fatalf("layer %d: production glmDsaTopKIndicesNormed refused the fixture", l)
		}
		if len(got) != seq {
			t.Fatalf("layer %d: production returned %d selection rows, want %d", l, len(got), seq)
		}
		for p := 0; p < seq; p++ {
			asc := append([]int(nil), got[p]...)
			sort.Ints(asc)
			want := tr.sel[l][p]
			if len(asc) != len(want) {
				t.Fatalf("layer %d pos %d: production selected %d keys %v, reference selected %d keys %v",
					l, p, len(asc), asc, len(want), want)
			}
			for i := range asc {
				if asc[i] != want[i] {
					t.Fatalf("layer %d pos %d: production selected %v, reference selected %v", l, p, asc, want)
				}
			}
		}
	}
}

func runGLMDsaOracle(t *testing.T, attentionBias bool) {
	t.Helper()
	m := newGLMOracleModel(t, attentionBias)
	sp := glmOracleSpecFor(attentionBias)
	glmOracleAssertAxes(t, m.Cfg, sp)

	ids := glmOracleIDs
	if len(ids) <= sp.topK {
		t.Fatalf("prompt length %d <= index_topk %d: the sparse path would never prune and this oracle would be vacuous", len(ids), sp.topK)
	}
	tr := glmOracleReference(t, m, ids, sp)

	// The sparse path must ACTUALLY have run. Without this the oracle would silently
	// degrade to a dense-causal witness wearing a SparseAttn name.
	if tr.dropped == 0 {
		t.Fatalf("the DSA indexer pruned nothing (index_topk=%d, prompt=%d) — this oracle is vacuous for SparseAttn", sp.topK, len(ids))
	}
	if want := len(ids) - 1 - sp.topK; tr.maxDropped < want {
		t.Fatalf("largest per-query prune = %d keys, want >= %d (position %d has a %d-key causal prefix and keeps %d)",
			tr.maxDropped, want, len(ids)-1, len(ids), sp.topK)
	}
	// A zero margin at the top-k boundary would mean the discrete decision is decided by
	// float noise rather than by semantics, and the comparison below would be flaky
	// rather than a witness.
	if !(tr.margin > 1e-9) {
		t.Fatalf("top-k boundary margin = %.3e (max |score| %.3e): the selection is precision-fragile in this fixture", tr.margin, tr.maxScore)
	}
	t.Logf("DSA sparsity: %d causal keys pruned across the stack, max %d per query; top-k margin %.3e (max |score| %.3e)",
		tr.dropped, tr.maxDropped, tr.margin, tr.maxScore)

	glmOracleSelectionWitness(t, m, sp, tr, len(ids))

	// Full-prefill Forward (batched residentMatMulBatch projections): every position.
	act := m.Forward(ids)
	for p := range ids {
		if d := cpuOracleMaxAbsDiff(act.Logits[p], tr.logits[p]); d > cpuOracleTol {
			t.Errorf("Forward logits pos %d: max|delta| vs reference = %.3e (tol %.0e)", p, d, cpuOracleTol)
		}
	}

	// Cached decode path (glmDsaKVCache, per-token GEMVs, per-position index step) —
	// a genuinely different instruction stream from Forward's batched GEMMs, compared
	// against the SAME reference.
	s := m.NewSession()
	pf := s.Prefill(ids)
	if d := cpuOracleMaxAbsDiff(pf, tr.logits[len(ids)-1]); d > cpuOracleTol {
		t.Errorf("Prefill last logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
	next := 13
	st := s.Step(next)
	ext := glmOracleReference(t, m, append(append([]int(nil), ids...), next), sp)
	if d := cpuOracleMaxAbsDiff(st, ext.logits[len(ids)]); d > cpuOracleTol {
		t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
}

// TestGLMDsaCPUNumericOracle is the GLM-5.2-DSA×cpu M4 witness at the attention_bias:false
// lineage (the DeepSeek-V3 default): the production forward — cacheless Forward AND the
// cached Prefill/Step decode path — must reproduce the independent HF-semantics reference
// on every position within cpuOracleTol, and must select the same sparse key sets. If
// this test reds, the honesty fence demotes the cell back to M3 (drop the covmatrix
// OracleInCI bit with it).
func TestGLMDsaCPUNumericOracle(t *testing.T) {
	runGLMDsaOracle(t, false)
}

// TestGLMDsaCPUNumericOracleAttentionBias is the same witness at the attention_bias:true
// lineage. HF's DeepseekV3Attention wires config.attention_bias onto exactly three
// projections — q_a_proj, kv_a_proj_with_mqa and o_proj — and production reads exactly
// those three through tensorOptional/addOptionalBias. A reference that ignored them, or
// a loader that added them at the wrong point in the norm order (the q_a bias is added
// BEFORE q_a_layernorm, the kv_a bias BEFORE the latent/rope split), stays green without
// this lineage.
func TestGLMDsaCPUNumericOracleAttentionBias(t *testing.T) {
	runGLMDsaOracle(t, true)
}

// TestGLMDsaCPUNumericOracleSparsityIsLoadBearing proves the sparse selection is not
// cosmetic: a reference that keeps EVERY causal key (index_topk = prompt length) — i.e.
// the dense-MLA path GLM-DSA would degrade to if the indexer were ignored — must diverge
// from production far beyond the tolerance. Without this, an implementation that silently
// attended the full prefix could still pass a logits comparison if the indexer happened
// to select nearly everything.
func TestGLMDsaCPUNumericOracleSparsityIsLoadBearing(t *testing.T) {
	m := newGLMOracleModel(t, false)
	sp := glmOracleSpecFor(false)
	ids := glmOracleIDs

	dense := sp
	dense.topK = len(ids) // no pruning at any position
	denseTr := glmOracleReference(t, m, ids, dense)
	if denseTr.dropped != 0 {
		t.Fatalf("the dense control still pruned %d keys — it is not a dense control", denseTr.dropped)
	}

	act := m.Forward(ids)
	var worst float64
	for p := range ids {
		if d := cpuOracleMaxAbsDiff(act.Logits[p], denseTr.logits[p]); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Fatalf("production matches a DENSE-causal reference within %.3e (tol %.0e): the DSA sparse selection is not load-bearing in this fixture",
			worst, cpuOracleTol)
	}
	t.Logf("dense-control divergence: max|delta| = %.3e (sparsity is load-bearing)", worst)
}

// TestGLMDsaCPUNumericOracleIsSensitive proves the comparison is non-vacuous: perturbing
// ONE raw f32 element of each distinct weight role must move the compared logits far
// beyond the tolerance. Every GLM-DSA-specific role is listed, including the indexer's
// four tensors (whose only effect is on the DISCRETE top-k selection — a perturbation
// there has to flip a selection to be visible at all, which is exactly the property that
// makes them worth listing) and the untied lm_head.
func TestGLMDsaCPUNumericOracleIsSensitive(t *testing.T) {
	for _, tc := range []struct {
		name   string
		tensor string
		// delta is the perturbation. The indexer roles feed a DISCRETE ranking, so a
		// perturbation must be large enough to actually flip a selection somewhere in
		// the 9-position prompt; the dense roles move the logits continuously.
		delta float32
		// elem is the f32 element index to perturb. It is 0 (the first element) for
		// every projection and norm, where element 0 is on the path for any input. The
		// embedding table is the exception: only the ROWS of the prompt's token ids are
		// ever read, so element 0 — row 0, which this prompt never names — is genuinely
		// dead weight and perturbing it is expected to move nothing. Pointing at a row
		// the prompt does use keeps the check a real sensitivity witness instead of
		// asserting something false about the kernel.
		elem int
	}{
		{"embed_tokens", "model.embed_tokens.weight", 0.5, glmOracleIDs[0] * 32},
		{"input_layernorm", "model.layers.0.input_layernorm.weight", 0.5, 0},
		{"q_a_proj", "model.layers.0.self_attn.q_a_proj.weight", 0.5, 0},
		{"q_a_layernorm", "model.layers.0.self_attn.q_a_layernorm.weight", 0.5, 0},
		{"q_b_proj", "model.layers.0.self_attn.q_b_proj.weight", 0.5, 0},
		{"kv_a_proj_with_mqa", "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", 0.5, 0},
		{"kv_a_layernorm", "model.layers.0.self_attn.kv_a_layernorm.weight", 0.5, 0},
		{"kv_b_proj", "model.layers.0.self_attn.kv_b_proj.weight", 0.5, 0},
		{"o_proj", "model.layers.0.self_attn.o_proj.weight", 0.5, 0},
		{"indexer_wq_b", "model.layers.0.self_attn.indexer.wq_b.weight", 5, 0},
		{"indexer_wk", "model.layers.0.self_attn.indexer.wk.weight", 5, 0},
		{"indexer_k_norm_weight", "model.layers.0.self_attn.indexer.k_norm.weight", 5, 0},
		{"indexer_k_norm_bias", "model.layers.0.self_attn.indexer.k_norm.bias", 5, 0},
		{"indexer_weights_proj", "model.layers.0.self_attn.indexer.weights_proj.weight", 5, 0},
		{"indexer_layer2_wq_b", "model.layers.2.self_attn.indexer.wq_b.weight", 5, 0},
		{"post_attention_layernorm", "model.layers.0.post_attention_layernorm.weight", 0.5, 0},
		{"mlp_gate_proj", "model.layers.0.mlp.gate_proj.weight", 0.5, 0},
		{"mlp_up_proj", "model.layers.0.mlp.up_proj.weight", 0.5, 0},
		{"mlp_down_proj", "model.layers.0.mlp.down_proj.weight", 0.5, 0},
		{"final_norm", "model.norm.weight", 0.5, 0},
		{"lm_head", "lm_head.weight", 0.5, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newGLMOracleModel(t, false)
			sp := glmOracleSpecFor(false)
			ids := glmOracleIDs
			ref := glmOracleReference(t, m, ids, sp)

			meta, ok := m.manifest[tc.tensor]
			if !ok {
				t.Fatalf("fixture tensor %q missing", tc.tensor)
			}
			at := meta.Offset + 4*tc.elem
			orig := math.Float32frombits(binary.LittleEndian.Uint32(m.raw[at:]))
			binary.LittleEndian.PutUint32(m.raw[at:], math.Float32bits(orig+tc.delta))

			act := m.Forward(ids)
			var worst float64
			for p := range ids {
				if d := cpuOracleMaxAbsDiff(act.Logits[p], ref.logits[p]); d > worst {
					worst = d
				}
			}
			if worst <= cpuOracleTol {
				t.Fatalf("perturbed fixture still within tolerance (max|delta|=%.3e) — the oracle is vacuous for this role", worst)
			}
		})
	}
}

// glmOracleAssertAxisLive mutates one spec axis to its "wrong but plausible" value and
// requires the reference to move beyond the tolerance — the anti-vacuity gate for the
// axes a size or scale choice could accidentally switch off.
func glmOracleAssertAxisLive(t *testing.T, m *Model, base [][]float32, mutated glmOracleSpec, label string) {
	t.Helper()
	got := glmOracleReference(t, m, glmOracleIDs, mutated)
	var worst float64
	for p := range base {
		if d := cpuOracleMaxAbsDiff(got.logits[p], base[p]); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Errorf("axis %q: mutating it moved the reference by only %.3e (<= tol %.0e) — this fixture does not witness it",
			label, worst, cpuOracleTol)
	}
}

// TestGLMDsaCPUNumericOracleAxesAreLive is the anti-vacuity gate for the GLM-5.2-DSA
// axes. Each is set to the value a plausible mis-implementation would produce, and the
// reference must move.
//
// Two axes are DELIBERATELY absent, and their absence is a finding rather than a gap:
// 1/sqrt(index_head_dim) and 1/sqrt(index_n_heads) both multiply every index score by
// the same positive constant, and relu(c*d) == c*relu(d) for c>0, so neither can change
// a RANKING. Since the scores are consumed only by dsaTopKIndices, no end-to-end witness
// — this one or any other — can observe them. Claiming them here would be exactly the
// vacuous "witnessed" bit the honesty fence forbids.
func TestGLMDsaCPUNumericOracleAxesAreLive(t *testing.T) {
	m := newGLMOracleModel(t, false)
	sp := glmOracleSpecFor(false)
	base := glmOracleReference(t, m, glmOracleIDs, sp).logits

	// The inv_freq denominator. Denominating by head_dim (12) instead of
	// qk_rope_head_dim (4) is the exact bug invFreqDenom's MLA branch exists to prevent:
	// the rotated RANGE stays right and only the frequency table is detuned, so it
	// survives every structural test.
	wrongDenom := sp
	wrongDenom.ropeDenom = 12
	glmOracleAssertAxisLive(t, m, base, wrongDenom, "inv_freq denominator = qk_rope_head_dim (not head_dim)")

	// The MLA rope convention: DeepSeek's interleaved de-interleave-then-rotate_half vs
	// plain rotate_half. Both consume the same slice and produce the same NORM, so a
	// conflation is invisible to any shape or magnitude check.
	wrongMLARope := sp
	wrongMLARope.mlaRopeInterleaved = false
	glmOracleAssertAxisLive(t, m, base, wrongMLARope, "MLA q_pe/k_pe use the INTERLEAVED rope")

	// The indexer rope convention is the OTHER one, in the same layer.
	wrongIdxRope := sp
	wrongIdxRope.indexerRopeInterleaved = true
	glmOracleAssertAxisLive(t, m, base, wrongIdxRope, "indexer rope prefix uses plain rotate_half")

	// index_topk: the sparsity itself.
	noSparsity := sp
	noSparsity.topK = len(glmOracleIDs)
	glmOracleAssertAxisLive(t, m, base, noSparsity, "index_topk sparsity")

	// IndexShare band placement: the checkpoint's pattern is full/shared/full, so the
	// wrong-but-plausible variant is a band that runs one layer too far — layer 2
	// REUSING layer 0's selection instead of computing its own. (The opposite mutation,
	// all-full, is not expressible against this checkpoint: a shared layer carries no
	// indexer tensors at all, so "layer 1 computes its own" has nothing to compute from.
	// That direction is witnessed structurally at the end of this test instead.)
	wrongShare := sp
	wrongShare.indexerTypes = []string{"full", "shared", "shared"}
	glmOracleAssertAxisLive(t, m, base, wrongShare, "indexer_types band placement (layer 2 is full, not shared)")

	// The MLA softmax scale denominator: qk_nope+qk_rope (10), not head_dim (12).
	wrongScale := sp
	wrongScale.attnScaleDenom = 12
	glmOracleAssertAxisLive(t, m, base, wrongScale, "MLA softmax scale = (qk_nope+qk_rope)**-0.5")

	// rope_theta.
	wrongTheta := sp
	wrongTheta.theta = 1e6
	glmOracleAssertAxisLive(t, m, base, wrongTheta, "rope_theta")

	// The other IndexShare direction, witnessed structurally: the shared layer must
	// reuse the previous full layer's set verbatim, and the two FULL layers must not
	// coincidentally agree (if they did, "shared" and "full" would be indistinguishable
	// here and the axis above would be vacuous). The end-to-end comparison in
	// runGLMDsaOracle is what binds these reference-side facts to production.
	tr := glmOracleReference(t, m, glmOracleIDs, sp)
	if !glmOracleSelEqual(tr.sel[0], tr.sel[1]) {
		t.Error("layer 1 (shared) did not reuse layer 0's selection — IndexShare is not wired")
	}
	if glmOracleSelEqual(tr.sel[0], tr.sel[2]) {
		t.Error("layer 2 (full) produced the SAME selection as layer 0 — shared and full are indistinguishable, so the IndexShare axis is unobservable in this fixture")
	}
}

func glmOracleSelEqual(a, b [][]int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}
